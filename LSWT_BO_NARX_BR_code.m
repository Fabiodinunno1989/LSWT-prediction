%% BO-NARX-BR code for Lake Surface Water Temperature (LSWT)

%% Data loading
cd('Insert here the data directory')

% Load here your data

clear opts

%% Data must be divided in exogenous input ("input"), in this case represented by the only air temperatur Tair, and LSWT ("output")
input = Tair;
output = LSWT;

%% Bayesian Optimization section - Example for the First lake --> (:,1)

input_data = input(:,1);
target_data = output(:,1);

% Define the objective function for Bayesian optimization
% This function evaluates the NARX model and returns a loss value
objectiveFunction = @(params) trainNARXAndGetLoss(params, input_data, target_data);

% Define the hyperparameter search space
vars = [
    optimizableVariable('numHiddenUnits', [10, 100], 'Type', 'integer'),
    optimizableVariable('lags', [1, 14], 'Type', 'integer'),
    optimizableVariable('delay', [1, 14], 'Type', 'integer')
];

% Run Bayesian optimization
results = bayesopt(objectiveFunction, vars);

% Get the best hyperparameters from the optimization results
bestParams = results.XAtMinObjective;

% Define the function to train NARX model and compute loss
function loss = trainNARXAndGetLoss(params, input_data, target_data)
    % Extract hyperparameters
    numHiddenUnits = params.numHiddenUnits;
    lags = params.lags;
    delay = params.delay;

    % Create NARX model
    narx_net = narxnet(1:lags, 1:numHiddenUnits, delay);

    % Reshape input and output into sequences
    input_seq = con2seq(input_data');
    target_seq = con2seq(target_data');

    % Prepare data for training
    [X, Xinit, Xfinal, T] = preparets(narx_net, input_seq, {}, target_seq);

    % Train NARX model
    narx_net = train(narx_net, X, T, Xinit, Xfinal);

    % Make predictions using the trained model
    y_pred = sim(narx_net, X, Xinit);

    % Calculate loss (MSE)
    loss = mse(cell2mat(T) - cell2mat(y_pred));
end

%% Hyperparameters tuning based on the BO algorithm

Delay = bestParams.delay;
inputDelays = 1:Delay; % lagged values of the exogenous inputs variables
feedbackDelays = 1:Delay; % lagged values of LSWT
hiddenLayerSize = bestParams.numHiddenUnits;

%% Selection of the backpropagation algorithm
trainFcn = 'trainbr';  % Bayesian Regularization backpropagation - Other backpropagations are Levenberg-Marquardt ('trainlm') and Scaled conjugate gradient ('trainscg')

%% The NARX process is terminated when one of the following conditions is met
trainParam.epochs  = 1000; % maximum number of epochs;
trainParam.mu_max = 1.0000e+10; % Maximum value for the LM adjustment parameter;
trainParam.min_grad = 1.0000e-07; % attaining an error gradient below a specified threshold;

%% Modeling

for i = 1:size(input,2) % i-th lakes
    
    % Open-loop
    X = tonndata(input(:,i),false,false);
    T = tonndata(output(:,i),false,false);
    net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
    [x,xi,ai,t] = preparets(net,X,{},T);
    net.divideParam.trainInd = 1:7304; % Training for the example from 1987 to 2006 
    net.divideParam.valInd = 7305:9880;  
    net.divideParam.testInd = 9881:length(input); % Remaining dataset for validation and testing
    net.trainParam.showWindow = false;
    [net,tr] = train(net,x,t,xi,ai);
    y = net(x,xi,ai);
    e = gsubtract(t,y);
    performance = perform(net,t,y);

    % Closed-loop
    netc = closeloop(net);
    netc.name = [net.name ' - Closed Loop'];
    [xc,xic,aic,tc] = preparets(netc,X,{},T);
    yc = netc(xc,xic,aic);
    closedLoopPerformance = perform(net,tc,yc);

    % One Step Ahead prediction
    nets = removedelay(net);
    nets.name = [net.name ' - Predict One Step Ahead'];
    [xs,xis,ais,ts] = preparets(nets,X,{},T);
    ys = nets(xs,xis,ais);
    stepAheadPerformance = perform(nets,ts,ys);

    % Time series of the measured and predicted values for each lake
    for j = 1:length(ts)
        measured(j,i) = ts{j};
        predicted(j,i) = ys{j};
    end

    disp(i)
    
end