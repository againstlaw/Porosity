clc
clear all


data = readtable('Data.xlsx');


% categorical predictor
data.Curing = categorical(data.Curing);

Tbl = data(:,{'WB','Binder','FlyAsh','Slag',...
    'Superplasticizer','Aggregate','Curing','CuringDays',...
    'Porosity'});


predictornames = Tbl.Properties.VariableNames(1:end-1);


y = Tbl.Porosity;



idxTrain = data.Training;
idxTest = ~idxTrain;


dataTrain = Tbl(idxTrain,:);
yTrain = y(idxTrain);

dataTest = Tbl(idxTest,:);
yTest = y(idxTest);


% Categorical variable flag
M = length(predictornames);
isCategorical = zeros(M,1); 
isCategorical(find(ismember(predictornames,'Curing'))) = 1;



maxMinLS = 20;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');

numPTS = optimizableVariable('numPTS',[1,size(dataTrain,2)-1],'Type','integer');

hyperparametersRF = [minLS; numPTS];



rng(58111);

results = bayesopt(@(params)oobErrRF(params,dataTrain),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);



bestHyperparameters = results.XAtMinObjective




Mdl = TreeBagger(300,dataTrain,'Porosity','Method','regression',...
    'OOBPrediction','on','OOBPredictorImportance','on',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS);



imp = Mdl.OOBPermutedPredictorDeltaError;

[myimp, ind] = sort(imp,'descend');

figure;
bar(myimp);
title('Out-of-Bag Permuted Predictor Importance Estimates');
ylabel('Importance');
xlabel('Predictors');
h = gca;
h.XTickLabel = Mdl.PredictorNames(ind);
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';




yTrain_pred = predict(Mdl,dataTrain);

yTest_pred = predict(Mdl,dataTest);


% training dataset
P = Performance(yTrain,yTrain_pred)


% testing dataset

SSE = sum((yTest_pred - yTest).^2);
SST = sum((yTest - mean(yTest)).^2);
Rsquared = 1-SSE/SST;


% prediction performance

N = length(yTest);

% Pearson correlation coefficient (R)
R = corr(yTest,yTest_pred);

% coefficient of determination (R^2)
RR = corr(yTest,yTest_pred)^2;

% root-mean squared error
RMSE = sqrt(sum((yTest-yTest_pred).^2)/N);

% mean absolute error
MAE = sum(abs(yTest-yTest_pred))/N;

% mean absolute percentage error (%)
MAPE = sum(abs((yTest-yTest_pred)./yTest))/N*100;



myPerformance = table(Rsquared,RMSE,MAE,MAPE)



figure;
plot(yTest,yTest_pred,'bo');
hold on
plot(yTest,yTest,'r','linewidth',2);
xlabel('Observed Response','FontSize',12);
ylabel('Fitted Response','FontSize',12);
title('Porosity (%)','FontSize',12);



figure;
plot(yTest,yTest_pred,'bo','markersize',5,'markeredgecolor','b','markerfacecolor','b');
hold on
plot(yTrain,yTrain_pred,'ro','markersize',5);
plot(y,y,'k','linewidth',2);
xlabel('Observed Response','FontSize',12);
ylabel('Fitted Response','FontSize',12);
title('Porosity (%)','FontSize',12);
legend({'Test','Train'},'FontSize',12,'Location','Southeast')




view(Mdl.Trees{100},'mode','graph')   % graphic description


figure;
plot(oobError(Mdl))




% Partial Dependence Plot (PDP)

figure;
subplot(2,2,1)
plotPartialDependence(Mdl,'CuringDays')

subplot(2,2,2)
plotPartialDependence(Mdl,'Binder')

subplot(2,2,3)
plotPartialDependence(Mdl,'WB')

subplot(2,2,4)
plotPartialDependence(Mdl,'Aggregate')










function P = Performance(observed,fitted)

% prediction performance

N = length(observed);

% Pearson correlation coefficient (R)
R = corr(observed,fitted);

% coefficient of determination (R^2)
SSE = sum((fitted - observed).^2);
SST = sum((observed - mean(observed)).^2);
Rsquared = 1-SSE/SST;

% root-mean squared error
RMSE = sqrt(sum((observed-fitted).^2)/N);

% mean absolute percentage error (%)
MAPE = sum(abs((observed-fitted)./observed))/N*100;

% create table
P = table(Rsquared,RMSE,MAPE);

end






function oobErr = oobErrRF(params,X)
%   oobErrRF Trains random forest of 300 trees and estimates out-of-bag MSE error
%   X is a table of training data
%   params is an array of OptimizableVariable objects corresponding to
%   minLS = the minimum leaf size 
%   numPTS = number of predictors to sample at each split
randomForest = TreeBagger(300,X,'Porosity','Method','R','OOBPrediction','on',...
    'MinLeafSize',params.minLS,...
    'NumPredictorstoSample',params.numPTS);

oobErr = oobError(randomForest,'Mode','Ensemble');

end



