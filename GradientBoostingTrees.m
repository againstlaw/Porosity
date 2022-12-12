clc
clear all


data = readtable('Data.xlsx');
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




rng(57214)    % For reproducibility

% Prepare Cross-Validation
n = length(yTrain);
cvp = cvpartition(n,'KFold',10);


% training vs testing index
numvalidsets = cvp.NumTestSets;

cvp_index = [];
for k = 1:numvalidsets   
    cvp_index = [cvp_index, cvp.training(k)];
end



% Get hyperparameters and display
Variables = hyperparameters('fitrensemble',dataTrain,'Porosity','Tree');
arrayfun(@disp, Variables);


% Adjust which hyperparameters to optimize

% Method
Variables(1).Optimize = false;

% NumLearningCycles
Variables(2).Optimize = true;

% LearnRate
Variables(3).Optimize = true;

% MinLeafSize
Variables(4).Optimize = true;

% MaxNumSplits
Variables(5).Range = [1,20];
Variables(5).Optimize = true;

% NumVariablesToSample
Variables(6).Optimize = false;


Mdl = fitrensemble(dataTrain,'Porosity','Method','LSBoost',...
    'OptimizeHyperparameters',Variables,'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus',...
    'CVPartition',cvp))


bestHyperparameters = Mdl.HyperparameterOptimizationResults.XAtMinEstimatedObjective;


yTrain_pred = predict(Mdl,dataTrain);

yTest_pred = predict(Mdl,dataTest);



% training dataset
P = Performance(yTrain,yTrain_pred)


% testing dataset

SSE = sum((yTest_pred - yTest).^2);
SST = sum((yTest - mean(yTest)).^2);
Rsquared = 1-SSE/SST


% prediction performance

N = length(yTest);

% Pearson correlation coefficient (R)
R = corr(yTest,yTest_pred)

% coefficient of determination (R^2)
RR = corr(yTest,yTest_pred)^2

% root-mean squared error
RMSE = sqrt(sum((yTest-yTest_pred).^2)/N)

% mean absolute error
MAE = sum(abs(yTest-yTest_pred))/N

% mean absolute percentage error (%)
MAPE = sum(abs((yTest-yTest_pred)./yTest))/N*100




figure;
plot(yTest,yTest_pred,'bo');
hold on
plot(yTest,yTest,'r','linewidth',2);
xlabel('Observed Response','FontSize',12);
ylabel('Fitted Response','FontSize',12);
title('Porosity','FontSize',12);



figure;
plot(yTest,yTest_pred,'bo','markersize',5,'markeredgecolor','b','markerfacecolor','b');
hold on
plot(yTrain,yTrain_pred,'ro','markersize',5);
plot(y,y,'k','linewidth',2);
xlabel('Observed Response','FontSize',12);
ylabel('Fitted Response','FontSize',12);
title('Porosity','FontSize',12);
legend({'Test','Train'},'FontSize',12,'Location','Southeast')



imp = predictorImportance(Mdl);

[myimp, ind] = sort(imp,'descend');

figure;
bar(myimp);
title('Predictor Importance Estimates');
ylabel('Importance');
xlabel('Predictors');
h = gca;
h.XTickLabel = Mdl.PredictorNames(ind);
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';





view(Mdl.Trained{10},'mode','graph')   % graphic description



% mse of resubstitution
% mean((yTrain_pred-yTrain).^2)
L = resubLoss(Mdl,'mode','cumulative');

figure;
plot(L);
xlabel('Number of trees');
ylabel('Cross-validated MSE');




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


figure;
plotPartialDependence(Mdl,{'WB','CuringDays'})


flyash = Tbl.FlyAsh;
slag = Tbl.Slag;


% OPC
index = find(flyash==0 & slag==0);
OPC = Tbl(index,:);


% GGBS concrete
index2 = find(flyash==0 & slag > 0);
GGBS = Tbl(index2,:);


% Fly ash concrete
index3 = find(flyash>0 & slag==0);
FAC = Tbl(index3,:);


figure;
plotPartialDependence(Mdl,'FlyAsh',FAC)


% figure;
% pt = linspace(min(FAC.CuringDays),max(FAC.CuringDays),50)';
% pt2 = linspace(min(FAC.FlyAsh),max(FAC.FlyAsh),50)';
% ax = plotPartialDependence(Mdl,{'FlyAsh','CuringDays'},FAC,'QueryPoints',{pt2,pt});
% view(140,30) % Modify the viewing angle

figure;
plotPartialDependence(Mdl,{'FlyAsh','CuringDays'},FAC)


figure;
plotPartialDependence(Mdl,'Slag',GGBS)


figure;
ax = plotPartialDependence(Mdl,{'Slag','CuringDays'},GGBS);




% print(figure(1), '-dtiff', 'myfigure.tiff','-r600');





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

