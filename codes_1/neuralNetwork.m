%% clear
clear
clc
close all

%% load data
load digits
[num_train,d] = size(X);
nLabels = max(y);
num_valid = size(Xvalid,1);
num_test = size(Xtest,1);
% convert y into one-hot coding (1 for correct class and -1 for incorrect classes)
yExpanded = linearInd2Binary(y,nLabels);

%% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(num_train,1) X];
d = d + 1;

%% Apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(num_valid,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(num_test,1) Xtest];

%% Choose network structure
nHidden = [10];

%% Count number of parameters and initialize weights 'w'
nParams = d * nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams + nHidden(h-1) * nHidden(h);
end
nParams = nParams + nHidden(end) * nLabels;
w = randn(nParams,1);

%% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
num = 20;
error_train = zeros(num,1);
error_valid = zeros(num,1);
funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    i = ceil(rand*num_train);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
    
    if mod(iter,round(maxIter/num)) == 0
        t = iter/round(maxIter/num);
        yhat_train = MLPclassificationPredict(w,X,nHidden,nLabels);
        yhat_valid = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        error_train(t) = sum(yhat_train~=y)/num_train;
        error_valid(t) = sum(yhat_valid~=yvalid)/num_valid;
    end
end

%% Plot
figure(1)
hold on 
plot((1:num)*round(maxIter/num),error_train,'b')
plot((1:num)*round(maxIter/num),error_valid,'g')
xlabel('iterations')
ylabel('error rate')
axis([-inf,inf,0,1])
legend('train','valid');

%% Evaluate test error
% yhat_valid = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
% fprintf('Test error with final model = %f\n',sum(yhat_valid~=ytest)/num_test);
