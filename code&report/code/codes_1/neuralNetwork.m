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
yExpanded_valid = linearInd2Binary(yvalid,nLabels);

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
nHidden = 130;

%% Count number of parameters and initialize weights 'w'
nParams = d * nHidden(1);
for h = 2:length(nHidden)
    % add a bias for each layer
    nParams = nParams + (nHidden(h-1) + 1) * nHidden(h);
end
nParams = nParams + (nHidden(end) + 1) * nLabels;
w = randn(nParams,1);
w_pre = 0;

%% Train with stochastic gradient
maxIter = 1e5;
stepSize = 1e-3;
momentum = 0.9;
lambda = 1e-2;
batch = 10;
num = 20;
loss_train = zeros(num,1);
loss_valid = zeros(num,1);
funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/num)) == 0
        t = (iter - 1) / round(maxIter/num) + 1;
        py_train = MLPclassificationPredict(w,X,nHidden,nLabels);
        py_valid = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        loss_train(t) = sum(-log(py_train(y==1))) / num_train;
        loss_valid(t) = sum(-log(py_valid(yvalid==1))) / num_valid;
        [~,yhat_train] = max(py_train,[],2);
        [~,yhat_valid] = max(py_valid,[],2);
        error_train = sum(yhat_train~=y) / num_train;
        error_valid = sum(yhat_valid~=yvalid) / num_valid;
        fprintf('Training iteration = %d, train error = %.4f, validation error = %.4f\n',...
        iter-1,error_train,error_valid);
    end
    
    i = ceil(rand(batch,1) * num_train);
    [f,g,bias] = funObj(w,i);
    w_tmp = w;
    w = w - stepSize * g + momentum * (w - w_pre);
    w_pre = w_tmp;
    % Exclude bias when regularizing
    w(~bias) = w(~bias) - stepSize * lambda * w(~bias);
    
end
py_train = MLPclassificationPredict(w,X,nHidden,nLabels);
py_valid = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
[~,yhat_train] = max(py_train,[],2);
[~,yhat_valid] = max(py_valid,[],2);
error_train = sum(yhat_train~=y) / num_train;
error_valid = sum(yhat_valid~=yvalid) / num_valid;
fprintf('Training iteration = %d, train error = %.4f, validation error = %.4f\n',...
    iter,error_train,error_valid);

%% Plot
figure(1)
hold on 
plot((0:num-1)*round(maxIter/num),loss_train,'b')
plot((0:num-1)*round(maxIter/num),loss_valid,'g')
xlabel('iterations')
ylabel('cross entropy')
legend('train','valid');

%% Evaluate test error
py_test = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
[~,yhat_test] = max(py_test,[],2);
fprintf('Test error with final model = %.4f\n',sum(yhat_test~=ytest)/num_test);
