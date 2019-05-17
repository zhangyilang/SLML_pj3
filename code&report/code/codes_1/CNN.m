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

%% Standardize columns
[X,mu,sigma] = standardizeCols(X);

%% Apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xtest = standardizeCols(Xtest,mu,sigma);

%% Choose CNN parameter
filter = 3;
stride = 1;
padsize = 0;

%% Initialize weights
nParams = filter' * filter;
outsize = sqrt(d);
for h = 1:length(filter)
    outsize = (outsize + 2 * padsize(h) - filter(h)) / stride(h) + 1;
end
% fully connected layer with bias
nParams = nParams + (outsize^2 + 1) * nLabels;
w = randn(nParams,1);
w_pre = 0;

%% Train with stochastic gradient
maxIter = 1e5;
learningRate = 1e-3;
momentum = 0.9;
lambda = 1e-2;
batch = 10;
num = 20;
% loss_train = zeros(num,1);
% loss_valid = zeros(num,1);
funObj = @(w,i)CNNclassificationLoss(w,X(i,:),yExpanded(i,:),filter,stride,padsize,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/num)) == 0
        t = (iter - 1) / round(maxIter/num) + 1;
        py_train = CNNclassificationPredict(w,X,filter,stride,padsize,nLabels);
        py_valid = CNNclassificationPredict(w,Xvalid,filter,stride,padsize,nLabels);
%         loss_train(t) = sum(-log(py_train(y==1))) / num_train;
%         loss_valid(t) = sum(-log(py_valid(yvalid==1))) / num_valid;
        [~,py_train] = max(py_train,[],2);
        [~,py_valid] = max(py_valid,[],2);
        error_train = sum(py_train~=y) / num_train;
        error_valid = sum(py_valid~=yvalid) / num_valid;
        fprintf('Training iteration = %d, train error = %.4f, validation error = %.4f\n',...
        iter-1,error_train,error_valid);
    end
    
    i = ceil(rand(batch,1) * num_train);
    [f,g,bias] = funObj(w,i);
    w_tmp = w;
    w = w - learningRate * g + momentum * (w - w_pre);
    w_pre = w_tmp;
    % Exclude bias when regularizing
    w(~bias) = w(~bias) - learningRate * lambda * w(~bias);
    
end

%% test
py_train = CNNclassificationPredict(w,X,filter,stride,padsize,nLabels);
py_valid = CNNclassificationPredict(w,Xvalid,filter,stride,padsize,nLabels);
loss_train = sum(-log(py_train(y==1))) / num_train;
loss_valid = sum(-log(py_valid(yvalid==1))) / num_valid;
[~,py_train] = max(py_train,[],2);
[~,py_valid] = max(py_valid,[],2);
error_train = sum(py_train~=y) / num_train;
error_valid = sum(py_valid~=yvalid) / num_valid;
fprintf('Training iteration = %d, train error = %.4f, validation error = %.4f\n',...
    iter-1,error_train,error_valid);

%% Plot
% figure(1)
% hold on 
% plot((0:num-1)*round(maxIter/num),loss_train,'b')
% plot((0:num-1)*round(maxIter/num),loss_valid,'g')
% xlabel('iterations')
% ylabel('cross entropy')
% legend('train','valid');

%% Evaluate test error
% yhat_valid = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
% fprintf('Test error with final model = %f\n',sum(yhat_valid~=ytest)/num_test);
