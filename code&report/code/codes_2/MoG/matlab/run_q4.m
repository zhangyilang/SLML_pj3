%% Clear all and load data
clear
clc
close all
load digits;

%% Train and predict
% initialization
errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];
iters = 20;
minVar = 0.01;

for i = 1 : 4
    K = numComponent(i);
% Train a MoG model with K components for digit 2
%-------------------- Add your code here --------------------------------
    [p2,mu2,vary2,logProbtr2] = mogEM(train2,K,iters,minVar,false,true);

% Train a MoG model with K components for digit 3
%-------------------- Add your code here --------------------------------
    [p3,mu3,vary3,logProbtr3] = mogEM(train3,K,iters,minVar,false,true);

% Caculate the probability P(d=1|x) and P(d=2|x), 
% classify examples, and compute the error rate
% Hints: you may want to use mogLogProb function
%-------------------- Add your code here --------------------------------
    % predict
    trainPre22 = mogLogProb(p2,mu2,vary2,train2);
    trainPre32 = mogLogProb(p3,mu3,vary3,train2);
    trainPre23 = mogLogProb(p2,mu2,vary2,train3);
    trainPre33 = mogLogProb(p3,mu3,vary3,train3);
    validPre22 = mogLogProb(p2,mu2,vary2,valid2);
    validPre32 = mogLogProb(p3,mu3,vary3,valid2);
    validPre23 = mogLogProb(p2,mu2,vary2,valid3);
    validPre33 = mogLogProb(p3,mu3,vary3,valid3);
    testPre22 = mogLogProb(p2,mu2,vary2,test2);
    testPre32 = mogLogProb(p3,mu3,vary3,test2);
    testPre23 = mogLogProb(p2,mu2,vary2,test3);
    testPre33 = mogLogProb(p3,mu3,vary3,test3);
    % compute error
    errorTrain(i) = (mean(trainPre22 < trainPre32) + mean(trainPre23 > trainPre33))/2;
    errorValidation(i) = (mean(validPre22 < validPre32) + mean(validPre23 > validPre33))/2;
    errorTest(i) = (mean(testPre22 < testPre32) + mean(testPre23 > testPre33))/2;
end

%% Plot the error rate
%-------------------- Add your code here --------------------------------
figure(1)
plot(numComponent, errorTrain,'b-o')
hold on
plot(numComponent, errorValidation,'g-x')
plot(numComponent, errorTest,'r-+')
legend('train','validation','test')
axis([2,25,0,0.1])
xlabel('number of components')
ylabel('error rate')
