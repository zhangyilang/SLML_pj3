%% Clear
clear
clc
close all

%% Load data
load digits;
x = [train2, train3];

%% Training
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 

% parameters
K = 20;
iters = 10;
minVar = 0.01;
% Kmeans = false; % Train with original initialization
Kmeans = true;  % Train with kmeans initialization
[p,mu,vary,logProbtr] = mogEM(x,K,iters,minVar,true,Kmeans);
