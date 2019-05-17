%% Clear
clear
clc
close all

%% Load data
load freyface.mat
X = double(X);
% show face
% showfreyface(X(:,1:100));

%% PCA
% eigenvalue decomposition
N = size(X, 2);
[Vun, Dun] = eig(X*X'/N);
[lambda_un, order] = sort(diag(Dun));
Vun = Vun(:, order);
Xctr = X - repmat(mean(X, 2), 1, N);
[Vctr, Dctr] = eig(Xctr*Xctr'/N);
[lambda_ctr, order] = sort(diag(Dctr));
Vctr = Vctr(:, order);

% plot
figure(1)
subplot(1,2,1)
plot(lambda_un)
title('eigenspectra without removing mean')
subplot(1,2,2)
plot(lambda_ctr,'g')
title('eigenspectra with removing mean')

%% Choose k
epsilon = 0.05; % the max proportion of abaondoned information
for k = 1:length(lambda_ctr)
    phi_k = Vctr(:,end-k+1:end);
    varX = sum(var(X,0,2));
    varphi = sum(var(X'*phi_k));
    res = varphi / varX;
    if res >= 1 - epsilon
        fprintf(1,'k = %d with %f variance reserved\n',k,res)
        break
    end
end

%% Top 16 eigenvectors
% show face
figure(2)
subplot(1,2,1)
showfreyface(Vun(:,end-15:end))
subplot(1,2,2)
showfreyface(Vctr(:,end-15:end))

%% 2D points
figure(3)
Yun = Vun(:,end-1:end)' * X;
plot(Yun(1,:), Yun(2,:), '.');
explorefreymanifold(Yun, X);
figure(4)
Yctr = Vctr(:,end-1:end)' * X;
plot(Yctr(1,:), Yctr(2,:), '.');
explorefreymanifold(Yctr, X);

%% Reconstruction
index = ceil(rand * N);
Xhat_un = pinv(Vun(:,end-1:end)') * Yun(:,index);
Xhat_ctr = pinv(Vctr(:,end-1:end)') * Yctr(:,index) + mean(X, 2);  % add back the mean

figure(5)
subplot(1,3,1)
showfreyface(Xhat_un)
subplot(1,3,3)
showfreyface(Xhat_ctr)
subplot(1,3,2)
showfreyface(X(:,index))

%% Reconstruction with noise
index = ceil(rand * N);
% add Gaussian white noise
Xn_ctr = Xctr(:,index) + randn(size(X,1),1);
Xn = Xn_ctr + mean(X, 2);
Yn = Vctr(:,end-1:end)' * Xn_ctr;
Xnhat = pinv(Vctr(:,end-1:end)') * Yn + mean(X, 2);
figure(6)
subplot(1,2,1)
showfreyface(Xn)
subplot(1,2,2)
showfreyface(Xnhat)
