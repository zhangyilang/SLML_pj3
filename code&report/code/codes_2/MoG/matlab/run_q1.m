%% Clear
clear
clc
close all

%% Load data
load digits.mat

%% Plot the digits as images
index2 = ceil(rand * size(train2, 2));
index3 = ceil(rand * size(train3, 2));
figure(1)
colormap(gray)
subplot(1,2,1)
imagesc(reshape(train2(:,index2),16,16))
subplot(1,2,2)
imagesc(reshape(train3(:,index3),16,16))

%% Train for digits 2
K = 2;
iters = 10;
minVar = 0.01;
[p2,mu2,vary2,logProbtr2] = mogEM(train2,K,iters,minVar,true);

% plot
figure(3)
colormap(gray)
subplot(2,2,1)
imagesc(reshape(mu2(:,1),16,16))
title('mean (k=1)')
subplot(2,2,2)
imagesc(reshape(mu2(:,2),16,16))
title('mean (k=2)')
subplot(2,2,3)
imagesc(reshape(vary2(:,1),16,16))
title('variance (k=1)')
subplot(2,2,4)
imagesc(reshape(vary2(:,2),16,16))
title('variance (k=2)')

% output
fprintf(1,'\x03C0_1 = %.4f, \x03C0_2 = %.4f\n',p2(1),p2(2))
%% Train for digits 3
K = 2;
iters = 20;
minVar = 0.01;
[p3,mu3,vary3,logProbtr3] = mogEM(train3,K,iters,minVar,true,true);

% Plot
figure(4)
colormap(gray)
subplot(2,2,1)
imagesc(reshape(mu3(:,1),16,16))
title('mean (k=1)')
subplot(2,2,2)
imagesc(reshape(mu3(:,2),16,16))
title('mean (k=2)')
subplot(2,2,3)
imagesc(reshape(vary3(:,1),16,16))
title('variance (k=1)')
subplot(2,2,4)
imagesc(reshape(vary3(:,2),16,16))
title('variance (k=2)')

% output
fprintf(1,'\x03C0_1 = %.4f, \x03C0_2 = %.4f\n',p3(1),p3(2))
