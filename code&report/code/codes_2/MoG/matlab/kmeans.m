% Performs clustering using the k-means algorithm.
%
% Input:
%
%   x(:,t) = the N-dimensional training vector for the tth training case
%   K = number of clusters
%   iters = number of iterations to apply
%
% Output:
%
%   means(:, c) = mean of the cth cluster


function [means] = kmeans(x, K, iters)
    % Initialize the means to K random training vectors
    perm = randperm(size(x, 2));
    means = x(:, perm(1:K));
    
    N = size(x, 2);
    for iter=1:iters
        dist = distmat(means', x');
        [~, class] = min(dist);
        for k=1:K
            means(:,k) = mean(x(:, class == k), 2);
        end
    end
