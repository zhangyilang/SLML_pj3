function [py] = MLPclassificationPredict(w,X,nHidden,nLabels)
[nInstances,nVars] = size(X);

% Dropout
% w = 0.5 * w;

% Form Weights
offset = nVars*nHidden(1);
inputWeights = reshape(w(1:offset),nVars,nHidden(1));
for h = 2:length(nHidden)
    hiddenWeights{h-1} = reshape(w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)),...
        nHidden(h-1)+1,nHidden(h));
    offset = offset + (nHidden(h-1) + 1) * nHidden(h);
end
outputWeights = w(offset+1:offset+(nHidden(end)+1)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end)+1,nLabels);

% Compute Output (fully vectorized)
% Add bias
ip{1} = [X * inputWeights,zeros(nInstances,1)]; % Add zeros just for convenience
fp{1} = tanh(ip{1});
fp{1}(:,end) = 1;
for h = 2:length(nHidden)
    % Add bias
    ip{h} = [fp{h-1} * hiddenWeights{h-1},zeros(nInstances,1)];
    fp{h} = tanh(ip{h});
    fp{h}(:,end) = 1;
end
yhat = fp{end} * outputWeights;

% Softmax layer
yhat_shift_exp = exp(yhat - max(yhat,[],2));
denom = sum(yhat_shift_exp,2) * ones(1,nLabels);
py = yhat_shift_exp ./ denom;

%y = binary2LinearInd(y);
