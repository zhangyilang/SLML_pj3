function [f,g,bias] = MLPclassificationLoss(w,X,y,nHidden,nLabels)
[nInstances,nVars] = size(X);

% Dropout
% mask = (rand(size(w)) > 0.5);
% w(mask) = 0;

% Form Weights
offset = nVars*nHidden(1);
inputWeights = reshape(w(1:offset),nVars,nHidden(1));
for h = 2:length(nHidden)
    % The last row filled with bias
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

% % Squared error
% relativeErr = yhat - y;
% f = relativeErr(:)' * relativeErr(:);
% err = 2 * relativeErr;

% Softmax layer
yhat_shift_exp = exp(yhat - max(yhat,[],2));
denom = sum(yhat_shift_exp,2) * ones(1,nLabels);
py = yhat_shift_exp ./ denom;

% Negative log-likelihood of the true label
index = (y == 1);
f = sum(-log(py(index)));
err = py;
err(index) = py(index) - 1;

if nargout > 1
    
    % Output Weights
    gOutput = fp{end}' * err;
    
    if length(nHidden) > 1
        % Last Layer of Hidden Weights
        clear backprop
        backprop = err * outputWeights' .* (sech(ip{end}).^2);
        backprop = backprop(:,1:end-1); % Remove extra column used for bias
        
        % Other Hidden Layers
        for h = length(nHidden)-2:-1:1
            backprop = backprop * hiddenWeights{h+1}' .* (sech(ip{h+1}).^2);    
            backprop = backprop(:,1:end-1); % Remove extra column used for bias
            gHidden{h} = fp{h}' * backprop;
        end
        
        % Input Weights
        backprop = backprop * hiddenWeights{1}' .* (sech(ip{1}).^2);
        backprop = backprop(:,1:end-1); % Remove extra column used for bias
        gInput = X' * backprop;
    else
        % Input Weights
        gInput = X' * (err * outputWeights' .* (sech(ip{end}).^2));
        gInput = gInput(:,1:end-1);  % Remove extra column used for bias
    end
end

% Put Gradient into vector
% And return a boolean vector which marks the position of bias
if nargout > 1
    g = zeros(size(w));
    offset = nVars * nHidden(1);
    g(1:offset) = gInput(:);
    bias =[];
    bias_tmp = zeros(size(inputWeights));
    bias_tmp(end,:) = 1;
    bias = [bias;bias_tmp(:)];
    for h = 2:length(nHidden)
        g(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)) = gHidden{h-1};
        offset = offset + (nHidden(h-1) + 1) * nHidden(h);
        bias_tmp = zeros(size(hiddenWeights{h-1}));
        bias_tmp(end,:) = 1;
        bias = [bias;bias_tmp(:)];
    end
    g(offset+1:offset+(nHidden(end)+1)*nLabels) = gOutput(:);
    bias_tmp = zeros(size(outputWeights));
    bias_tmp(end,:) = 1;
    bias = [bias;bias_tmp(:)];
end
