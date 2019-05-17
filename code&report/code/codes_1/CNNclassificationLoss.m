function [f,g,bias] = CNNclassificationLoss(w,X,y,filter,stride,padsize,nLabels)
[nInstances,nVars] = size(X);
width = ceil(sqrt(nVars));

% Form Weights
hiddenWeights = cell(length(filter),1);
ihiddenWeights = cell(length(filter),1);
offset = 1;
for h = 1:length(filter)
    hiddenWeights{h} = reshape(w(offset:offset+filter(h)^2-1),filter(h),filter(h));
    % inverse order for the convenience of back propagation
    ihiddenWeights{h} = reshape(hiddenWeights{h}(end:-1:1),filter(h),filter(h));
    offset = offset + filter(h)^2;
end
outputWeights = reshape(w(offset:end),[],nLabels);

% Initialize gradients and loss
if nargout > 1
    gHidden = cell(length(filter),1);
    for h = 1:length(filter)
        gHidden{h} = zeros(size(hiddenWeights{h}));
    end
    gOutput = zeros(size(outputWeights));
end
f = 0;

for i = 1:nInstances
    % Forward evaluation
    ip = cell(h,1);
    fp = cell(h,1);
    padding = padarray(reshape(X(i,:),width,width),[padsize(1),padsize(1)],0,'both');
    ip{1} = conv2(padding,hiddenWeights{1},'valid');
    fp{1} = tanh(ip{1});
    for h = 2:length(filter)
        padding = padarray(fp{h-1},[padsize(h),padsize(h)],0,'both');
        ip{h} = conv2(padding,hiddenWeights{h},'valid');
        fp{h} = tanh(ip{h});
    end
    yhat = [fp{end}(:);1]' * outputWeights;
    
    % Softmax layer
    yhat_shift_exp = exp(yhat - max(yhat,[],2));
    denom = sum(yhat_shift_exp,2) * ones(1,nLabels);
    py = yhat_shift_exp ./ denom;
    
    % Negative log-likelihood of the true label
    index = (y(i,:) == 1);
    f = f + sum(-log(py(index)));
    err = py;
    err(index) = py(index) - 1;
    
    % back propagation
    % Output Weights
    gOutput = gOutput + [fp{end}(:);1] * err;
    
    % Last Layer of Hidden Weights
    clear backprop
    % Remove extra column used for bias
    backprop = reshape(err * outputWeights(1:end-1,:)',size(fp{end}));
    backprop = backprop .* (sech(ip{end}).^2);
    
    % Other Hidden Layers
    for h = length(filter)-1:-1:1
        gHidden{h+1} = gHidden{h+1} + conv2(fp{h},backprop,'valid');
        backprop = conv2(backprop,ihiddenWeights{h+1}) .* (sech(ip{h}).^2);
    end
    
    % Input Weights
    ix = reshape(X(i,end:-1:1),width,width);
    gHidden{1} = gHidden{1} + conv2(ix,backprop,'valid');
end

% Compute average
f = f / nInstances;
gOutput = gOutput / nInstances;
for h = 1:length(h)
    gHidden{h} = gHidden{h} / nInstances;
end

% Put Gradient into vector
% And return a boolean vector which marks the position of bias
if nargout > 1
    g = [];
    for h = 1:length(filter)
        g = [g;gHidden{h}(:)];
    end
    bias = zeros(size(g));
    g = [g;gOutput(:)];
    bias_tmp = zeros(size(gOutput));
    bias_tmp(end,:) = 1;
    bias = [bias;bias_tmp(:)];
end
