function [py] = CNNclassificationPredict(w,X,filter,stride,padsize,nLabels)
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

py = zeros(nInstances,nLabels);
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
    py(i,:) = yhat_shift_exp ./ denom;
end

%y = binary2LinearInd(y);
