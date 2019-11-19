function [oob_err, test_err] = RandomForest(X_tr, y_tr, X_te, y_te, numBags, m)
% RandomForest: Learns an ensemble of numBags CART decision trees using a random subset of
%               the features at each split on the input dataset and also plots the 
%               out-of-bag error as a function of the number of bags
%       Inputs:
%               X_tr: Training data
%               y_tr: Training labels
%               X_te: Testing data
%               y_te: Testing labels
%               numBags: Number of trees to learn in the ensemble
% 				m: Number of randomly selected features to consider at each split
% 				   (hint: read the "Name-Value Pair Arguments" part of the fitctree documentation)
%      Outputs: 
%	            oob_err: Out-of-bag classification error of the final learned ensemble
%               test_err: Classification error of the final learned ensemble on test data
%
% You may use "fitctree" but not "TreeBagger" or any other inbuilt bagging function


n = size(X_tr,1);
y_unique = unique(y_tr);
low_indices = find(y_tr == y_unique(1));
high_indices = find(y_tr == y_unique(2));
y_tr(low_indices) = 0;
y_tr(high_indices) = 1;

low_indices2 = find(y_te == y_unique(1));
high_indices2 = find(y_te == y_unique(2));
y_te(low_indices2) = 0;
y_te(high_indices2) = 1;

sizes = zeros(n,1);
sums = zeros(n,1);
test_sums = zeros(size(y_te));

for i=1:numBags
    indices = randsample(n,n,true);

    non_indices = 1:n;
    non_indices(indices)=0;
    non_indices = non_indices(find(non_indices~=0));
    
    non_indices_size = size(unique(non_indices));
    X_te1 = X_tr(non_indices,:);
    y_te1 = y_tr(non_indices);
    
    Xsample = X_tr(indices,:);
    ysample = y_tr(indices);
    
    tree = fitctree(Xsample, ysample, 'NumVariablesToSample', m);
    
%     OOB Data
    labels = predict(tree,X_te1);
    sums(non_indices) = sums(non_indices)+labels;
    sizes(non_indices) = sizes(non_indices)+1;
    
    
%     Test Data
    labels2 = predict(tree,X_te);
    test_sums = test_sums+labels2;
    

end

agg_preds = round(sums ./ sizes);
sum(abs(y_tr-agg_preds))
oob_err = sum(abs(y_tr-agg_preds))/n;

agg_preds2 = round(test_sums / numBags);
test_err = sum(abs(y_te-agg_preds2))/n;

end
