function [train_err, test_err] = AdaBoost(X_tr, y_tr, X_te, y_te, numTrees)
% AdaBoost: Implement AdaBoost with decision stumps as the weak learners. 
%           (hint: read the "Name-Value Pair Arguments" part of the "fitctree" documentation)
%   Inputs:
%           X_tr: Training data
%           y_tr: Training labels
%           X_te: Testing data
%           y_te: Testing labels
%           numTrees: The number of trees to use
%  Outputs: 
%           train_err: Classification error of the learned ensemble on the training data
%           test_err: Classification error of the learned ensemble on test data
% 
% You may use "fitctree" but not any inbuilt boosting function

y_unique = unique(y_tr);
low_indices = find(y_tr == y_unique(1));
high_indices = find(y_tr == y_unique(2));
y_tr(low_indices) = -1;
y_tr(high_indices) = 1;

low_indices2 = find(y_te == y_unique(1));
high_indices2 = find(y_te == y_unique(2));
y_te(low_indices2) = 0;
y_te(high_indices2) = 1;

iter_err = 1;
train_err = zeros(numTrees);
test_err = zeros(numTrees);
test_sums = 0;

n = size(X_tr,1);
numF = size(X_tr,2);
w = zeros(n,1)+1/n;

for t = 1:numTrees
    errors = zeros(numF,1);
    tree = fitctree(X_tr,y_tr,'SplitCriterion','deviance','MaxNumSplits',1, 'Weights', w);
    tree_pred = predict(tree, X_tr);
    
    eps = sum(w .* (tree_pred ~= y_tr));
    
    alpha = 0.5*log((1-eps)/eps);
    Z = sum(w.*exp(-alpha.*y_tr.*tree_pred));
    iter_err = iter_err * Z;
    train_err(t) = iter_err;
    
%     Test data
    labels = predict(tree,X_te);
    test_sums = test_sums+labels;
    tree_pred2 = sign(test_sums./numTrees);
    test_err(t) = sum(abs(y_te-tree_pred2))/(2*n);
    
    
    w2 = (w .* exp(-alpha.*y_tr.*tree_pred))./Z;
    w = w2;
end

train_err



end