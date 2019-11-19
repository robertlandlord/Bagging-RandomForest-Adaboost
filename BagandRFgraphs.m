oob_bag = zeros(10);
test_bag = zeros(10);
oob_forest = zeros(10);
test_forest = zeros(10);

zip_train = readmatrix('zip_train.csv');
zip_test = readmatrix('zip_test.csv');
% 
% subsample = zip_train(find(zip_train(:,1) == 1 | zip_train(:,1) == 3),:);
% X_tr = subsample(:,2:257);
% y_tr = subsample(:,1);
% subsample = zip_test(find(zip_test(:,1) == 1 | zip_test(:,1) == 3),:);
% X_te = subsample(:,2:257);
% y_te = subsample(:,1);

    
subsample = zip_train(find(zip_train(:,1) == 3 | zip_train(:,1) == 5),:);
X_tr = subsample(:,2:257);
y_tr = subsample(:,1);
subsample = zip_test(find(zip_test(:,1) == 3 | zip_test(:,1) == 5),:);
X_te = subsample(:,2:257);
y_te = subsample(:,1);

    [oob_bag, testbag] = RandomForest_plotter(X_tr, y_tr, X_te, y_te, 200,floor(size(X_tr, 2)/3));
     %[oob_bag, testbag] = BaggedTrees_plotter(X_tr, y_tr, X_te, y_te, 200);
%     fprintf('oob %.4f\n', 
% fprintf('test %.4f\n', testErr_bagTrees);
% [oobErr_randForest, testErr_randForest] = RandomForest(X_tr, y_tr, X_te, y_te, 200, floor(size(X_tr, 2)/3));



X = 1:200;
plot(oob_bag)

