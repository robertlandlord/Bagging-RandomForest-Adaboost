zip_train = readmatrix('zip_train.csv');
zip_test = readmatrix('zip_test.csv');

subsample = zip_train(find(zip_train(:,1) == 5 | zip_train(:,1) == 3),:);
X_tr = subsample(:,2:257);
y_tr = subsample(:,1);
subsample = zip_test(find(zip_test(:,1) == 5 | zip_test(:,1) == 3),:);
X_te = subsample(:,2:257);
y_te = subsample(:,1);

[oobErr_ada, testErr_ada] = AdaBoost(X_tr, y_tr, X_te, y_te, 200);
X = 1:200;

a1 = plot(oobErr_ada(:,1), 'Color','b');
hold on;
a2 = plot(testErr_ada(:,1), 'Color','r');
legend('Training Error', 'Test Error');

oobErr_ada(200)
testErr_ada(200)