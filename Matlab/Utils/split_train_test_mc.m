function [Xtrain, Xtest, mask, mask_train, mask_test] = split_train_test_mc(X,den)

s = size(X);
mask = zeros(size(X));
mask(X ~= 0) = 1;

[indr,indc] = find(mask(:) == 1);
sptest = full(logical(sprand(length(indr),1,den))).*indr;
sptest(sptest == 0) = [];
X = X(:);
Xtrain = X;
Xtrain(sptest) = 0;

Xtest = X - Xtrain;

Xtest = reshape(Xtest,size(X,1),size(X,2));
Xtrain = reshape(Xtrain,size(X,1),size(X,2));

mask_train = zeros(size(Xtrain));
mask_train(Xtrain ~= 0) = 1;

mask_test = zeros(size(Xtest));
mask_test(Xtest ~= 0) = 1;

Xtrain =  reshape(Xtrain,s(1),s(2));
Xtest =  reshape(Xtest,s(1),s(2));
mask_train =  reshape(mask_train,s(1),s(2));
mask_test =  reshape(mask_test,s(1),s(2));


end