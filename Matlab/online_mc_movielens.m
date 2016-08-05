
clear
close all

%% movielens
load 'ml-10M_small_40.mat';
X = full(ratings_small);
X = X(1:400,1:400);
[Xtrain, Xtest, mask, mask_train, mask_test] = split_train_test_mc(X,0.2);
%G = construct_rmse_graphs(Xtrain)


%% rescale
[Xtrain, y_lims_init] = rescale_mc(Xtrain,mask_train);

%% add noise

X_noisy = Xtrain + 0.2*randn(size(X,1),size(X,2)).*mask_train;

%% offline
% gamma =[0.5 0.5];
% T_gctp = gsp_fastmc_2g(X_noisy, mask_train, gamma(1),gamma(2),0,0);


%% run the online matrix completion with knn update
gamma =[0.5 0.5]*0.2;
Ninit = 20;
batch = 20;
iters = 100;
a_final = [1:size(X,2)];

T_gctp = X_noisy;
for i = 1 : 10
    a = randperm(size(T_gctp,2));
    mask_train = mask_train(:,a);
    mask_test = mask_test(:,a);
    X_noisy = X_noisy(:,a);
    T_gctp = T_gctp(:,a);
    Xtest = Xtest(:,a);
    T_gctp(logical(mask_train)) = X_noisy(logical(mask_train));
    T_gctp = gsp_fastmc_2g_online_knn2(T_gctp, mask_train, gamma(1),gamma(2), Ninit, batch, iters, 0,0);
    a_final = a_final(a);
end

[a_final,ind_final] = sort(a_final,'ascend');
mask_train = mask_train(:,ind_final);
mask_test = mask_test(:,ind_final);
X_noisy = X_noisy(:,ind_final);
T_gctp = T_gctp(:,ind_final);
Xtest = Xtest(:,ind_final);

T_gctp = lin_map(T_gctp, y_lims_init);
A = Xtest; B = T_gctp.*mask_test;
A(A == 0) = nan; B(B == 0) = nan;
disp(['RMSE:' num2str( rmse(A,B))])

figure; subplot(131); imagesc(X_noisy); title('actual noisy matrix');
subplot(132); imagesc(T_gctp); title(['online knn update: error =' num2str(rmse(A,B))]);

%% run the online matrix completion with full update
% gamma =[0.5 0.5]*0.5;
% Ninit = 100;
% batch = 1;
% iters = 1;
% T_gctp = gsp_fastmc_2g_online(X_noisy, mask_train, gamma(1),gamma(2), Ninit, batch, iters);
% 
% T_gctp = lin_map(T_gctp, y_lims_init); B = T_gctp.*mask_test;  
% A(A == 0) = nan; 
% disp(['RMSE:' num2str( rmse(A,B))])
% subplot(133); imagesc(T_gctp); title(['full online update: error =' num2str(rmse(A,B))]);

