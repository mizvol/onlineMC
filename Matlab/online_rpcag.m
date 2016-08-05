
Nx = 176;
Ny = 144;
N = 400;

start = 1001;
scale = 0.5;
Nx = Nx*scale; Ny = Ny*scale;
X = zeros(N,Ny*Nx);
in = 1;
for n=start:N+start-1
    X(in,:) = reshape(double(imresize(rgb2gray(imread(strcat(...
        '/Users/naumanshahid/Dropbox/code lap eigenmaps/hall_raw/airport',num2str(n)...
        ,'.bmp'),'bmp')),scale)),1,Nx*Ny);
    in = in + 1;
end
X = X/max(X(:));


%% run the online matrix completion with knn update
gamma =[1000 1000];
Ninit = 10;
batch = 10;
iters = 10;
a_final = [1:size(X,2)];

X = X';
T_gctp = X;
for i = 1 : 5
    a = randperm(size(T_gctp,2));
    X = X(:,a);
    T_gctp = T_gctp(:,a);
    T_gctp = gsp_frpcag_2g_online_knn(T_gctp, gamma(1),gamma(2), Ninit, batch, iters, 0,0);
    a_final = a_final(a);
end

[a_final,ind_final] = sort(a_final,'ascend');
X = X(:,ind_final)';
T_gctp = T_gctp(:,ind_final)';

%%
convert_video(X,T_gctp,X-T_gctp,X,T_gctp,Ny,Nx);



