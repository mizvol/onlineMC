function [Lr, Gu, Gv, Eval] = gsp_fastmc_2g_online_knn2(X, M, gamma_u, gamma_v, Ninit, batch, iters, Gu, Gv, param)
%GSP_FASTMC_2G_ONLINE_KNN2 Fast online matrix completion on 2 graphs with
%the batch updates of the users and movies
%   Usage: [Lr] = gsp_fastmc_2g_online_knn2(X, M, gamma_u, gamma_v, Ninit, batch, iters, Gu, Gv)
%          [Lr] = gsp_fastmc_2g_online_knn2(X, M, gamma_u, gamma_v,  Ninit, batch, iters, Gu, Gv, param)
%          [Lr, Gu, Gv] = gsp_fastmc_2g_online_knn2( ... );
%          [Lr, Gu, Gv, Eval] = gsp_fastmc_2g_online_knn2( ... );
%
%   Input parameters:
%       X       : Input data (matrix of double)
%       M       : the mask operator
%       gamma_u  : Regularization parameter for  graph of users
%       gamma_v  : Regularization parameter for  graph of videos
%       Ninit   : size of the initial batch
%       iters   : number of iterations for the optimization of small
%       batches
%       param   : Optional optimization parameters
%
%   Output Parameters:
%       Lr      : Low-rank part of the data
%       Gu      : the graph between users of X
%       Gv      : the graph between videos of X
%       Eval    : the values of objective function just to see if the
%       solution is going down (converging), not diverging
%
%   This function computes a low rank approximation of the incomplete data stored in
%   *X* by solving an optimization problem in online manner:
%
%   .. argmin_Lr  ||M(X - Lr)||^2_2  + gamma_u tr( Lr Lu Lr^t) + gamma_v tr( Lr^t Lv Lr)
%
%   .. math:: argmin_Lr  ||M(X - Lr)||^2_2 + gamma_u tr( Lr Lu Lr^t) + gamma_v tr( Lr^t Lv Lr)
%
%
%   If $0$ is given for *G*, the corresponding graph will
%   be computed internally. The graph construction can be tuned using the
%   optional parameter: *param.paramnn*.
%
%
%   If the number of output argument is greater than 1, The function, will
%   additionally return the graphs Gu and Gv as well and the objective
%   function evaluation 'Eval'
%
%   This function uses the UNLocBoX to be working.
%
% Author: Nauman Shahid
% Date  : 15th June 2016
% references: see "Matrix Completion on Graphs" by Vassilis Kalofolias

%% Optional parameters

if nargin<10
    param = struct;
end

if ~isfield(param,'paramnn'),  param.paramnn = struct; end
if ~isfield(param.paramnn,'k'), param.paramnn.k = 10; end
if ~isfield(param.paramnn,'use_flann'), param.paramnn.use_flann = 0; end
if ~isfield(param.paramnn,'use_l1'), param.paramnn.use_l1 = 0; end

if ~isfield(param,'param_solver'), param.param_solver = struct; end
if ~isfield(param.param_solver, 'verbose'), param.param_solver.verbose = 2; end
if ~isfield(param.param_solver, 'maxit'), param.param_solver.maxit = 100; end
if ~isfield(param.param_solver, 'tol'), param.param_solver.tol = 1e-5; end
if ~isfield(param.param_solver, 'stopping_criterion'), param.param_solver.stopping_criterion = 'rel_norm_primal'; end

if ~isstruct(Gu)
    Gu = gsp_nn_graph(X, param.paramnn);
end

if ~isstruct(Gv)
    Gv = gsp_nn_graph(X', param.paramnn);
end

if ~isfield(Gu,'lmax')
    Gu = gsp_estimate_lmax(Gu);
end

if ~isfield(Gv,'lmax')
    Gv = gsp_estimate_lmax(Gv);
end
%% Optimization
%% initial small batch part

% solve the initial batch part
f1.grad = @(x,T) 2*M(1:Ninit,1:Ninit).*(x-X(1:Ninit,1:Ninit));
f1.beta = 2;
f1.eval = @(x) norm(M(1:Ninit,1:Ninit).*(x-X(1:Ninit,1:Ninit)),'fro');

G1_small.L = Gu.L(1:Ninit,1:Ninit);
G1_small.lmax = Gu.lmax;
G2_small.L = Gv.L(1:Ninit,1:Ninit);
G2_small.lmax = Gv.lmax;

f3.grad = @(x) gamma_u*2*G1_small.L*x;
f3.eval = @(x) gamma_u*sum(gsp_norm_tik(G1_small,x));
f3.beta = 2*gamma_u*G1_small.lmax;

f4.grad = @(x) gamma_v*(2*x*G2_small.L);
f4.eval = @(x) gamma_v*sum(gsp_norm_tik(G2_small,x'));
f4.beta = 2*gamma_v*G2_small.lmax;

Lr = solvep(X(1:Ninit,1:Ninit),{f1,f3,f4},param.param_solver);
param.param_solver.maxit = iters;
%% online update part
tempr_size = [];
tempc_size = [];
Eval = [];
for i = 1 : batch : size(X,2) - Ninit
    
    %%  update the new rows
    indicesr_knn = [];
    for j = 1 : batch
        [temp, temp_ind] = sort(full(Gu.W(1:Ninit+i+j-1,Ninit+i-1)),'descend');
        indicesr_knn = union(indicesr_knn,temp_ind(temp~=0));
    end
    indicesr_knn = union(indicesr_knn,[Ninit+i:Ninit+i+batch-1]);
    tempr_size = [tempr_size length(indicesr_knn)];
    
    G1new.L = Gu.L(indicesr_knn,indicesr_knn);
    G1new.lmax = Gu.lmax;
    
    G2new.L = Gv.L(1:Ninit+i-1,1:Ninit+i-1);
    G2new.lmax = Gv.lmax;
    
    f1.grad = @(x,T) 2*M(indicesr_knn,1:Ninit+i-1).*(x-X(indicesr_knn,1:Ninit+i-1));
    f1.beta = 2;
    f1.eval = @(x) norm(M(indicesr_knn,1:Ninit+i-1).*(x-X(indicesr_knn,1:Ninit+i-1)),'fro');
    
    f3.grad = @(x) gamma_u*2*G1new.L*x;
    f3.eval = @(x) gamma_u*sum(gsp_norm_tik(G1new,x));
    f3.beta = 2*gamma_u*G1new.lmax;
    
    f4.grad = @(x) gamma_v*(2*x*G2new.L);
    f4.eval = @(x) gamma_v*sum(gsp_norm_tik(G2new,x'));
    f4.beta = 2*gamma_v*G2new.lmax;
    
%     Lrtemp = solvep([Lr(indicesr_knn(1:end-batch),1:Ninit+i-1)...
%         ; X(Ninit+i:Ninit+i+batch-1,1:Ninit+i-1)],{f1,f3,f4},param.param_solver);
    
    Lrtemp = solvep(X(indicesr_knn,1:Ninit+i-1),{f1,f3,f4},param.param_solver);
    
    if ~isempty(indicesr_knn)
        Lr(indicesr_knn(indicesr_knn < Ninit+i),:) = Lrtemp(1:end-batch,:);
    end
    Lr = [Lr ; Lrtemp(end-batch+1:end,:)];
    
    %%
    % update the new columns
    indicesc_knn = [];
    for j = 1 : batch
        [temp, temp_ind] = sort(full(Gv.W(1:Ninit+i-1,Ninit+i+j-1)),'descend');
        indicesc_knn = union(indicesc_knn,temp_ind(temp~=0));
    end
    indicesc_knn = union(indicesc_knn,[Ninit+i:Ninit+i+batch-1]);
    tempc_size = [tempc_size length(indicesc_knn)];
    
    G2new.L = Gv.L(indicesc_knn,indicesc_knn);
    G2new.lmax = Gv.lmax;
    
    G1new.L = Gu.L(1:Ninit+i+batch-1,1:Ninit+i+batch-1);
    G1new.lmax = Gu.lmax;
    
    % solve the optimization
    f1.grad = @(x,T) 2*M(1:Ninit+i+batch-1,indicesc_knn).*(x-X(1:Ninit+batch+i-1,indicesc_knn));
    f1.beta = 2;
    f1.eval = @(x) norm(M(1:Ninit+i+batch-1,indicesc_knn).*(x-X(1:Ninit+i+batch-1,indicesc_knn)),'fro');
    
    f3.grad = @(x) gamma_u*2*G1new.L*x;
    f3.eval = @(x) gamma_u*sum(gsp_norm_tik(G1new,x));
    f3.beta = 2*gamma_u*G1new.lmax;
    
    f4.grad = @(x) gamma_v*(2*x*G2new.L);
    f4.eval = @(x) gamma_v*sum(gsp_norm_tik(G2new,x'));
    f4.beta = 2*gamma_v*G2new.lmax;
    
%     Lrtemp = solvep([Lr(:,indicesc_knn(1:end-batch)) X(1:Ninit+i+batch-1,Ninit+i:Ninit+i+batch-1)]...
%         ,{f1,f3,f4},param.param_solver);
     Lrtemp = solvep(X(1:Ninit+i+batch-1,indicesc_knn),{f1,f3,f4},param.param_solver);
    
    if ~isempty(indicesc_knn)
        Lr(:,indicesc_knn(indicesc_knn < Ninit+i)) = Lrtemp(:,1:end-batch);
    end
    
    Lr = [Lr Lrtemp(:,end-batch+1:end)];
    
    
end

Eval = [Eval norm(M.*(Lr-X),'fro')+gamma_u*sum(gsp_norm_tik(G1new,Lr)) + gamma_v*sum(gsp_norm_tik(G2new,Lr'))];

end


