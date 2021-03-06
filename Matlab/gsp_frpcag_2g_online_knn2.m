function [Lr, Sp, G1, G2, U, S, V] = gsp_frpcag_2g_online_knn2(X, gamma1, gamma2, Ninit, batch, iters, G1, G2, param)
%GSP_FASTMC_2G_ONLINE Fast online matrix completion on 2 graphs
%   Usage: [Lr] = gsp_fastmc_2g_online_knn2(X, M, gamma1, gamma2, G1, G2)
%          [Lr] = gsp_fastmc_2g_online_knn2(X, M, gamma1, gamma2, G1, G2, param)
%          [Lr, Sp, G1, G2] = gsp_fastmc_2g_online_knn2( ... );
%          [Lr, Sp, G1, G2, U, S, V] = gsp_fastmc_2g_online_knn2( ... );
%
%   Input parameters:
%       X       : Input data (matrix of double)
%       M       : the mask operator
%       gamma1  : Regularization parameter for  graph 1
%       gamma2  : Regularization parameter for  graph 2
%       param   : Optional optimization parameters
%
%   Output Parameters:
%       Lr      : Low-rank part of the data
%       Sp      : Sparse part of the data
%       G1      : the graph between columns of X
%       G2      : the graph between rows of X
%       U       : Part of the SVD of Lr
%       S       : Part of the SVD of Lr
%       V       : Part of the SVD of Lr
%
%   This function computes a low rank approximation of the incomplete data stored in
%   *X* by solving an optimization problem in online manner:
%
%   .. argmin_Lr  ||M(X - Lr)||^2_2  + gamma1 tr( Lr^T L1 Lr) + gamma2 tr( Lr L2 Lr^T)
%
%   .. math:: argmin_Lr  ||M(X - Lr)||^2_2 + gamma1 tr( Lr^T L1 Lr) + gamma2 tr( Lr L2 Lr^T)
%
%
%   If $0$ is given for *G*, the corresponding graph will
%   be computed internally. The graph construction can be tuned using the
%   optional parameter: *param.paramnn*.
%
%
%   If the number of output argument is greater than 2. The function, will
%   additionally compute a very economical SVD such that $ Lr = U S V^T$.
%
%   This function uses the UNLocBoX to be working.
%
% Author: Nauman Shahid
% Date  : 15th June 2016
% references: see "Matrix Completion on Graphs" by Vassilis Kalofolias

%% Optional parameters

if nargin<9
    param = struct;
end

if ~isfield(param,'paramnn'),  param.paramnn = struct; end
if ~isfield(param.paramnn,'k'), param.paramnn.k = 10; end
if ~isfield(param.paramnn,'use_flann'), param.paramnn.use_flann = 0; end
if ~isfield(param.paramnn,'use_l1'), param.paramnn.use_l1 = 0; end

if ~isfield(param,'param_solver'), param.param_solver = struct; end
if ~isfield(param.param_solver, 'verbose'), param.param_solver.verbose = 2; end
if ~isfield(param.param_solver, 'maxit'), param.param_solver.maxit = 10; end
if ~isfield(param.param_solver, 'tol'), param.param_solver.tol = 1e-5; end
if ~isfield(param.param_solver, 'stopping_criterion'), param.param_solver.stopping_criterion = 'rel_norm_primal'; end

if ~isstruct(G1)
    G1 = gsp_nn_graph(X, param.paramnn);
end

if ~isstruct(G2)
    G2 = gsp_nn_graph(X', param.paramnn);
end

if ~isfield(G1,'lmax')
    G1 = gsp_estimate_lmax(G1);
end

if ~isfield(G2,'lmax')
    G2 = gsp_estimate_lmax(G2);
end
%% Optimization
%% initial small batch part

% solve the initial batch part
paraml1.verbose = 2;
paraml1.y = X(1:Ninit,1:Ninit);
f1.prox = @(x,T) prox_l1(x,T,paraml1);
f1.eval = @(x) norm(x(:),1);

G1_small.L = G1.L(1:Ninit,1:Ninit);
G1_small.lmax = G1.lmax;
G2_small.L = G2.L(1:Ninit,1:Ninit);
G2_small.lmax = G2.lmax;

f3.grad = @(x) gamma1*2*G1_small.L*x;
f3.eval = @(x) gamma1*sum(gsp_norm_tik(G1_small,x));
f3.beta = 2*gamma1*G1_small.lmax;

f4.grad = @(x) gamma2*(2*x*G2_small.L);
f4.eval = @(x) gamma2*sum(gsp_norm_tik(G2_small,x'));
f4.beta = 2*gamma2*G2_small.lmax;

Lr = solvep(X(1:Ninit,1:Ninit),{f1,f3,f4},param.param_solver);
% Lr = [Lr ; zeros(size(X,1)-Ninit,size(Lr,2))];
% Lr = [Lr zeros(size(Lr,1),size(X,2)-Ninit)];
param.param_solver.maxit = iters;
%% online update part
tempr_size = [];
tempc_size = [];
for i = 1 : batch : size(X,2) - Ninit
    
    %%  update the new rows
    indicesr_knn = [];
    for j = 1 : batch
        [temp, temp_ind] = sort(full(G1.W(1:Ninit+i+j-1,Ninit+i-1)),'descend');
        indicesr_knn = union(indicesr_knn,temp_ind(temp~=0));
    end
    indicesr_knn = union(indicesr_knn,[Ninit+i:Ninit+i+batch-1]);
    tempr_size = [tempr_size length(indicesr_knn)];
    
    G1new.L = G1.L(indicesr_knn,indicesr_knn);
    G1new.lmax = G1.lmax;
    
    G2new.L = G2.L(1:Ninit+i-1,1:Ninit+i-1);
    G2new.lmax = G2.lmax;
    
    paraml1.y = X(indicesr_knn,1:Ninit+i-1);
    f1.prox = @(x,T) prox_l1(x,T,paraml1);
    f1.eval = @(x) norm(x(:),1);
    
    f3.grad = @(x) gamma1*2*G1new.L*x;
    f3.eval = @(x) gamma1*sum(gsp_norm_tik(G1new,x));
    f3.beta = 2*gamma1*G1new.lmax;
    
    f4.grad = @(x) gamma2*(2*x*G2new.L);
    f4.eval = @(x) gamma2*sum(gsp_norm_tik(G2new,x'));
    f4.beta = 2*gamma2*G2new.lmax;
    
    Lrtemp = solvep(X(indicesr_knn,:),{f1,f3,f4},param.param_solver);
    
%     Lrtemp = solvep([Lr(indicesr_knn(1:end-batch),1:Ninit+i-1)...
%         ; X(Ninit+i:Ninit+i+batch-1,1:Ninit+i-1)],{f1,f3,f4},param.param_solver);
    
    if ~isempty(indicesr_knn)
        Lr(indicesr_knn(indicesr_knn < Ninit+i),:) = Lrtemp(1:end-batch,:);
    end
    Lr = [Lr ; Lrtemp(end-batch+1:end,:)];
    
    %%
    % update the new columns
    indicesc_knn = [];
    for j = 1 : batch
        [temp, temp_ind] = sort(full(G2.W(1:Ninit+i-1,Ninit+i+j-1)),'descend');
        indicesc_knn = union(indicesc_knn,temp_ind(temp~=0));
    end
    indicesc_knn = union(indicesc_knn,[Ninit+i:Ninit+i+batch-1]);
    tempc_size = [tempc_size length(indicesc_knn)];
    
    G2new.L = G2.L(indicesc_knn,indicesc_knn);
    G2new.lmax = G2.lmax;
    
    G1new.L = G1.L(1:Ninit+i+batch-1,1:Ninit+i+batch-1);
    G1new.lmax = G1.lmax;
    
    % solve the optimization
    paraml1.y = X(1:Ninit+i-1,indicesc_knn);
    f1.prox = @(x,T) prox_l1(x,T,paraml1);
    f1.eval = @(x) norm(x(:),1);
    
    f3.grad = @(x) gamma1*2*G1new.L*x;
    f3.eval = @(x) gamma1*sum(gsp_norm_tik(G1new,x));
    f3.beta = 2*gamma1*G1new.lmax;
    
    f4.grad = @(x) gamma2*(2*x*G2new.L);
    f4.eval = @(x) gamma2*sum(gsp_norm_tik(G2new,x'));
    f4.beta = 2*gamma2*G2new.lmax;
    
    Lrtemp = solvep(X(:,indicesc_knn) ,{f1,f3,f4},param.param_solver);
    
    %Lrtemp = solvep([Lr(:,indicesc_knn(1:end-batch)) X(1:Ninit+i+batch-1,Ninit+i:Ninit+i+batch-1)]...
      %  ,{f1,f3,f4},param.param_solver);
    
    if ~isempty(indicesc_knn)
        Lr(:,indicesc_knn(indicesc_knn < Ninit+i)) = Lrtemp(:,1:end-batch);
    end
    
    Lr = [Lr Lrtemp(:,end-batch+1:end)];
end

Sp = X - Lr;
%% Optional output parameters
if nargout>4
    [U, S , V] = svdecon(Lr);
end

end


