function [Lr, Sp, G1, G2, U, S, V] = gsp_frpcag_2g_online_knn(X, gamma1, gamma2, Ninit, batch, iters, G1, G2, param)
%GSP_FASTMC_2G_ONLINE Fast online matrix completion on 2 graphs
%   Usage: [Lr] = gsp_fastmc_2g_online(X, M, gamma1, gamma2, G1, G2)
%          [Lr] = gsp_fastmc_2g_online(X, M, gamma1, gamma2, G1, G2, param)
%          [Lr, Sp, G1, G2] = gsp_fastmc_2g_online( ... );
%          [Lr, Sp, G1, G2, U, S, V] = gsp_fastmc_2g_online( ... );
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
if ~isfield(param.paramnn,'use_flann'), param.paramnn.use_flann = 1; end
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
paraml1.y = X(:,1:Ninit);
f1.prox = @(x,T) prox_l1(x,T,paraml1);
f1.eval = @(x) norm(x(:),1);

%G1_small = gsp_nn_graph(X(:,1:Ninit),'normalized');
G1_small = gsp_nn_graph(X(:,1:Ninit));
G1_small = gsp_estimate_lmax(G1_small);
G2_small.L = G2.L(1:Ninit,1:Ninit);
G2_small.lmax = G2.lmax;

f3.grad = @(x) gamma1*2*G1_small.L*x;
f3.eval = @(x) gamma1*sum(gsp_norm_tik(G1_small,x));
f3.beta = 2*gamma1*G1_small.lmax;

f4.grad = @(x) gamma2*(2*x*G2_small.L);
f4.eval = @(x) gamma2*sum(gsp_norm_tik(G2_small,x'));
f4.beta = 2*gamma2*G2_small.lmax;

Lr = solvep(X(:,1:Ninit),{f1,f3,f4},param.param_solver);
Lr = [Lr zeros(size(Lr,1),size(X,2)-Ninit)];
param.param_solver.maxit = iters;
%% online update part
temp_size = [];
for i = 1 : batch : size(X,2) - Ninit
    %G1new = gsp_nn_graph(X(:,1:Ninit+i+batch-1),'normalized');
%     G1new = gsp_nn_graph(X(:,1:Ninit+i+batch-1));
%     G1new = gsp_estimate_lmax(G1new);
    G1new = G1;
    
    indices_knn = [];
    Linit = [];
    for j = 1 : batch
        [temp, temp_ind] = sort(full(G2.W(1:Ninit+i+j-1,Ninit+i+j-1)),'descend');
        temp_ind = temp_ind(temp~=0);
        indices_knn = union(indices_knn,temp_ind(1:min([length(temp_ind) param.paramnn.k])));
        val_temp = temp(temp_ind(temp~=0) < Ninit+i);
        %         if ~isempty(val_temp)
        %             Linit = [Linit sum(Lr(:,1:length(val_temp)).*repmat(val_temp',...
        %                 size(Lr,1),1),2)/sum(val_temp)];
        %         else
        %             Linit = [Linit X(:,Ninit+i+j-1)];
        %         end
        Linit = [Linit X(:,Ninit+i+j-1)];
    end
    indices_knn = union(indices_knn,[Ninit+i:Ninit+i+batch-1]);
    temp_size = [temp_size length(indices_knn)];
    
    G2new.L = G2.L(indices_knn,indices_knn);
    G2new.lmax = G2.lmax;
    
    paraml1.verbose = 2;
    paraml1.y = X(:,indices_knn);
    f1.prox = @(x,T) prox_l1(x,T,paraml1);
    f1.eval = @(x) norm(x(:),1);
    
    paramnuclear.single = 1;
    f3.prox = @(x,T) double(prox_nuclearnorm(x,gamma2*T,paramnuclear));
    f3.eval = @(x) lambda*norm_nuclear(x);
    
    %     f3.grad = @(x) gamma1*2*G1new.L*x;
    %     f3.eval = @(x) gamma1*sum(gsp_norm_tik(G1new,x));
    %     f3.beta = 2*gamma1*G1new.lmax;
    
    f4.grad = @(x) gamma2*(2*x*G2new.L);
    f4.eval = @(x) gamma2*sum(gsp_norm_tik(G2new,x'));
    f4.beta = 2*gamma2*G2new.lmax;
    
    Lrtemp = solvep([X(:,indices_knn(1:end-batch)) Linit],{f1,f3,f4},param.param_solver);
    %Lrtemp = solvep([Lr(:,indices_knn(1:end-batch)) X(:,Ninit+i:Ninit+i+batch-1)],{f1,f3,f4},param.param_solver);
    Lr(:,indices_knn(indices_knn < Ninit+i)) = (Lr(:,indices_knn(indices_knn < Ninit+i))...
        + Lrtemp(:,1:end-batch))/2;
    Lr(:,indices_knn(end-batch+1:end)) = Lrtemp(:,end-batch+1:end);
end

Sp = X - Lr;
%% Optional output parameters
if nargout>4
    [U, S , V] = svdecon(Lr);
end

end


