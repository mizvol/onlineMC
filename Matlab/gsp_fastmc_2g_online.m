function [Lr, Sp, G1, G2, U, S, V] = gsp_fastmc_2g_online(X, M, gamma1, gamma2, Ninit, batch, iters, G1, G2, param)
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

if nargin<10
    param = struct;
end

if ~isfield(param,'paramnn'),  param.paramnn = struct; end
if ~isfield(param.paramnn,'k'), param.paramnn.k = 10; end
if ~isfield(param.paramnn,'use_flann'), param.paramnn.use_flann = 0; end
if ~isfield(param.paramnn,'use_l1'), param.paramnn.use_l1 = 0; end

if ~isfield(param,'param_solver'), param.param_solver = struct; end
if ~isfield(param.param_solver, 'verbose'), param.param_solver.verbose = 2; end
if ~isfield(param.param_solver, 'maxit'), param.param_solver.maxit = 200; end
if ~isfield(param.param_solver, 'tol'), param.param_solver.tol = 1e-5; end
if ~isfield(param.param_solver, 'stopping_criterion'), param.param_solver.stopping_criterion = 'rel_norm_primal'; end


%% Optimization
%% initial small batch part

% solve the initial batch part
f1.grad = @(x,T) 2*M(:,1:Ninit).*(x-X(:,1:Ninit));
f1.beta = 2;
f1.eval = @(x) norm(M(:,1:Ninit).*(x-X(:,1:Ninit)),'fro');

G1_small = gsp_nn_graph(X(:,1:Ninit));
G2_small = gsp_nn_graph(X(:,1:Ninit)');
G1_small = gsp_estimate_lmax(G1_small);
G2_small = gsp_estimate_lmax(G2_small);

f3.grad = @(x) gamma1*2*G1_small.L*x;
f3.eval = @(x) gamma1*sum(gsp_norm_tik(G1_small,x));
f3.beta = 2*gamma1*G1_small.lmax;

f4.grad = @(x) gamma2*(2*x*G2_small.L);
f4.eval = @(x) gamma2*sum(gsp_norm_tik(G2_small,x'));
f4.beta = 2*gamma2*G2_small.lmax;

Lr = solvep(X(:,1:Ninit),{f1,f3,f4},param.param_solver);

param.param_solver.maxit = iters;
%% online update part

for i = 1 : batch : size(X,2) - Ninit
    G1 = gsp_nn_graph(X(:,1:Ninit+i+batch-1));
    G2 = gsp_nn_graph(X(:,1:Ninit+i+batch-1)');
    G1 = gsp_estimate_lmax(G1);
    G2 = gsp_estimate_lmax(G2);
    
    f1.grad = @(x,T) 2*M(:,1:Ninit+i+batch-1).*(x-X(:,1:Ninit+i+batch-1));
    f1.beta = 2;
    f1.eval = @(x) norm(M(:,1:Ninit+i+batch-1).*(x-X(:,1:Ninit+i+batch-1)),'fro');
    
    f3.grad = @(x) gamma1*2*G1.L*x;
    f3.eval = @(x) gamma1*sum(gsp_norm_tik(G1,x));
    f3.beta = 2*gamma1*G1.lmax;
    
    f4.grad = @(x) gamma2*(2*x*G2.L);
    f4.eval = @(x) gamma2*sum(gsp_norm_tik(G2,x'));
    f4.beta = 2*gamma2*G2.lmax;
    
    Lr = solvep([Lr X(:,Ninit+i:Ninit+i+batch-1)],{f1,f3,f4},param.param_solver);
    
end

Sp = X - Lr;
%% Optional output parameters
if nargout>4
    [U, S , V] = svdecon(Lr);
end


end


