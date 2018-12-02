
function [XX_train, XX_test, W_dv, b, dv, dv_dim] = train_weightreg(X_train, X_test, Y_train, view_num, gamma, iters, epsilon)

% X_train/test:  cell with d * n, each column is a data point
% Y_tain:        n * 1, labels, a column vector, such as  [1, 2, 3, 4, 1, 3, 2, ...]'
% gamma:         the regularization \lambda parameter in Equation (5)
% iters:         max iteration times
% epsilon:       terminate convergence

% XX:            dd * n, all view X
% Y:             n * c, +-1 form of Y_train
% dv:            viewnum * 1, weight parameter
% dim:           viewnum * 1, records the dims of each view
% dv_dim         all_dimension_num (dd) * 1

%% generate dim & all-view XX, initialize dv

XX_train = []; XX_test = [];
dv = zeros(view_num, 1);
dim = zeros(view_num, 1);

flag = 0;
for i = 1 : view_num
    [dim(i), N] = size(X_train{1,i}); % N is the number of samples in the training set
    XX_train((flag+1):(flag + dim(i)),:) = X_train{1,i};
    XX_test((flag+1):(flag + dim(i)),:) = X_test{1,i};
    dv(i) = 1 / view_num;
    flag = flag + dim(i);
end

num_class = max(Y_train);

%% Code class matrix Y

Y = -1.0 * ones(N, num_class);

for i = 1 : N
    Y(i, Y_train(i)) = 1.0;
end

%% initialize W, b & dv_dim

dv_dim = []; flag=0;
for i = 1 : view_num
    dv_dim((flag+1):(flag + dim(i)),:) = repmat(dv(i),dim(i),1);
    flag = flag + dim(i);
end

[W_dv0, b0] = least_squares_regression(XX_train, Y', dv_dim, gamma);
[dv0, dv_dim0] = optimize_dv_dim(view_num, dim, W_dv0);

W_dv = W_dv0;
b = b0;
dv_dim = dv_dim0;
% obj0 = inf;

%% Training...
obj_vector = zeros(iters,1);

for i = 1: iters
%     tic;
    % optimize matrix M
    P = XX_train' * W_dv0 + ones(N, 1) * b0' - Y;
    M = optimize_m_matrix(P, Y);
    R = Y + (Y .* M);
    [dv, dv_dim] = optimize_dv_dim(view_num, dim, W_dv);
    [W_dv, b] = least_squares_regression(XX_train, R', dv_dim, gamma);
%     toc;
    para_va = trace ( (W_dv-W_dv0)' * (W_dv-W_dv0) ) + (b-b0)' * (b-b0)  + (dv-dv0)' * (dv-dv0);
    obj = trace((XX_train'*W_dv+ones(N,1)*b'-R)' * (XX_train'*W_dv+ones(N,1)*b'-R))...
        + trace((sqrt(gamma * diag(1./dv_dim)) * W_dv)' * sqrt(gamma * diag(1./dv_dim)) * W_dv);
    
%     fprintf('Iter %d: OBJ == %.5f <=====> paras_updated == %.9f\n', i, obj, para_va);
%     if (para_va < epsilon || obj > obj0)
    if (para_va < epsilon)
        break; 
    end
        
    W_dv0 = W_dv; b0 = b; dv0 = dv; obj0 = obj;
    obj_vector(i) = obj0;

end
return;
