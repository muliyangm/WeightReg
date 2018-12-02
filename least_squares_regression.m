
function [W_dv, b] = least_squares_regression(X, Y, dv_dim, gamma) % input Y = R'

% X:                         each column is a data point
% Y:                         each column is an target data point: such as  [-1, 1, -1, ..., -1]'
% gamma:                     a positive scalar

% return W_dv and b

N = size(X, 2); % dd*n
% [dim_reduced, N] = size(Y); % c*n, here Y is transposed 

% Remove the mean
XMean = mean(X')';                               % is a column vector
XX = X - repmat(XMean, 1, N);                    % each column is a data point

% W_dv = pinv(XX * XX' + gamma * diag(1./dv_dim)) * (XX * Y');
W_dv = (XX * XX' + gamma * diag(1./dv_dim)) \ (XX * Y');    % Faster

b = Y - W_dv' * X;     % each column is an error vector
b = mean(b')';      % now b is a column vector

end
