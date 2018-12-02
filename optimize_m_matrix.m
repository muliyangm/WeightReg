
function M = optimize_m_matrix(P, Y)

% P:              the residual matrix, each row is a residual vector
% Y:              as you know, the coding matrix of class label

% return:         The optimized matrix

N = size(P, 1); % Here N is the number of rows of matrix P, i.e. number of samples
num_class = size(Y, 2);

M0 = zeros(N, num_class); % n*c

M = max(Y .* P, M0);

return;