
% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B

function D = L2_distance(A, B)
% A, B: two matrices. each column is a data point: m * n
% D:    distance matrix of a and b

if (size(A, 1) == 1)
    A = [A; zeros(1, size(A, 2))]; 
    B = [B; zeros(1, size(B, 2))]; 
end

AA=sum(A.*A); BB=sum(B.*B); AB=A'*B; 
D = repmat(AA',[1 size(BB,2)]) + repmat(BB, [size(AA,2) 1]) - 2*AB;

D = real(D);
D = max(D, 0);
