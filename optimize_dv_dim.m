
function [dv, dv_dim] = optimize_dv_dim(view_num, dim, W_dv)

dv = zeros(view_num, 1);
sum_norm_Wdv = 0.0;
flag = 0;
for i = 1 : view_num
    dv(i) = norm( W_dv( (flag+1):(flag+dim(i)), : ), 'fro' );
    flag = flag + dim(i);
    sum_norm_Wdv = sum_norm_Wdv + dv(i);
end

dv = dv / sum_norm_Wdv;

dv_dim = zeros(flag,1);
flag = 0;
for i = 1 : view_num
    dv_dim( (flag+1) : (flag+dim(i)) , :) = repmat(dv(i),dim(i),1);
    flag = flag + dim(i);
end

return;