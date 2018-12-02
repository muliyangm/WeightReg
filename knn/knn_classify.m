
function [acc, predict_label] = knn_classify(train_label, test_label, distance, K, flag)
% distance: test_num * train_num
% flag:'descend' or 'ascend'
% K,the parameter of knn. it can be a single value like 1/3/5, or a vector
% like [1,3,5] in order to test multiple performance under different k
% value
num_test = length(test_label); num_k = length(K); 
acc = zeros(num_k,1); predict_label = zeros(num_k,num_test);
[~,id] = sort(distance, 2, flag);
Label_matrix = train_label(id);
for k = 1:num_k
    first_k_label = Label_matrix(:,1:K(k));
    num=0;
    knn_label = zeros(num_test,1);
    for i=1:num_test 
        table = tabulate(first_k_label(i,:));
        [~,I]=max(table(:,2));
        knn_label(i)= I;
        if knn_label(i)==test_label(i)
            num=num+1;
        end
    end
    predict_label(k,:) =  knn_label;
    acc(k) = num / num_test;
end
