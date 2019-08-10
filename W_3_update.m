function [] = W_3_update(method)
global data hidden_var  weights sizes
weights.W_3_keep = weights.W_3;
Y_train = data.Y_train;
X_2 = hidden_var.X_2;
n_2 = sizes.n_2;

if strcmp(method,'direct')
    precision_num = 10^(-6);
    W_3_new=Y_train*X_2'/(X_2*X_2'+precision_num*eye(n_2));
end

weights.W_3=W_3_new;
end