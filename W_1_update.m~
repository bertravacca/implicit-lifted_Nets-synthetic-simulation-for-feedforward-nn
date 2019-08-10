function [W_1_new] = W_1_update(method, precision)
global data hidden_var sizes  weights tests iter
weights.W_1_keep = weights.W_1;
X_1 = hidden_var.X_1;
X_train = data.X_train;
W_1 =  weights.W_1;

if strcmp(method,'gradient')
    W_1_new=W_1;
    alpha=1/norm(X_train,'fro')^2;
    num_iter =1;
    max_iter=1000;
    name=['W_1_update_fval_iter_',num2str(iter)];
    tests.(name) = NaN*zeros(max_iter,1);
    
    f_val=0;
    f_val_prev=1;
    while abs(f_val-f_val_prev)>precision
        f_val_prev=f_val;
        W_1_new=W_1_new-alpha*(tanh(W_1_new*X_train)*X_train'-X_1*X_train');
        f_val=(1/sizes.m_train)*(phi_star(X_1)+phi(W_1_new*X_train)-trace(X_1'*W_1_new*X_train));
        tests.(name)(num_iter) = f_val;
        num_iter = num_iter+1;
    end
end

weights.W_1=W_1_new;
end