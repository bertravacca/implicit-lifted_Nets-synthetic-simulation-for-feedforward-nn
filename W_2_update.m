function [] = W_2_update(method,precision)
global weights hidden_var sizes tests iter
weights.W_2_keep = weights.W_2;
X_2 = hidden_var.X_2;
X_1 = hidden_var.X_1;
W_2 = weights.W_2;

if strcmp(method,'gradient')
    W_2_new=W_2;
    alpha=1/norm(X_1,'fro')^2;
    num_iter =1;
    max_iter=1000;
    name=['W_2_update_fval_iter_',num2str(iter)];
    tests.(name) = NaN*zeros(max_iter,1);
    f_val=0;
    f_val_prev=1;
    while abs(f_val-f_val_prev)>precision
        f_val_prev=f_val;
        W_2_new=W_2_new-alpha*(tanh(W_2_new*X_1)*X_1'-X_2*X_1');
        f_val=(1/sizes.m_train)*(phi_star(X_2)+phi(W_2_new*X_1)-trace(X_2'*W_2_new*X_1));
        tests.(name)(num_iter) = f_val;
        num_iter = num_iter+1;
    end
end


weights.W_2=W_2_new;

end