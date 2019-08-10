function[]=X_2_update(option,precision)
global weights hidden_var data sizes hparam iter tests 
W_1 = weights.W_1;
W_2 = weights.W_2;
W_3 = weights.W_3;
X_train=data.X_train;

 
 if isfield(hidden_var,'X_1')
     X_1 = hidden_var.X_1;
 else
     X_1=tanh(W_1*X_train);
 end
 
 
 if isfield(hparam,'epsilon')
     mu_2 = hparam.epsilon;
 end
 
Y_train = data.Y_train;
m_train = sizes.m_train;
n_2 = sizes.n_2;

if strcmp(option,'feedforward')
    X_2_new=tanh(W_2*X_1);
end

if strcmp(option,'newton')
    alpha=0.3;
    numerical_precision=10^-9;
    num_iter = 1;
    max_iter=1000;

    % initialize
    X_2_new=tanh(W_2*X_1);
    
    name=['X_2_update_fval_iter_',num2str(iter)];
    tests.(name) = NaN*zeros(max_iter,1);
    tests.(name)(1) = mu_2*(phi_star(X_2_new)+phi(W_2*X_1)-trace(X_2_new'*W_2*X_1))+0.5*norm(Y_train-W_3*X_2_new,'fro')^2;
    
    max_grad= 1;
    while max_grad>precision && num_iter<100
        max_grad=0;
        for i=1:m_train
            grad=W_3'*W_3*X_2_new(:,i)-W_3'*Y_train(:,i)+mu_2*atanh(X_2_new(:,i))-mu_2*W_2*X_1(:,i);
            norm_grad=norm(grad);
            if norm_grad>max_grad
                max_grad = norm_grad;
            end
            hessian=W_3'*W_3+mu_2./(1-X_2_new(:,i).^2).*eye(n_2);
            direction=hessian\grad;
            X_2_new(:,i)=min(max(-1+numerical_precision,X_2_new(:,i)-alpha*direction),1-numerical_precision);
        end
        tests.(name)(num_iter+1) = mu_2*(phi_star(X_2_new)+phi(W_2*X_1)-trace(X_2_new'*W_2*X_1))+0.5*norm(Y_train-W_3*X_2_new,'fro')^2;
        num_iter = num_iter+1;
    end
end

hidden_var.X_2 = X_2_new;
end