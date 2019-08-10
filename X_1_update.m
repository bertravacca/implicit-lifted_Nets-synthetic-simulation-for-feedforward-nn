function[ ]=X_1_update(option,precision)
global weights data sizes hparam hidden_var iter tests 
 W_1 = weights.W_1;
 W_2 =weights.W_2;
 X_train =data.X_train;
 m_train = sizes.m_train;
 n_1 = sizes.n_1;
 n_2 = sizes.n_2;
  
 if isfield(hidden_var,'X_1')
     X_1 = hidden_var.X_1;
 else
     X_1=tanh(W_1*X_train);
 end
 
 if isfield(hidden_var,'X_2')
     X_2 = hidden_var.X_2;
 else
     X_2=tanh(W_2*X_1);
 end
 
 if isfield(hparam,'epsilon')
     mu_2 = hparam.epsilon;
     mu_1 = mu_2^2;
 end
 
 if strcmp(option,'feedforward')
     X_1_new=tanh(W_1*X_train);
 end
 
if strcmp(option,'newton')
    alpha=0.3;
    numerical_precision=10^(-9);
    num_iter = 1;
    max_iter=1000;
    
    X_1_new=tanh(W_1*X_train);
   
    name=['X_1_update_fval_iter_',num2str(iter)];
    tests.(name) = NaN*zeros(max_iter,1);
    tests.(name)(1) = mu_2*(phi_star(X_2)+phi(W_2*X_1_new)-trace(X_2'*W_2*X_1_new))+mu_1*(phi(W_1*X_train)+phi_star(X_1_new)-trace(X_1_new'*W_1*X_train));
    
    max_grad = 1;
    while max_grad>precision && num_iter<100
        max_grad = 0;
        for i=1:m_train
            grad=mu_2*W_2'*tanh(W_2*X_1_new(:,i))-mu_2*W_2'*X_2(:,i)+mu_1*atanh(X_1_new(:,i))-mu_1*W_1*X_train(:,i);
            norm_grad = norm(grad);
            if norm_grad>max_grad
                max_grad=norm_grad;
            end
            hessian=mu_2*W_2'*diag(ones(n_2,1)-tanh(W_2*X_1_new(:,i)).^2)*W_2+mu_1*1./(1-X_1_new(:,i).^2).*eye(n_1);
            direction=hessian\grad;
            X_1_new(:,i)=min(max(-1+numerical_precision,X_1_new(:,i)-alpha*direction),1-numerical_precision);
        end
        
        tests.(name)(num_iter+1) = mu_2*(phi_star(X_2)+phi(W_2*X_1_new)-trace(X_2'*W_2*X_1_new))+mu_1*(phi(W_1*X_train)+phi_star(X_1_new)-trace(X_1_new'*W_1*X_train));
        
        num_iter = num_iter+1;
    end
end

hidden_var.X_1=X_1_new;
end