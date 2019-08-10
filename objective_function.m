function[val_obj, val_con]=objective_function()
global data weights hparam hidden_var
Y_train = data.Y_train;
X_train = data.X_train;
W_3 = weights.W_3;
W_2 = weights.W_2;
W_1 = weights.W_1;
X_1 = hidden_var.X_1;
X_2 = hidden_var.X_2;
mu_2 = hparam.epsilon;
mu_1 = mu_2^2;
val_obj = 0.5*norm(Y_train-W_3*X_2,'fro')^2+mu_1*(phi_star(X_2)+phi(W_2*X_1)-trace(X_2'*(W_2*X_1)))+mu_2*(phi_star(X_1)+phi(W_1*X_train)-trace(X_1'*(W_1*X_train)));
val_con = mu_1*(phi_star(X_2)+phi(W_2*X_1)-trace(X_2'*(W_2*X_1)))+mu_2*(phi_star(X_1)+phi(W_1*X_train)-trace(X_1'*(W_1*X_train)));
end