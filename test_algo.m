function[cv_activation_layer_1, cv_activation_layer_2, W_1_change, W_2_change, W_3_change]=test_algo()
global weights sizes data hidden_var 
 m_train = sizes.m_train;
 n = sizes.n; 
 n_1 = sizes.n_1;
 n_2 = sizes.n_2;
 p = sizes.p;
 X_train = data.X_train;
 X_1 = hidden_var.X_1; 
 X_2 = hidden_var.X_2;
 W_1 = weights.W_1;
 W_2 = weights.W_2;
 W_3 = weights.W_3;
 W_1_keep = weights.W_1_keep;
 W_2_keep = weights.W_2_keep;
 W_3_keep = weights.W_3_keep;
 
 
 cv_activation_layer_1 = 1/(m_train*n_1)*norm(X_1-tanh(W_1*X_train),'fro');
 cv_activation_layer_2 = 1/(m_train*n_2)*norm(X_2-tanh(W_2*X_1),'fro');
 W_1_change = 1/(n*n_1)*norm(W_1-W_1_keep,'fro');
 W_2_change = 1/(n_1*n_2)*norm(W_2-W_2_keep,'fro');
 W_3_change = 1/(n_2*p)*norm(W_3-W_3_keep,'fro');

end