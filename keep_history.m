function[]=keep_history()
global objective tests RMSE iter weights data sizes hidden_var

[objective.fval_fenchel(1,iter),objective.fval_fenchel(2,iter)] = objective_function();
[tests.cv_activation_layer_1(iter), tests.cv_activation_layer_2(iter), tests.W_1_change(iter), tests.W_2_change(iter), tests.W_3_change(iter)]=test_algo();
[RMSE.train(iter),RMSE.test(iter)] = error_model();
tests.error_feedforward(iter)=(1/sqrt(sizes.m_train))*norm(weights.W_3*hidden_var.X_2-weights.W_3*tanh(weights.W_2*tanh(weights.W_1*data.X_train)),'fro');
tests.error_feedforward_first_layer(iter)=(1/sqrt(sizes.m_train))*norm(hidden_var.X_1-tanh(weights.W_1*data.X_train),'fro');

end