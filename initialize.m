function[]=initialize()
global weights RMSE tests objective K
weights.W_1=weights.W_1_init;
weights.W_2=weights.W_2_init;
weights.W_3=weights.W_3_init;
X_1_update('feedforward');
X_2_update('feedforward');
RMSE.train = NaN*zeros(K,1);
RMSE.test = NaN*zeros(K,1);
tests.cv_activation_layer_1 = NaN*zeros(K,1);
tests.cv_activation_layer_2 = NaN*zeros(K,1);
tests.W_1_change = NaN*zeros(K,1);
tests.W_2_change = NaN*zeros(K,1);
tests.W_3_change = NaN*zeros(K,1);
tests.error_feedforward =  NaN*zeros(K,1);
tests.error_feedforward_first_layer =  NaN*zeros(K,1);
objective.fval_fenchel = NaN*zeros(2,K);
end