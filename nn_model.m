function[y]=nn_model(x)
global weights
x_1 = tanh(weights.W_1_nn*x+weights.bias_1_nn);
x_2 = tanh(weights.W_2_nn*x_1+weights.bias_2_nn);
y = weights.W_3_nn*x_2+weights.bias_3_nn;
end