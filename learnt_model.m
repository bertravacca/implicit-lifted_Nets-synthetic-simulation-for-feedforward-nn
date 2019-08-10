function[y]=learnt_model(x)
global weights
y=weights.W_3*tanh(weights.W_2*tanh(weights.W_1*x));
end