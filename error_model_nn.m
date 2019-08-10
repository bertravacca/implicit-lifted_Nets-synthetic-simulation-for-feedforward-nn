function[training_error,test_error]=error_model_nn()
global data sizes
training_error=sqrt((1/sizes.m_train)*norm(nn_model(data.X_train)-data.Y_train,'fro')^2);
test_error=sqrt((1/sizes.m_test)*norm(nn_model(data.X_test)-data.Y_test,'fro')^2);
end