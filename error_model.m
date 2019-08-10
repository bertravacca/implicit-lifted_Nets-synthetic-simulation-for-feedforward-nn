function[training_error,test_error]=error_model()
global data sizes
training_error=sqrt((1/sizes.m_train))*norm(learnt_model(data.X_train)-data.Y_train,'fro');
test_error=sqrt((1/sizes.m_test))*norm(learnt_model(data.X_test)-data.Y_test,'fro');
end