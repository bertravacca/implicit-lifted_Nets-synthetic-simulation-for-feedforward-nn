function[training_error]=error()
global X_train Y_train X_test Y_test 
training_error=norm(learnt_model(X_train)-Y_train,'fro');
test_error=norm(learnt_model(X_test)-Y_test,'fro');
end