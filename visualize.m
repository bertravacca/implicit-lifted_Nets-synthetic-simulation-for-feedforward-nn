function[]=visualize()
global count_figure K RMSE tests
count_figure=count_figure+1;
if ishandle(count_figure)
    close(count_figure)
end

fs=18;
[train_error_nn, ~] = error_model_nn();
figure(count_figure)
plot([RMSE.init_train;RMSE.train],'b','LineWidth',2)
hold on
plot([RMSE.init_test;RMSE.test],'r','LineWidth',2)
plot(train_error_nn*ones(K+1,1),'k--','LineWidth',2)
disp('RMSE (test set): '); disp(RMSE.test(K));
disp('RMSE (train set): '); disp(RMSE.train(K));
legend('RMSE train','RMSE test','RMSprop baseline','FontSize',fs)
title('RMSE across iterations','FontSize',fs)


if isfield(tests,'W_3_update_fval_iter_1')
    count_figure=count_figure+1;
    if ishandle(count_figure)
        close(count_figure)
    end
    
    figure(count_figure)
    plot(tests.W_3_update_fval_iter_1,'Color',[0 0 1],'LineWidth',2)
    hold on 
    for k =2:K
        name=['W_3_update_fval_iter_',num2str(k)];
        plot(tests.(name),'Color',[0,(1-k)/(1-K),1+(k-1)/(1-K)],'LineWidth',2)
    end
    title('Fval history for W_3 update [RMSE]','FontSize',fs)
end

if isfield(tests,'W_2_update_fval_iter_1')
    count_figure=count_figure+1;
    if ishandle(count_figure)
        close(count_figure)
    end
    
    figure(count_figure)
    plot(tests.W_2_update_fval_iter_1,'b','LineWidth',2)
    hold on 
    for k =2:K
        name=['W_2_update_fval_iter_',num2str(k)];
        plot(tests.(name),'Color',[0,(1-k)/(1-K),1+(k-1)/(1-K)],'LineWidth',2)
    end
    title('Fval history for W_2 update','FontSize',fs)
end


if isfield(tests,'W_1_update_fval_iter_1')
    count_figure=count_figure+1;
    if ishandle(count_figure)
        close(count_figure)
    end
    
    figure(count_figure)
    plot(tests.W_1_update_fval_iter_1,'b','LineWidth',2)
    hold on 
    for k =2:K
        name=['W_1_update_fval_iter_',num2str(k)];
        plot(tests.(name),'Color',[0,(1-k)/(1-K),1+(k-1)/(1-K)],'LineWidth',2)
    end
    title('Fval history for W_1 update','FontSize',fs)
end

if isfield(tests,'X_1_update_fval_iter_1')
    count_figure=count_figure+1;
    if ishandle(count_figure)
        close(count_figure)
    end
    
    figure(count_figure)
    plot(tests.X_1_update_fval_iter_1,'b','LineWidth',2)
    hold on 
    for k =2:K
        name=['X_1_update_fval_iter_',num2str(k)];
        plot(tests.(name),'Color',[0,(1-k)/(1-K),1+(k-1)/(1-K)],'LineWidth',2)
    end
    title('Fval history for X_1 update','FontSize',fs)
end

if isfield(tests,'X_2_update_fval_iter_1')
    count_figure=count_figure+1;
    if ishandle(count_figure)
        close(count_figure)
    end
    
    figure(count_figure)
    plot(tests.X_2_update_fval_iter_1,'b','LineWidth',2)
    hold on 
    for k =2:K
        name=['X_2_update_fval_iter_',num2str(k)];
        plot(tests.(name),'Color',[0,(1-k)/(1-K),1+(k-1)/(1-K)],'LineWidth',2)
    end
    title('Fval history for X_2 update','FontSize',fs)
end
end