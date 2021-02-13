def perfor():
    from train_test import train_n_test as tnt
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix ,accuracy_score ,f1_score ,recall_score, precision_score, classification_report, roc_curve, auc, roc_auc_score
    import numpy as np
    
    out, y_test, mdl_selection=tnt()
    models=["Logistic Regression","Desicion Tree Classifier", "Random Forest Classifier", "KNN", "Liner SVM", "Kernal SVM", "Guassian NB"]
    cm_lr=confusion_matrix(y_test,out[2])
    sns.heatmap(cm_lr,annot=True, cmap="Reds")
    plt.title("Confusion Matrix for {}".format(models[mdl_selection-1]))
    
    acs = accuracy_score(y_test,out[2])
    rs = recall_score(y_test, out[2])
    fs = f1_score(y_test, out[2])
    ps = precision_score(y_test, out[2])
    
    #Report Bar Plot
    report = pd.DataFrame(classification_report(y_test, out[2], output_dict=True))
    rg = report.drop(report.index[3]).drop(report.columns[2:], axis=1)
    plt.style.use('seaborn')
    rg.plot(kind = 'bar', color=["red","salmon"])
    plt.title("Classification Report of {}".format(models[mdl_selection-1]))
    plt.legend(report.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
    plt.yticks(np.arange(0, 1.05, step=0.05))            
    
    print('\n\t The accuracy score of {} with given parameters is: {}%.'.format(models[mdl_selection-1],acs*100))
    print('\n\t The recall score of {} with given parameters is: {}%.'.format(models[mdl_selection-1],rs*100))
    print('\n\t The precision score of {} with given parameters is: {}%.'.format(models[mdl_selection-1],ps*100))
    print('\n\t The F1 score of {} with given parameters is: {}%.'.format(models[mdl_selection-1],fs*100))
    print('\n\t The training and testing time taken by {} with given parameters is: {} seconds.'.format(models[mdl_selection-1],out[1]))
    
    prob=out[3]
    prob=prob[:,1]
    #ROC 
    false_pos, true_pos, thresh=roc_curve(y_test, prob, pos_label=1)
    auc_score=roc_auc_score(y_test, prob)
    rand_pr=[0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, rand_pr, pos_label=1)
    
    plt.figure()
    plt.style.use('seaborn')
    plt.plot(false_pos, true_pos, linestyle='--',color='orange',label=models[mdl_selection-1])
    plt.plot(p_fpr, p_tpr, linestyle='--', color='green')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    
    return out[0],out[2], auc_score

trained_model,  pred, auc =perfor()