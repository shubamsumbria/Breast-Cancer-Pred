def perfor():
    from train_test import train_n_test as tnt
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix ,accuracy_score ,f1_score ,recall_score, precision_score, classification_report
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
    rg.plot(kind = 'bar', color=["red","salmon"])
    plt.title("Classification Report of {}".format(models[mdl_selection-1]))
    plt.legend(report.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
    plt.yticks(np.arange(0, 1, step=0.05))            
    
    
    return [acs,rs,fs,ps]

# mdl, time, pred, prob,y_test=out
accuracy_score, recall, f1, precision=perfor()
    