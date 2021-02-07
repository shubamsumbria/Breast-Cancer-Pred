def train_n_test():
    from src import feat, models
    ft=feat.feature()
    models=[models.lr,models.dtc,models.rfc,models.knn,models.svc_l,models.svc_r,models.gnb]
    print("'\t The number '1' stands for 'LOGISTIC REGRESSION'. \n \t The number '2' stands for 'Decision Tree' . \n \t The number '3' stands for 'Random Forest Classifier'. \n \t The number '4' stands for 'KNN'. \n \t The number '5' stands for 'Liner SVM'. \n \t The number '6' stands for 'Kernal SVM'. \n \t The number '7' stands for 'Guassian NB'.'")
    mdl_selection=int(input("Please enter your selection for models: \t"))
    model=models[mdl_selection - 1]
    return model(ft),ft[3], mdl_selection
