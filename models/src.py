class feat:
    def feat1():
        from data_pre import data_clean as dc, data_load as dl, X_y_split as splt, data_split_scale as dss
        # All Features
        # Loading Dataset into Dataframe
        df=dl()
        X , y = splt(dc(df))
        return dss(X,y)   
    def feat2():
       from data_pre import data_clean as dc, data_load as dl, X_y_split as splt, data_split_scale as dss      
       # Loading Dataset into Dataframe
       df=dl()
       df = dc(df)
       df_mean = df[df.columns[:11]]
       X , y = splt(df_mean)
       return dss(X,y)  
    def feat3():
       from data_pre import data_clean as dc, data_load as dl, X_y_split as splt, data_split_scale as dss
       # Loading Dataset into Dataframe
       df=dl()
       df = dc(df) 
       df_se = df.drop(df.columns[1:11], axis=1); df_se = df_se.drop(df_se.columns[11:], axis=1)
       X , y = splt(df_se)
       return dss(X,y)   
    def feat4():
       from data_pre import data_clean as dc, data_load as dl, X_y_split as splt, data_split_scale as dss
       # Loading Dataset into Dataframe
       df=dl()
       df = dc(df)
       df_worst = df.drop(df.columns[1:21], axis=1)
       X , y = splt(df_worst)
       return dss(X,y)   
    def feat5():
       from data_pre import data_clean as dc, data_load as dl, X_y_split as splt, data_split_scale as dss
       # Selected Features  
       # Loading Dataset into Dataframe
       df = dl()  
       df = dc(df)  
       drop_cols = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
            'symmetry_worst', 'fractal_dimension_worst','perimeter_mean','perimeter_se', 
             'area_mean', 'area_se','concavity_mean','concavity_se', 'concave points_mean', 
             'concave points_se']
       df_sf = df.drop(drop_cols, axis=1) 
       X , y = splt(df_sf)
       return dss(X,y)       
    def feature():
        print("'\t The number '1' stands for 'ALL- FEATURES'. \n \t The number '2' stands for 'MEAN- FEATURES' . \n \t The number '3' stands for 'SQUARED- ERROR FEATURES'. \n \t The number '4' stands for 'WORST- FEATURES'. \n \t The number '5' stands for 'SELECTED- FEATURES'.'")
        selection=input("\t Enter your choice of feature selection: \t")
        feat_options=[feat.feat1(), feat.feat2(), feat.feat3(), feat.feat4(), feat.feat5()]
        return feat_options[int(selection) - 1]

class models:
    def lr(dat):
        #Logistic Regression
        from sklearn.linear_model import LogisticRegression
        import time as t
        start=t.time()
        lr=LogisticRegression()
        model_lr=lr.fit(dat[0],dat[2])
        pred=model_lr.predict(dat[1]); pred_prob=model_lr.predict_proba(dat[1])
        stop = t.time()
        return model_lr, (stop-start),pred, pred_prob
    def dtc(dat):
        #Decision Tree Classifier
        import time as t
        from sklearn.tree import DecisionTreeClassifier
        start=t.time()
        dtc=DecisionTreeClassifier()
        model_dtc=dtc.fit(dat[0],dat[2])
        pred=model_dtc.predict(dat[1]); pred_prob=model_dtc.predict_proba(dat[1])
        stop = t.time()
        return model_dtc, (stop-start),pred, pred_prob
    def rfc(dat):
        import time as t
        from sklearn.ensemble import RandomForestClassifier
        start=t.time()
        rfc=RandomForestClassifier()
        model_rfc = rfc.fit(dat[0],dat[2])
        pred=model_rfc.predict(dat[1]); pred_prob=model_rfc.predict_proba(dat[1])
        stop = t.time()
        return model_rfc, (stop-start),pred, pred_prob
    def knn(dat):
        import time as t
        from sklearn.neighbors import KNeighborsClassifier
        start=t.time()
        knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        model_knn = knn.fit(dat[0],dat[2])
        pred=model_knn.predict(dat[1]); pred_prob=model_knn.predict_proba(dat[1])
        stop = t.time()
        return model_knn, (stop-start),pred, pred_prob
    def svc_l(dat):
        import time as t
        from sklearn.svm import SVC
        start=t.time()
        svc_l = SVC(kernel = 'linear', random_state = 0,probability=True)
        model_svc_l = svc_l.fit(dat[0],dat[2])
        pred=model_svc_l.predict(dat[1]); pred_prob=model_svc_l.predict_proba(dat[1])
        stop = t.time()
        return model_svc_l, (stop-start),pred, pred_prob
    def svc_r(dat):
        import time as t
        from sklearn.svm import SVC
        start=t.time()
        svc_r = SVC(kernel = 'rbf', random_state = 0,probability=True)
        model_svc_r = svc_r.fit(dat[0],dat[2])
        pred=model_svc_r.predict(dat[1]); pred_prob=model_svc_r.predict_proba(dat[1])
        stop = t.time()
        return model_svc_r, (stop-start),pred, pred_prob
    def gnb(dat):
        import time as t
        from sklearn.naive_bayes import GaussianNB
        start=t.time()
        gnb = GaussianNB()
        model_gnb = gnb.fit(dat[0],dat[2])
        pred=model_gnb.predict(dat[1]); pred_prob=model_gnb.predict_proba(dat[1])
        stop = t.time()
        return model_gnb, (stop-start), pred, pred_prob
    
    