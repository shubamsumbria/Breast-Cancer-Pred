class data_pre:   
    def data_load(): #check for the availability of the dataset and change cwd if not found
        import os
        from pathlib import Path
        import pandas as pd
        if os.path.exists("data.csv"):
            print('Dataset Found!')
        else:
            print('Dataset not found. Now changing cwd to search for it.(will return back here if found in dataset folder)')
            os.chdir(str(Path(os.getcwd()).parent) + '\dataset')
            return pd.read_csv("data.csv")            
    def data_clean(df):
        df.drop('id',axis=1,inplace=True)
        df.drop('Unnamed: 32',axis=1,inplace=True)
        df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
        return df
    def X_y_split(df):
        X = df.drop(['diagnosis'],axis=1)
        y = df['diagnosis']
        return X,y
    def data_split_scale(X,y):
         #Splitting dataset into Train and Test Set
        from sklearn.model_selection import train_test_split
        X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=30)      
        #Feature Scaling using Standardization
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        X_train=ss.fit_transform(X_train)
        X_test=ss.fit_transform(X_test)      
        return X_train, X_test, y_train, y_test 

class feat:
    def feat1():
        from data_pre import data_clean as dc, data_load as dl, X_y_split as splt, data_split_scale as dss
        # All Features
        # Loading Dataset into Dataframe
        df=dl()
        df = dc(df) 
        X , y = splt(df)
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
        return feat_options[selection + 1]

class models:
    def lr(dat):
        #Logistic Regression
        from sklearn.linear_model import LogisticRegression
        import time as t
        start=t.time()
        lr=LogisticRegression()
        model_lr=lr.fit(dat[0],dat[2])
        stop = t.time()
        return model_lr, (stop-start)
    def dtc(dat):
        #Decision Tree Classifier
        import time as t
        from sklearn.tree import DecisionTreeClassifier
        start=t.time()
        dtc=DecisionTreeClassifier()
        model_dtc=dtc.fit(dat[0],dat[2])
        stop = t.time()
        return model_dtc, (stop-start)
    def rfc(dat):
        import time as t
        from sklearn.ensemble import RandomForestClassifier
        start=t.time()
        rfc=RandomForestClassifier()
        model_rfc = rfc.fit(dat[0],dat[2])
        stop = t.time()
        return model_rfc, (stop-start)
    def knn(dat):
        import time as t
        from sklearn.neighbors import KNeighborsClassifier
        start=t.time()
        knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        model_knn = knn.fit(dat[0],dat[2])
        stop = t.time()
        return model_knn, (stop-start)
    def svm_l(dat):
        import time as t
        from sklearn.svm import SVC
        start=t.time()
        svc_l = SVC(kernel = 'linear', random_state = 0)
        model_svc_l = svc_l.fit(dat[0],dat[2])
        stop = t.time()
        return model_svc_l, (stop-start)
    def svm_r(dat):
        import time as t
        from sklearn.svm import SVC
        start=t.time()
        svc_r = SVC(kernel = 'rbf', random_state = 0)
        model_svc_r = svc_r.fit(dat[0],dat[2])
        stop = t.time()
        return model_svc_r, (stop-start)
    def gnb(dat):
        import time as t
        from sklearn.naive_bayes import GaussianNB
        start=t.time()
        gnb = GaussianNB()
        model_gnb = gnb.fit(dat[0],dat[2])
        pred=model_gnb.predict(dat[1]); pred_prob=model_gnb.predict_proba(dat[1])
        stop = t.time()
        return model_gnb, (stop-start), pred, pred_prob