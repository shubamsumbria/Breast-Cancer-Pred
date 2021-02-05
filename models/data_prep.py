def data_prep():
    import pandas as pd
    import os
    from pathlib import Path
    #Searching dataset
    if os.path.exists("finaldata.csv"):
        print('Dataset Found!')
    else:
        print('Dataset not found. Now changing cwd to search for it.(will return back here if found in dataset folder)')
        os.chdir(str(Path(os.getcwd()).parent) + '\dataset')     
    # Loading Dataset into Dataframe
    df = pd.read_csv("finaldata.csv")
    X=df.drop(['diagnosis'],axis=1)
    y = df['diagnosis']
    #Splitting dataset into Train and Test Set
    from sklearn.model_selection import train_test_split
    X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=50)
    #Feature Scaling using Standardization
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    X_train=ss.fit_transform(X_train)
    X_test=ss.fit_transform(X_test)
    return X_train, X_test, y_train, y_test
