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
    import pandas as pd
    X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)      
    #Feature Scaling using Standardization
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    X_train=ss.fit_transform(X_train)
    X_test=ss.fit_transform(X_test)      
    return pd.DataFrame(X_train), pd.DataFrame(X_test), y_train, y_test 