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
    from src import sampling as s
    import pandas as pd
    X_tr, X_test,y_tr, y_test=train_test_split(X,y,test_size=0.3)      
    #Feature Scaling using Standardization
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    X_tr=ss.fit_transform(X_tr)
    X_test=ss.fit_transform(X_test)     
    print("'For 'Sampling strategies', I have 3 options. \n \t'1' stands for 'Upsampling'\n \t'2' stands for 'downsampling'. \n \t'3' stands for 'SMOTE''")
    samp_sel=int(input("Now enter your selection for sampling strategy: \t"))
    samp=[s.upsample, s.downsample, s.smote]
    temp=samp[samp_sel-1]
    X_train,y_train=temp(X_train=pd.DataFrame(X_tr), y_train=pd.DataFrame(y_tr))
    return pd.DataFrame(X_train), pd.DataFrame(X_test), y_train, y_test 