def corrwithdia(dfx):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(20,8))
    dfx.drop('diagnosis', axis=1).corrwith(dfx.diagnosis)\
    .plot(kind='bar', grid=True, \
          title="Correlation of Mean Features with Diagnosis",color="cornflowerblue");
def corrheat(dfx):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Mask for Heatmap
    mask = np.zeros_like(dfx.corr(), dtype = np.bool)
    mask[np.triu_indices_from(dfx.corr())] = True 
    # Correlation Matrix Heatmap including all features
    fig, ax = plt.subplots(figsize=(22, 10))
    ax = sns.heatmap(dfx.corr(),mask=mask,annot=True,linewidths=0.5,fmt=".2f",cmap="YlGn");
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5);
    ax.set_title("Correlation Matrix Heatmap including all features",fontsize=20)
    
def distwithdia(dfx):
    import numpy as np
    import matplotlib.pyplot as plt
    # Extracting Mean, Squared Error and Worst Features
    if dfx == df_mean:
        dfx_cols = list(dfx.columns[1:11])
    elif dfx == df_se:
        dfx_cols = list(dfx.columns[11:21])
    elif dfx == df_worst:
        dfx_cols = list(dfx.columns[21:])
    #Split into two Parts Based on Diagnosis
    dfM=dfx[dfx['diagnosis'] ==1]
    dfB=dfx[dfx['diagnosis'] ==0]
    # Nucleus Mean Features vs Diagnosis
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
    axes = axes.ravel()
    for idx,ax in enumerate(axes):
        ax.figure
        binwidth= (max(dfx[dfx_cols[idx]]) - min(dfx[dfx_cols[idx]]))/50
        ax.hist([dfM[dfx_cols[idx]],dfB[dfx_cols[idx]]], bins=np.arange(min(df[dfx_cols[idx]]), max(df[dfx_cols[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, label=['M','B'],color=['b','g'])
        ax.legend(loc='upper right')
        ax.set_title(dfxcols[idx])
    plt.tight_layout()
    plt.show()
    
def pplot(dfx):
    import seaborn as sns
    sns.pairplot(data=dfx, hue='diagnosis', palette='crest')
    