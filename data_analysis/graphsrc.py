def corrwithdia(dfx):
    import matplotlib.pyplot as plt
    import seaborn as sns
    name = str([x for x in globals() if globals()[x] is dfx][0])
    if name == 'df_mean':
        x = "Mean"
    elif name == 'df_se':
        x = "Squared Error"
    elif name == 'df_worst':
        x = "Worst"
    plt.figure(figsize=(20, 8))
    dfx.drop('diagnosis', axis=1).corrwith(dfx.diagnosis).plot(kind='bar', grid=True, title="Correlation of {} Features with Diagnosis".format(x), color="cornflowerblue");

def corrheat(dfx):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    name = str([x for x in globals() if globals()[x] is dfx][0])
    if name == 'df':
        x = "All"
    elif name == 'df_mean':
        x = "Mean"
    elif name == 'df_se':
        x = "Squared Error"
    elif name == 'df_worst':
        x = "Worst"
    # Mask for Heatmap
    mask = np.zeros_like(dfx.corr(), dtype=np.bool)
    mask[np.triu_indices_from(dfx.corr())] = True
    # Correlation Matrix Heatmap including all features
    if name == "df":
        q,r =20, 15
    else:
        q,r = 10, 5
    fig, ax = plt.subplots(figsize=(q, r))
    ax = sns.heatmap(dfx.corr(), mask=mask, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGn");
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5);
    ax.set_title("Correlation Matrix Heatmap including {} features".format(x), fontsize=10)

def distwithdia(dfx):
    import numpy as np
    import matplotlib.pyplot as plt
    name = str([x for x in globals() if globals()[x] is dfx][0])
    # Extracting Mean, Squared Error and Worst Columns
    if name == 'df_mean':
        dfx_cols = list(dfx.columns[1:11])
    elif name == 'df_se':
        dfx_cols = list(dfx.columns[11:21])
    elif name == 'df_worst':
        dfx_cols = list(dfx.columns[21:])
    # Split into two Parts Based on Diagnosis
    dfM = dfx[dfx['diagnosis'] == 1]
    dfB = dfx[dfx['diagnosis'] == 0]
    # Nucleus Features vs Diagnosis
    idx, ax = 0, 0
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        ax.figure
        binwidth = (max(dfx[dfx_cols[idx]]) - min(dfx[dfx_cols[idx]])) / 50
        ax.hist([dfM[dfx_cols[idx]], dfB[dfx_cols[idx]]],
                bins=np.arange(min(dfx[dfx_cols[idx]]), max(dfx[dfx_cols[idx]]) + binwidth, binwidth), alpha=0.5,
                stacked=True, label=['M', 'B'], color=['b', 'g'])
        ax.legend(loc='upper right')
        ax.set_title(dfx_cols[idx])
    plt.tight_layout()
    plt.show()

def pairplot(dfx):
    import seaborn as sns
    name = str([x for x in globals() if globals()[x] is dfx][0])
    if name == 'df_mean':
        x = "Mean"
    elif name == 'df_se':
        x = "Squared Error"
    elif name == 'df_worst':
        x = "Worst"
    sns.pairplot(data=dfx, hue='diagnosis', palette='crest', corner=True).fig.suptitle('Pairplot for {} Featrues'.format(x), fontsize = 20)
