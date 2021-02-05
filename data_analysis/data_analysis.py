import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
#Now search for the dataset
if os.path.exists("data.csv"):
    print('Dataset Found!')
else:
    print('Dataset not found. Now changing cwd to search for it.(will return back here if found in dataset folder)')
    os.chdir(str(Path(os.getcwd()).parent) + '\dataset')
# Loading Dataset into Dataframe
df = pd.read_csv("data.csv")
# cleaning Extra Columns
df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
# Mapping values of Diagnosis
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
# Count Based On Diagnosis
df.diagnosis.value_counts()\
    .plot(kind="bar",width=0.1,color=["lightgreen","cornflowerblue"],legend=1,figsize=(8,5))
plt.xlabel("(0 = Benign) (1 = Malignant)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(["Benign"],fontsize=12)
plt.show()
corr_matrix = df.corr() #Correlation Matrix
# Mask for Heatmap
mask = np.zeros_like(corr_matrix, dtype = np.bool)
mask[np.triu_indices_from(corr_matrix)] = True
# Correlation Matrix Heatmap including all features
fig, ax = plt.subplots(figsize=(22, 10))
ax = sns.heatmap(corr_matrix,mask=mask,annot=True,linewidths=0.5,fmt=".2f",cmap="YlGn");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5);
ax.set_title("Correlation Matrix Heatmap including all features")