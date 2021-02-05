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

# Checking Null and Missing Values
print("\nNull Values:\n", df.isnull().sum())
print("\nMissing Values:\n", df.isna().sum())

# Count Based On Diagnosis
df.diagnosis.value_counts()\
    .plot(kind="bar",width=0.1,color=["lightgreen","cornflowerblue"],legend=1,figsize=(8,5))
plt.xlabel("(0 = Benign) (1 = Malignant)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(["Benign"],fontsize=12)
plt.show()

# Extracting Mean, Squared Error and Worst Features with Diagnosis
df_mean = df[df.columns[:11]]
df_se = df.drop(df.columns[1:11], axis=1); df_se = df_se.drop(df_se.columns[11:], axis=1)
df_worst = df.drop(df.columns[1:21], axis=1)

# Correlation of Mean Features with Diagnosis
plt.figure(figsize=(20,8))
df_mean.drop('diagnosis', axis=1).corrwith(df_mean.diagnosis)\
    .plot(kind='bar', grid=True, \
          title="Correlation of Mean Features with Diagnosis",color="cornflowerblue");
        
# Correlation of Squared Error Features with Diagnosis
plt.figure(figsize=(20,8))
df_se.drop('diagnosis', axis=1).corrwith(df_se.diagnosis)\
    .plot(kind='bar', grid=True, \
          title="Correlation of Squared Error Features with Diagnosis",color="cornflowerblue");
# Correlation of Worst Features with Diagnosis
plt.figure(figsize=(20,8))
df_worst.drop('diagnosis', axis=1).corrwith(df_worst.diagnosis)\
    .plot(kind='bar', grid=True, \
          title="Correlation of Worst Error Features with Diagnosis",color="cornflowerblue");

# Correlation of each Feature with Diagnonis
plt.figure(figsize=(20,8))
df.drop('diagnosis', axis=1).corrwith(df.diagnosis)\
    .plot(kind='bar', grid=True, \
          title="Correlation of each Feature with Diagnonis",color="lightgreen");

# Extracting Mean, Squared Error and Worst Features
df_mean_cols = list(df.columns[1:11])
df_se_cols = list(df.columns[11:21])
df_worst_cols = list(df.columns[21:])
#Split into two Parts Based on Diagnosis
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]

# Nucleus Mean Features vs Diagnosis
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[df_mean_cols[idx]]) - min(df[df_mean_cols[idx]]))/50
    ax.hist([dfM[df_mean_cols[idx]],dfB[df_mean_cols[idx]]], bins=np.arange(min(df[df_mean_cols[idx]]), max(df[df_mean_cols[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, label=['M','B'],color=['b','g'])
    ax.legend(loc='upper right')
    ax.set_title(df_mean_cols[idx])
plt.tight_layout()
plt.show()

# Nucleus Squared Error Features vs Diagnosis
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[df_se_cols[idx]]) - min(df[df_se_cols[idx]]))/50
    ax.hist([dfM[df_se_cols[idx]],dfB[df_se_cols[idx]]], bins=np.arange(min(df[df_se_cols[idx]]), max(df[df_se_cols[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, label=['M','B'],color=['b','g'])
    ax.legend(loc='upper right')
    ax.set_title(df_se_cols[idx])
plt.tight_layout()
plt.show()

# Nucleus Worst Features vs Diagnosis
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[df_worst_cols[idx]]) - min(df[df_worst_cols[idx]]))/50
    ax.hist([dfM[df_worst_cols[idx]],dfB[df_worst_cols[idx]]], bins=np.arange(min(df[df_worst_cols[idx]]), max(df[df_worst_cols[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, label=['M','B'],color=['b','g'])
    ax.legend(loc='upper right')
    ax.set_title(df_worst_cols[idx])
plt.tight_layout()
plt.show()

#Checking Multicollinearity Between Different Features 
for i in df_mean , df_se , df_worst:
    sns.pairplot(data=i, hue='diagnosis', palette='crest')

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

#Multicollinearity is a problem as it undermines the significance of independent varibales.
#After verifying Multicollinearity, we can now remove some Highly Correlated features.

drop_cols = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
        'smoothness_worst', 'compactness_worst', 'concavity_worst','concave points_worst', 
        'symmetry_worst', 'fractal_dimension_worst','perimeter_mean','perimeter_se', 
        'area_mean', 'area_se','concavity_mean','concavity_se', 'concave points_mean', 
        'concave points_se']
df_final = df.drop(drop_cols, axis=1)

print(df_final.columns) # verify remaining columns

final_corr_matrix = df_final.corr() #Correlation Matrix of final Dataframe

# Mask for final Heatmap
mask = np.zeros_like(final_corr_matrix, dtype = np.bool)
mask[np.triu_indices_from(final_corr_matrix)] = True
#
fig, ax = plt.subplots(figsize=(22, 10))
ax = sns.heatmap(final_corr_matrix,mask=mask,annot=True,linewidths=0.5,fmt=".2f",cmap="YlGn");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5);
ax.set_title("Correlation Matrix Heatmap including all features")

#Saving final dataframe to csv file in dataset folder
df_final.to_csv("finaldata.csv", index=False)