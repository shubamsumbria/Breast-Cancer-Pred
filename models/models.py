import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time as t
from sklearn.metrics import confusion_matrix ,accuracy_score ,f1_score ,recall_score, precision_score, classification_report
from data_prep import data_prep as dp
X_train, X_test, y_train, y_test = dp()


#Logistic Regression
start=t.time()
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model_lr=lr.fit(X_train,y_train)
pred_lr=model_lr.predict(X_test)
stop = t.time()
time_lr = (stop - start)

#Confusion Matirx Plot
cm_lr=confusion_matrix(y_test,pred_lr)
sns.heatmap(cm_lr,annot=True, cmap="GnBu")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

#Accuracy Score, Recall Score, f1 Score and Precision Score
acs_lr = accuracy_score(y_test,pred_lr)
rs_lr = recall_score(y_test, pred_lr)
fs_lr = f1_score(y_test, pred_lr)
ps_lr = precision_score(y_test, pred_lr)

#Report Bar Plot
report_lr = pd.DataFrame(classification_report(y_test, pred_lr, output_dict=True))
rg_lr = report_lr.drop(report_lr.index[3]).drop(report_lr.columns[2:], axis=1)
rg_lr.plot(kind = 'bar', color=["lightgreen","cornflowerblue"])
plt.title("Classification Report of Logistic Regression")
plt.legend(report_lr.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()


#Decision Tree Classifier
start=t.time()
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
model_dtc=dtc.fit(X_train,y_train)
pred_dtc=model_dtc.predict(X_test)
stop = t.time()
time_dtc = (stop - start)

#Confusion Matirx Plot
cm_dtc=confusion_matrix(y_test,pred_dtc)
sns.heatmap(cm_dtc,annot=True,  cmap="GnBu")
plt.title("Confusion Matrix for Decision Tree Classifier")
plt.show()

#Accuracy Score, Recall Score, f1 Score and Precision Score
acs_dtc = accuracy_score(y_test,pred_dtc)
rs_dtc = recall_score(y_test, pred_dtc)
fs_dtc = f1_score(y_test, pred_dtc)
ps_dtc = precision_score(y_test, pred_dtc)

#Report Bar Plot
report_dtc = pd.DataFrame(classification_report(y_test, pred_dtc, output_dict=True))
rg_dtc = report_dtc.drop(report_dtc.index[3]).drop(report_dtc.columns[2:], axis=1)
rg_dtc.plot(kind = 'bar', color=["lightgreen","cornflowerblue"])
plt.title("Classification Report of Desicion Tree Classifier")
plt.legend(report_dtc.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()


#Random Forest Classifier
start=t.time()
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
model_rfc = rfc.fit(X_train, y_train)
pred_rfc = model_rfc.predict(X_test)
stop = t.time()
time_rfc = (stop - start)

#Confusion Matirx Plot
cm_rfc=confusion_matrix(y_test,pred_rfc)
sns.heatmap(cm_rfc,annot=True,  cmap="GnBu")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()

#Accuracy Score, Recall Score, f1 Score and Precision Score
acs_rfc = accuracy_score(y_test,pred_rfc)
rs_rfc = recall_score(y_test, pred_rfc)
fs_rfc = f1_score(y_test, pred_rfc)
ps_rfc = precision_score(y_test, pred_rfc)

#Report Bar Plot
report_rfc = pd.DataFrame(classification_report(y_test, pred_rfc, output_dict=True))
rg_rfc = report_rfc.drop(report_rfc.index[3]).drop(report_rfc.columns[2:], axis=1)
rg_rfc.plot(kind = 'bar', color=["lightgreen","cornflowerblue"])
plt.title("Classification Report of Random Forest Classifier")
plt.legend(report_rfc.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()


#KNeighborsClassifier
start=t.time()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model_knn = knn.fit(X_train, y_train)
pred_knn = model_knn.predict(X_test)
stop = t.time()
time_knn = (stop - start)

#Confusion Matirx Plot
cm_knn=confusion_matrix(y_test,pred_knn)
sns.heatmap(cm_knn,annot=True,  cmap="GnBu")
plt.title("Confusion Matrix for KNeighborsClassifier")
plt.show()

#Accuracy Score, Recall Score, f1 Score and Precision Score
acs_knn = accuracy_score(y_test,pred_knn)
rs_knn = recall_score(y_test, pred_knn)
fs_knn = f1_score(y_test, pred_knn)
ps_knn = precision_score(y_test, pred_knn)

#Report Bar Plot
report_knn = pd.DataFrame(classification_report(y_test, pred_knn, output_dict=True))
rg_knn = report_knn.drop(report_knn.index[3]).drop(report_knn.columns[2:], axis=1)
rg_knn.plot(kind = 'bar', color=["lightgreen","cornflowerblue"])
plt.title("Classification Report of KNeighborsClassifier")
plt.legend(report_knn.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()


#Linear SVM
start=t.time()
from sklearn.svm import SVC
svc_l = SVC(kernel = 'linear', random_state = 0)
model_svc_l = svc_l.fit(X_train, y_train)
pred_svc_l = model_svc_l.predict(X_test)
stop = t.time()
time_svc_l = (stop - start)

#Confusion Matirx Plot
cm_svc_l=confusion_matrix(y_test,pred_svc_l)
sns.heatmap(cm_svc_l,annot=True,  cmap="GnBu")
plt.title("Confusion Matrix for Linear SVM")
plt.show()

#Accuracy Score, Recall Score, f1 Score and Precision Score
acs_svc_l = accuracy_score(y_test,pred_svc_l)
rs_svc_l = recall_score(y_test, pred_svc_l)
fs_svc_l = f1_score(y_test, pred_svc_l)
ps_svc_l = precision_score(y_test, pred_svc_l)

#Report Bar Plot
report_svc_l = pd.DataFrame(classification_report(y_test, pred_svc_l, output_dict=True))
rg_svc_l = report_svc_l.drop(report_svc_l.index[3]).drop(report_svc_l.columns[2:], axis=1)
rg_svc_l.plot(kind = 'bar', color=["lightgreen","cornflowerblue"])
plt.title("Classification Report of Linear SVM")
plt.legend(report_svc_l.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()


#Kernel SVM
start=t.time()
from sklearn.svm import SVC
svc_r = SVC(kernel = 'rbf', random_state = 0)
model_svc_r = svc_r.fit(X_train, y_train)
pred_svc_r = model_svc_r.predict(X_test)
stop = t.time()
time_svc_r = (stop - start)

#Confusion Matirx Plot
cm_svc_r=confusion_matrix(y_test,pred_rfc)
sns.heatmap(cm_svc_r,annot=True,  cmap="GnBu")
plt.title("Confusion Matrix for Kernel SVM")
plt.show()

#Accuracy Score, Recall Score, f1 Score and Precision Score
acs_svc_r = accuracy_score(y_test,pred_svc_r)
rs_svc_r = recall_score(y_test, pred_svc_r)
fs_svc_r = f1_score(y_test, pred_svc_r)
ps_svc_r = precision_score(y_test, pred_svc_r)

#Report Bar Plot
report_svc_r = pd.DataFrame(classification_report(y_test, pred_svc_r, output_dict=True))
rg_svc_r = report_svc_r.drop(report_svc_r.index[3]).drop(report_svc_r.columns[2:], axis=1)
rg_svc_r.plot(kind = 'bar', color=["lightgreen","cornflowerblue"])
plt.title("Classification Report of Kernel SVM")
plt.legend(report_svc_r.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()


#GaussianNB
start=t.time()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model_gnb = gnb.fit(X_train, y_train)
pred_gnb = model_gnb.predict(X_test)
stop = t.time()
time_gnb = (stop - start)

#Confusion Matirx Plot
cm_gnb=confusion_matrix(y_test,pred_rfc)
sns.heatmap(cm_gnb,annot=True, cmap="GnBu")
plt.title("Confusion Matrix for GaussianNB")
plt.show()

#Accuracy Score, Recall Score, f1 Score and Precision Score
acs_gnb = accuracy_score(y_test,pred_gnb)
rs_gnb = recall_score(y_test, pred_gnb)
fs_gnb = f1_score(y_test, pred_gnb)
ps_gnb = precision_score(y_test, pred_gnb)

#Report Bar Plot
report_gnb= pd.DataFrame(classification_report(y_test, pred_gnb, output_dict=True))
rg_gnb = report_gnb.drop(report_gnb.index[3]).drop(report_gnb.columns[2:], axis=1)
rg_gnb.plot(kind = 'bar', color=["lightgreen","cornflowerblue"])
plt.title("Classification Report of GaussianNB")
plt.legend(report_gnb.columns, ncol=2 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()


#Performance Comparison
prfrm_comp = {"Logistic Regression": [time_lr,acs_lr,rs_lr,fs_lr,ps_lr],
             "Decision Tree Classifier":[time_dtc,acs_dtc,rs_dtc,fs_dtc,ps_dtc],
             "Random Forest Classifier": [time_rfc,acs_rfc,rs_rfc,fs_rfc,ps_rfc],
             "KNeighborsClassifier": [time_knn,acs_knn,rs_knn,fs_knn,ps_knn],
             "Linear SVM": [time_svc_l, acs_svc_l,rs_svc_l,fs_svc_l,ps_svc_l],
             "Kernel SVM": [time_svc_r,acs_svc_r,rs_svc_r,fs_svc_r,ps_svc_r],
             "GaussianNB": [time_gnb,acs_gnb,rs_gnb,fs_gnb,ps_gnb]}

#Performance Comparison Dictionary to Performance Dataframe
prfrm_df = pd.DataFrame.from_dict(prfrm_comp)

#Performance Comparison Bar Plot
prfrm_df[1:].plot(kind = "bar",width=0.8, color=["tomato","coral","yellow","lightgreen","deepskyblue","cornflowerblue","mediumslateblue"])
plt.title("Performance Comparison of all Classifiers", fontsize=12)
plt.xlabel(("1: Accuracy    2: Recall Score    3: F1 Score    4:Precision Score"), fontsize=12)
plt.legend(prfrm_df.columns, ncol=4 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()

#Time Comparison Bar Plot
prfrm_df[:1].plot(kind = "bar",width=0.8, color=["tomato","coral","yellow","lightgreen","deepskyblue","cornflowerblue","mediumslateblue"])
plt.title("Time Comparison of all Classifiers", fontsize=12)
plt.xlabel(("0: Time"), fontsize=12)
plt.legend(prfrm_df.columns, ncol=4 ,loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.yticks(np.arange(0, 0.16, step=0.01))
plt.show()