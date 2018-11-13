# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:51:46 2018

@author: Kiran.Boddupalli
"""

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
import pandas as pd
import numpy as np


        # Reading Data Set and creating Pandas Data Frame
Data = pd.read_csv('F:/6. EXPLORATION/2. R/GermanCredit.csv')

list(Data)

Data['Class'].value_counts()
Data['CLASS'] = np.where(Data['Class']=='Good', 1, 0)
Data['CLASS'].value_counts()

#### Creating Y and X (Independent variables)

Y = Data.CLASS
X = Data.drop(labels=['Class','CLASS'], axis=1)

# Slitting the data into Train and Test

seed = 7
test_size = 0.30
X_train, X_test, y_train, y_test =  train_test_split(X,Y, test_size=test_size, random_state=seed)


# Modeling Xtreem Gradient Boosting Algorithm
# Training on Train Data Set
model = XGBClassifier()
model.fit(X_train, y_train)
# PRedicting on Test Data Set
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# Accuracy check
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

### ROC Curve for Xtreem Gradient Boosting Algorithm
model_roc_auc = roc_auc_score(y_test, model.predict(X_test))
fprB_1, tprB_1, thresholdsB_1 = roc_curve(y_test, model.predict_proba(X_test)[:,1])


# Modeling Gradient Boosting Algorithm
GBbaseline = GradientBoostingClassifier()
GBbaseline.fit(X_train,y_train)
y_pred_GB = GBbaseline.predict(X_test)
predictions_GB = [round(value) for value in y_pred_GB]

accuracy_GB = accuracy_score(y_test, predictions_GB)
print(accuracy_GB)

### ROC Curve for Gradient Boosting Algorithm
GBbaseline_roc_auc = roc_auc_score(y_test, GBbaseline.predict(X_test))
fpr1_1, tpr1_1, thresholds1_1 = roc_curve(y_test, GBbaseline.predict_proba(X_test)[:,1])

### Logistic Regression 
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_log = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_log)
print(confusion_matrix)

accuracy_LOG = accuracy_score(y_test, y_pred_log)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_log))

### ROC Curve for Logistic Regression
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])


### Decision Tree Regression 
dtree=DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_pred_DTREE = dtree.predict(X_test)
print(classification_report(y_test, y_pred_DTREE))
accuracy_DTREE = accuracy_score(y_test, y_pred_DTREE)

DTREE_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
fpr1_2, tpr1_2, thresholds1_2 = roc_curve(y_test, dtree.predict_proba(X_test)[:,1])

### Random Forest

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred_RF=clf.predict(X_test)
print(classification_report(y_test, y_pred_RF))
accuracy_RF = accuracy_score(y_test, y_pred_RF)

RF_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr1_rf, tpr1_rf, thresholds1_rf = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

### KNN Algorithm

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  
y_pred_KNN = classifier.predict(X_test)  
print(confusion_matrix(y_test, y_pred_KNN))  
print(classification_report(y_test, y_pred_KNN))  
accuracy_KNN = accuracy_score(y_test, y_pred_KNN)
KNN_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr1_KNN, tpr1_KNN, thresholds1_KNN = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])

### Gussian Naive Bayes Algorithm
gnb = GaussianNB()
#gnb = BernoulliNB()
gnb.fit(X_train, y_train)  
y_pred_gnb = gnb.predict(X_test)  
print(confusion_matrix(y_test, y_pred_gnb))  
print(classification_report(y_test, y_pred_gnb))  
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
gnb_roc_auc = roc_auc_score(y_test, gnb.predict(X_test))
fpr1_gnb, tpr1_gnb, thresholds1_gnb = roc_curve(y_test, gnb.predict_proba(X_test)[:,1])

### SVM Model
SVMMOD = svm.SVC(kernel='sigmoid',probability=True)
SVMMOD.fit(X_train, y_train)  
y_pred_SVM = SVMMOD.predict(X_test)  
print(confusion_matrix(y_test, y_pred_SVM))  
print(classification_report(y_test, y_pred_SVM))  
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
SVM_roc_auc = roc_auc_score(y_test, SVMMOD.predict(X_test))
fpr1_SVM, tpr1_SVM, thresholds1_SVM = roc_curve(y_test, SVMMOD.predict_proba(X_test)[:,1])


### Model Accuracies
print ("XGB Accuracy : " ,accuracy)
print ("GB Accuracy : " ,accuracy_GB)
print ("LOGIT Accuracy : " ,accuracy_LOG)
print ("DTREE Accuracy : " ,accuracy_DTREE)
print ("RFOREST Accuracy : " ,accuracy_RF)
print ("KNN Accuracy : " ,accuracy_KNN)
print ("NAIVEBAYES Accuracy : " ,accuracy_gnb)
print ("SVM Accuracy : " ,accuracy_SVM)


### Plot All the ROC curves in one plot

plt.figure()
plt.plot(fprB_1, tprB_1, label='XGBM Baseline (area = %0.2f)' % model_roc_auc)
plt.plot(fpr1_1, tpr1_1, label='GBM Model 1 (area = %0.2f)' % GBbaseline_roc_auc)
plt.plot(fpr_log, tpr_log, label='Logit Model  (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr1_2, tpr1_2, label='DTREE Model  (area = %0.2f)' % DTREE_roc_auc)
plt.plot(fpr1_rf, tpr1_rf, label='RFOREST Model  (area = %0.2f)' % RF_roc_auc)
plt.plot(fpr1_KNN, tpr1_KNN, label='KNN Model  (area = %0.2f)' % KNN_roc_auc)
plt.plot(fpr1_gnb, tpr1_gnb, label='GNB Model  (area = %0.2f)' % gnb_roc_auc)
plt.plot(fpr1_SVM, tpr1_SVM, label='SVM Model  (area = %0.2f)' % SVM_roc_auc)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

