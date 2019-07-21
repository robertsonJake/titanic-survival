#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:01:54 2019

@author: jakerobertson
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pp

#%%
genders = pd.read_csv('gender_submission.csv')
test = pd.read_csv('test.csv')
df = pd.read_csv('train.csv')

#%%
class_dummies = pd.get_dummies(df['Pclass'])
class_dummies.columns = ['Class ' + str(i) for i in class_dummies.columns]
df = df.join(class_dummies)
df = df.join(pd.get_dummies(df['Sex']))
df = df.join(pd.get_dummies(df['Embarked']))
print(df.columns)
class_dummies = pd.get_dummies(test['Pclass'])
class_dummies.columns = ['Class ' + str(i) for i in class_dummies.columns]
test = test.join(class_dummies)
test = test.join(pd.get_dummies(test['Sex']))
test = test.join(pd.get_dummies(test['Embarked']))
test = test.merge(genders,how='inner',left_on=['PassengerId'],right_on=['PassengerId'])
print(test.columns)
#%%
df = df[['Class 1','Class 2','Class 3','female','male',
         'Age','SibSp','Embarked','Survived','C','Q','S']].dropna()
test = test[['Class 1','Class 2','Class 3','female','male',
         'Age','SibSp','Embarked','Survived','C','Q','S']].dropna()
X_train = df[['Class 1','Class 2','Class 3',
              'female','male','Age','SibSp']]
y_train = df['Survived']
X_test = test[X_train.columns]
y_test = test['Survived']

from sklearn.preprocessing import StandardScaler
#I scale here even though I don't need to for trees because I use SVC later
X_new = pd.concat([X_train,X_test],ignore_index=True)
scaler = StandardScaler()
X = scaler.fit_transform(X_new)
#X = X_new
y = pd.concat([y_train,y_test],ignore_index=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

tree= DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print("DECISION TREE")
print(classification_report(y_test,y_pred))
print('CV score: ',np.mean(cross_val_score(tree,X,y,scoring='f1_macro')))
print("-----")

tree = RandomForestClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print("RANDOM FOREST")
print(classification_report(y_test,y_pred))
print('CV score: ',np.mean(cross_val_score(tree,X,y,scoring='f1_macro')))
print('-----')

s = SVC()
s.fit(X_train,y_train)
y_pred = s.predict(X_test)
print("SVM")
print(classification_report(y_test,y_pred))
print('CV score: ',np.mean(cross_val_score(s,X,y,scoring='f1_macro')))
print('-----')

#%%
print("-----")

X = X_new[['male','female','Age']]
X = scaler.fit_transform(X)
tree = RandomForestClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print("RANDOM FOREST LIMITED FEATURES")
print(classification_report(y_test,y_pred))
print('CV score: ',np.mean(cross_val_score(tree,X,y,scoring='f1_macro')))

print("-----")

s = SVC()
s.fit(X_train,y_train)
y_pred = s.predict(X_test)
print("SVM")
print(classification_report(y_test,y_pred))
print('CV score: ',np.mean(cross_val_score(s,X,y,scoring='f1_macro')))
print('-----')

#%%
from sklearn.model_selection import GridSearchCV
tree = RandomForestClassifier()
params = {
        'n_estimators':[5,10,20],
        'max_features':[3,5,7]}
grid_search = GridSearchCV(tree,param_grid=params)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test,y_pred))
print('CV score: ',np.mean(cross_val_score(tree,X,y,scoring='f1_macro')))
print('-----')
