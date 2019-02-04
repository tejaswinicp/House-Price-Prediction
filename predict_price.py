#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:51:18 2019

@author: tejaswinicp
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('Housing.csv')

X = data["price"] 
Y = data["lotsize"]
X = X.values.reshape(len(X), 1)
Y = Y.values.reshape(len(Y), 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
print(regr.score(X_train, Y_train))

svm = LinearSVC()
svm.fit(X_train, Y_train)
svm.predict(X_train)
print(svm.score(X_train, Y_train))

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
print(rf.score(X_train, Y_train))
#print(rf.score(X_test, y_test))