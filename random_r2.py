# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:30:43 2021

@author: lloth
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
np.random.seed(42)

X = np.linspace(0,1,5040)
y = np.cumsum(np.random.normal(size=5040))



X_train, X_test, y_train, y_test = train_test_split(
     X[:,np.newaxis], y, test_size=0.20, random_state=42)

regs = [GradientBoostingRegressor(random_state=42),
        SVR(),
        MLPRegressor(random_state=42),
        DecisionTreeRegressor(random_state=42)]

print("Incorrect split results:")
for reg in regs:    
    reg.fit(X_train,y_train)
    print(f'{reg.score(X_test,y_test):.3f}')
    
    
X_train = X[:2520,np.newaxis]
X_test = X[2520:,np.newaxis]
y_train = y[:2520]
y_test = y[2520:]

print("Correct split results:")
for reg in regs:    
    reg.fit(X_train,y_train)
    print(f'{reg.score(X_test,y_test):.3f}')