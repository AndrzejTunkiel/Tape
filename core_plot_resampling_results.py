#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:54:29 2021

@author: llothar
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('deleteme.csv')

resampling = ['radius', 'knn']
weight = ['uniform', 'distance']


for r in resampling:
    for w in weight:
        dft = df.loc[(df['resampling'] == r) & (df['weight'] == w)]
        
        x = dft['K'].astype(int)
        y = dft['MAE'].rolling(window=5).mean()
        
        plt.plot(x,y, label=f'{r} n {w}')

plt.legend()
plt.show()

for r in resampling:
    for w in weight:
        dft = df.loc[(df['resampling'] == r) & (df['weight'] == w)]
        
        x = dft['K'].astype(int)
        y = dft['usable'].rolling(window=5).mean()
        
        plt.plot(x,y, label=f'{r} n {w}')

plt.legend()
plt.show()