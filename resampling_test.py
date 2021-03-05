# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:14:17 2021

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
#%%
extensions = np.linspace(1,10,50)
global_results = []
carlo = 100
start = 0
stop = 5
samples = 100
#%%
for i in extensions:
    print(".", end="")
    local_results = []
    for j in range(carlo):
        start = 0
        stop = 10
        samples = 100
        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = np.sin(x)
        
        
        max_step = np.max(np.diff(x))
        extension = i

        reg = RadiusNeighborsRegressor(radius=max_step*extension,
                                       weights='distance',
                                       algorithm = 'auto')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(start, stop, samples)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        

        
        diff = y[10:-10]-y_reg[10:-10]
        result = np.mean(np.abs(diff))
        
        local_results.append(result)

        
    global_results.append(np.mean(local_results))

plt.scatter(extensions, global_results, marker = '.', label='RNR', c='black')
plt.xlabel('Radius multiplier')
plt.ylabel('MAE')
plt.grid()

#%%
print()
carlo=100

extensions = np.arange(1,101,1)
global_results = []
for i in extensions:
    print(",", end="")
    local_results = []
    for j in range(carlo):
        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = np.sin(x)
        
        max_step = np.max(np.diff(x))
        extension = i
        

        reg = KNeighborsRegressor(n_neighbors = i,
                                  weights='distance')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(start, stop , samples)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        
        
        diff = y[10:-10]-y_reg[10:-10]

        result = np.mean(np.abs(diff))
        
        local_results.append(result)

        
    global_results.append(np.mean(local_results))

plt.scatter(extensions, global_results, marker = '.', label = 'KNN', c='black')
plt.xlabel('Neighbour count')
plt.ylabel('MAE')
plt.legend()

#%%

from statsmodels.nonparametric.smoothers_lowess import lowess



carlo=100

extensions = np.linspace(0.01,1,100)
global_results = []
for i in extensions:
    print(",", end="")
    local_results = []
    for j in range(carlo):
        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = np.sin(x)
        
        max_step = np.max(np.diff(x))
        extension = i
        
    
        x_uniform = np.linspace(start, stop , samples)
        
        y_reg = lowess(y, x, frac=i, xvals=x_uniform)
        
        
        diff = y[10:-10]-y_reg[10:-10]

        result = np.mean(np.abs(diff))
        
        local_results.append(result)

        
    global_results.append(np.mean(local_results))

plt.scatter(extensions, global_results, marker = '.', label = 'LOWESS', c='black')
plt.xlabel('Data fraction')
plt.ylabel('MAE')
plt.legend()
