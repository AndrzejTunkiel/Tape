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
extensions = np.linspace(1,10,100)
global_results = []
global_p95 = []
global_p5 = []

start = 0
stop = 5
samples = 1000

carlo = 1000
fig, axs = plt.subplots(1,3, figsize = (9,3), sharey=True)

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
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))

axs[0].scatter(extensions, global_results, marker = '.', label='RNR',
               c='black', s=10,)

axs[0].scatter(extensions, global_p5, marker = '_', label='RNR',
               c='gray', s=10,)

axs[0].scatter(extensions, global_p95, marker = '_', label='RNR',
               c='gray', s=10)
axs[0].set_xlabel('Radius multiplier')
axs[0].set_ylabel('MAE')
axs[0].grid()
axs[0].set_axisbelow(True)
#plt.legend()
#plt.ylim(0.175,0.3)
#plt.show()
#%%
print()
carlo=1000
global_p95 = []
global_p5 = []
extensions = np.arange(1,101,1)
global_results = []
for i in extensions:
    print(",", end="")
    local_results = []
    for j in range(carlo):
        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = np.sin(x)
        

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
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))
    
    
axs[1].scatter(extensions, global_results, marker = '.', label='RNR',
               c='black', s=10)

axs[1].scatter(extensions, global_p5, marker = '_', label='RNR',
               c='gray', s=10)

axs[1].scatter(extensions, global_p95, marker = '_', label='RNR',
               c='gray', s=10)
axs[1].set_xlabel('Neighbour count')
#plt.ylabel('MAE')
#plt.legend()
axs[1].grid()
axs[1].set_axisbelow(True)
#plt.ylim(0.15,0.3)
#plt.show()
#%%

from statsmodels.nonparametric.smoothers_lowess import lowess



carlo=200
global_p95 = []
global_p5 = []
extensions = np.linspace(0.01,0.4,100)
global_results = []
for i in extensions:
    print("#", end="")
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
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))
    
    
axs[2].scatter(extensions, global_results, marker = '.', label='mean',
               c='black', s=10)

axs[2].scatter(extensions, global_p5, marker = '_', label='5th/95th\npercentile',
               c='gray', s=10)

axs[2].scatter(extensions, global_p95, marker = '_', 
               c='gray', s=10)
axs[2].set_xlabel('Data fraction')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.ylabel('MAE')
#plt.legend()
axs[2].grid()
axs[2].set_axisbelow(True)
#plt.ylim(0.15,0.3)
plt.tight_layout()
plt.savefig('resampling.pdf')
plt.show()