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
for i in extensions:
    print(".", end="")
    local_results = []
    for j in range(10):
        x = np.random.uniform(0,10,200)
        x = np.sort(x)
        y = np.sin(x)
        
        
        max_step = np.max(np.diff(x))
        extension = i
        
        # plt.plot(y)
        # plt.plot(x*20,y)
        # plt.show()
        reg = RadiusNeighborsRegressor(radius=max_step*extension,
                                       weights='distance',
                                       algorithm = 'auto')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(0,10,100)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        
        # plt.plot(x_uniform, y_reg)
        # plt.show()
        
        diff = np.sin(x_uniform[10:-10])-y_reg[10:-10]
        # plt.plot(diff)
        result = np.mean(np.abs(diff))
        
        local_results.append(result)
        # print(f'Mean error is {np.mean(np.abs(diff))}')
        
    global_results.append(np.mean(local_results))

plt.scatter(extensions, global_results, marker = '.')
plt.grid()
#plt.ylim(0,0.008)

print()
extensions = np.linspace(1,10,50)
global_results = []
for i in extensions:
    print(",", end="")
    local_results = []
    for j in range(10):
        x = np.random.uniform(0,10,200)
        x = np.sort(x)
        y = np.sin(x)
        
        
        max_step = np.max(np.diff(x))
        extension = i
        

        reg = KNeighborsRegressor(n_neighbors = 5,
                                  weights='distance')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(0,10,100)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        
        # plt.plot(x_uniform, y_reg)
        # plt.scatter(x,y, alpha=0.6, c='red')
        # plt.show()
        
        diff = np.sin(x_uniform[10:-10])-y_reg[10:-10]
        # plt.plot(diff)
        result = np.mean(np.abs(diff))
        
        local_results.append(result)
        # print(f'Mean error is {np.mean(np.abs(diff))}')
        
    global_results.append(np.mean(local_results))

plt.scatter(extensions, global_results, marker = '.')
#plt.grid()
#plt.ylim(0,0.008)