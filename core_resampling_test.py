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

start = 0
stop = 5
samples = 100

sto=100

carlo = 10*sto


fig, axs = plt.subplots(2,2, figsize = (6,8), sharey=True)
global_results = []
global_p95 = []
global_p5 = []

for i in extensions:
    print(".", end="")
    local_results = []
    for j in range(carlo):
        start = 0
        stop = 10

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
        y_uniform = np.sin(x_uniform)

        
        diff = y_uniform[2:-2]-y_reg[2:-2]
        result = np.mean(np.abs(diff))
        
        local_results.append(result)

        
    global_results.append(np.mean(local_results))
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))

axs[0,0].scatter(extensions, global_results, marker = '.', label='RNR',
               c='black', s=10,)

axs[0,0].scatter(extensions, global_p5, marker = '_', label='RNR',
               c='gray', s=10,)

axs[0,0].scatter(extensions, global_p95, marker = '_', label='RNR',
               c='gray', s=10)
axs[0,0].set_xlabel('Radius multiplier')
axs[0,0].set_ylabel('Mean Absolute Error')
axs[0,0].grid()

axs[0,0].set_axisbelow(True)
axs[0,0].title.set_text('RadiusNeighborsRegressor,\nweights=distance')
axs[0,0].axhline(y=0.199, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\naverage')
axs[0,0].axhline(y=0.101, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
axs[0,0].axhline(y=0.35, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')

print()

carlo = 10*sto

global_results = []
global_p95 = []
global_p5 = []

for i in extensions:
    print("$", end="")
    local_results = []
    for j in range(carlo):
        start = 0
        stop = 10

        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = np.sin(x)
        
        
        max_step = np.max(np.diff(x))
        extension = i

        reg = RadiusNeighborsRegressor(radius=max_step*extension,
                                       weights='uniform',
                                       algorithm = 'auto')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(start, stop, samples)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        

        y_uniform = np.sin(x_uniform)        
        diff = y_uniform[2:-2]-y_reg[2:-2]
        result = np.mean(np.abs(diff))
        
        local_results.append(result)

        
    global_results.append(np.mean(local_results))
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))

axs[0,1].scatter(extensions, global_results, marker = '.', label='RNR',
               c='black', s=10,)

axs[0,1].scatter(extensions, global_p5, marker = '_', label='RNR',
               c='gray', s=10,)

axs[0,1].scatter(extensions, global_p95, marker = '_', label='RNR',
               c='gray', s=10)
axs[0,1].set_xlabel('Radius multiplier')

axs[0,1].grid()
axs[0,1].set_axisbelow(True)
axs[0,1].title.set_text('RadiusNeighborsRegressor,\nweights=uniform')
axs[0,1].axhline(y=0.199, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\naverage')
axs[0,1].axhline(y=0.101, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
axs[0,1].axhline(y=0.35, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')
print()

carlo=10*sto
global_p95 = []
global_p5 = []
extensions = np.arange(1,51,1)
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
        
        y_uniform = np.sin(x_uniform)        
        diff = y_uniform[2:-2]-y_reg[2:-2]

        result = np.mean(np.abs(diff))
        
        local_results.append(result)

        
    global_results.append(np.mean(local_results))
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))
    
    
axs[1,0].scatter(extensions, global_results, marker = '.', label='mean',
               c='black', s=10)

axs[1,0].scatter(extensions, global_p5, marker = '_', label='5th/95th percentile',
               c='gray', s=10)

axs[1,0].scatter(extensions, global_p95, marker = '_',
               c='gray', s=10)
axs[1,0].set_xlabel('Neighbour count')

axs[1,0].grid()
axs[1,0].set_ylabel('Mean Absolute Error')
axs[1,0].set_axisbelow(True)
axs[1,0].title.set_text('KNeighborsRegressor,\nweights=distance')
axs[1,0].axhline(y=0.199, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\naverage')
axs[1,0].axhline(y=0.101, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
axs[1,0].axhline(y=0.35, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')
axs[1,0].legend()
print()

carlo=10*sto
global_p95 = []
global_p5 = []
extensions = np.arange(1,51,1)
global_results = []
for i in extensions:
    print("^", end="")
    local_results = []
    for j in range(carlo):
        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = np.sin(x)
        

        extension = i
        

        reg = KNeighborsRegressor(n_neighbors = i,
                                  weights='uniform')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(start, stop , samples)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        
        y_uniform = np.sin(x_uniform)        
        diff = y_uniform[2:-2]-y_reg[2:-2]

        result = np.mean(np.abs(diff))
        
        local_results.append(result)

        
    global_results.append(np.mean(local_results))
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))
    
    
axs[1,1].scatter(extensions, global_results, marker = '.', label='mean',
               c='black', s=10)

axs[1,1].scatter(extensions, global_p5, marker = '_', label='5th/95th percentile',
               c='gray', s=10)

axs[1,1].scatter(extensions, global_p95, marker = '_',
               c='gray', s=10)
axs[1,1].set_xlabel('Neighbour count')

axs[1,1].grid()

axs[1,1].set_axisbelow(True)
axs[1,1].title.set_text('KNeighborsRegressor,\nweights=uniform')
axs[1,1].axhline(y=0.199, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\naverage')
axs[1,1].axhline(y=0.101, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
axs[1,1].axhline(y=0.35, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')
#axs[1,1].legend()

# from statsmodels.nonparametric.smoothers_lowess import lowess



# carlo=2*sto
# global_p95 = []
# global_p5 = []
# extensions = np.linspace(0.01,0.4,100)
# global_results = []
# for i in extensions:
#     print("#", end="")
#     local_results = []
#     for j in range(carlo):
#         x = np.random.uniform(start, stop, samples)
#         x = np.sort(x)
#         y = np.sin(x)
        
#         max_step = np.max(np.diff(x))
#         extension = i
        
    
#         x_uniform = np.linspace(start, stop , samples)
        
#         y_reg = lowess(y, x, frac=i, xvals=x_uniform)
        
        
#         diff = y[10:-10]-y_reg[10:-10]

#         result = np.mean(np.abs(diff))
        
#         local_results.append(result)

        
#     global_results.append(np.mean(local_results))
#     global_p5.append(np.percentile(local_results,5))
#     global_p95.append(np.percentile(local_results,95))
    
    
# axs[2,0].scatter(extensions, global_results, marker = '.', label='mean',
#                c='black', s=10)

# axs[2,0].scatter(extensions, global_p5, marker = '_', label='5th/95th\npercentile',
#                c='gray', s=10)

# axs[2,0].scatter(extensions, global_p95, marker = '_', 
#                c='gray', s=10)
# axs[2,0].set_xlabel('Data fraction')

# axs[2,0].set_ylabel('Mean Absolute Error')
# axs[2,0].grid()
# axs[2,0].set_axisbelow(True)
# axs[2,0].title.set_text('LOWESS')



# axs[2,1].xaxis.set_visible(False)
# axs[2,1].yaxis.set_visible(False)
# axs[2,1].axis('off')
# axs[2,1].scatter([], [], marker = '.', label='mean',
#                c='black', s=10)

# axs[2,1].scatter([], [], marker = '_', label='5th/95th\npercentile',
#                c='gray', s=10)

# axs[2,1].scatter([], [], marker = '_', 
#                c='gray', s=10)
# axs[2,1].legend(loc='center')
plt.tight_layout()
plt.savefig('resampling.pdf')
plt.show()

#%%
np.random.seed(42)
start = 0
stop = 10
samples = 50
x = np.random.uniform(start, stop, samples)
x = np.sort(x)
y = np.sin(x)


xtrue = np.linspace(start, stop , 1000)
ytrue = np.sin(xtrue)
reg = KNeighborsRegressor(n_neighbors = 3,
                          weights='distance')
reg.fit(x.reshape(-1,1),y)

x_uniform = np.linspace(start, stop , samples)

y_reg = reg.predict(x_uniform.reshape(-1,1))

fig,ax = plt.subplots(1, figsize=(6,4))

#plt.scatter(x_uniform, y, s=10, c='blue', marker='v', label='raw, unindexed')
plt.plot(x_uniform, y, linewidth=0.5, c='blue', alpha=0.5,  marker='v',
         label='raw, unindexed', markersize=5)
plt.scatter(x_uniform, y_reg, s=30, marker='x', c='black', label='resampled,\nunindexed')

plt.scatter(x, y, s=5, marker='o', c='red', label='raw, indexed')
plt.plot(xtrue,ytrue, c='black', linestyle='--', linewidth=1, alpha=0.5, label='true signal')

handles,labels = ax.get_legend_handles_labels()

handles = [handles[1], handles[3], handles[0], handles[2]]
labels = [labels[1], labels[3], labels[0], labels[2]]
plt.legend(handles, labels)
plt.grid()
plt.xlabel('x\n[index]')
plt.ylabel('y=sin(x)')
plt.tight_layout()
plt.savefig('resampling_theory.pdf')

#%%

local_results = []
for j in range(1000):
    start = 0
    stop = 10
    samples = 100
    x = np.random.uniform(start, stop, samples)
    x = np.sort(x)
    y = np.sin(x)
    
    
    max_step = np.max(np.diff(x))


    
    x_uniform = np.linspace(start, stop, samples)
    
    y_uniform = np.sin(x_uniform)
    
    
    reg = KNeighborsRegressor(n_neighbors = 3,
                              weights='uniform')
    reg.fit(x.reshape(-1,1),y)
    
    x_uniform = np.linspace(start, stop , samples)
    
    y_reg = reg.predict(x_uniform.reshape(-1,1))
    
    
    # plt.plot(x_uniform, y_uniform)
    # plt.plot(x_uniform, y)
    # plt.plot(x_uniform, y_reg)
    # plt.show()
    diff = y - y_uniform
    result = np.mean(np.abs(diff))
    
    local_results.append(result)
    
print(f'Average: {np.average(local_results)}')
print(f'5th percentile {np.percentile(local_results,5)}')
print(f'95th percentile {np.percentile(local_results,95)}')