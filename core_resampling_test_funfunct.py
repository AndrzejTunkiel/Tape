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
plt.style.use(['science','no-latex'])
#%%


def sawtooth(x):
    from scipy.signal import sawtooth
    return sawtooth(x,width = 0.5)


# def sawtooth(x):
#     from scipy.signal import gausspulse
#     x = (x-5)/20
#     i, q, e = gausspulse(x, fc=5, retquad=True, retenv=True)
    
#     return q


# def sawtooth(x):
#     return np.sin(x)
#%%
start = 0
stop = 5
samples = 100
RMSs = []
for i in range(100):
    x = np.random.uniform(start, stop, samples)
    x = np.sort(x)
    y = np.sin(x)
    
    x_uniform = np.linspace(start, stop, samples)
    y_uniform = np.sin(x_uniform)
    RMS = np.power(np.sum(np.power((y - y_uniform),2))/samples,0.5)
    RMSs.append(RMS)

noremean = np.mean(RMSs)
nore5 = np.percentile(RMSs,5)
nore95 = np.percentile(RMSs,95)

print([noremean, nore5, nore95])
#%%

def myr2multi(x_start, y_start, x_stop, y_stop, data_x, data_y, res):
  try:
      loc_results = []
      x_range = np.linspace(x_start, x_stop, res+1)[:-1]
      y_range = np.linspace(y_start, y_stop, res+1)[:-1]
      
      for i in range(res):
        x = x_range[i]
        y = y_range[i]
        x1 = np.max(data_x[data_x <= x])
        x2 = np.min(data_x[data_x > x])
        
        loc1 = np.where(data_x == x1)
        loc2 = np.where(data_x == x2)
        
        y1 = data_y[loc1][-1]
        y2 = data_y[loc2][0]
        
        
        
        m = (y1-y2)/(x1-x2)
        b = (x1*y2 - x2*y1)/(x1-x2)
        
        
        y_inter = m * x + b

        loc_results.append(np.power(y-y_inter, 2))
        
      return loc_results
  except:
    return [0]*res
    print('oops')


res = 10
#%%
extensions = np.linspace(1,10,100)

start = 0
stop = 5
samples = 100

sto=1

carlo = 10*sto


fig, axs = plt.subplots(2,2, figsize = (5,6), sharey=True)
global_results = []
global_p95 = []
global_p5 = []

areas_global = []
for i in extensions:
    print(".", end="")
    local_results = []
    areas_local = []
    for j in range(carlo):
        start = 0
        stop = 10

        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = sawtooth(x)
        
        
        max_step = np.max(np.diff(x))
        extension = i

        reg = RadiusNeighborsRegressor(radius=max_step*extension,
                                       weights='distance',
                                       algorithm = 'auto')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(start, stop, samples)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        y_uniform = sawtooth(x_uniform)

        delta_sq = np.power((y_uniform[2:-2]-y_reg[2:-2]),2)
        diff = np.power(np.sum(delta_sq)/len(delta_sq),0.5)
        result = diff
        
        local_results.append(result)
        
        totals = []
        newdata = np.rot90([x_uniform,y_reg])
        for j in range(1,len(newdata)):
                  x_start = newdata[j-1][0]
                  y_start = newdata[j-1][1]
                  x_stop = newdata[j][0]
                  y_stop = newdata[j][1]
                  r2result = myr2multi(x_start, y_start,
                                          x_stop, y_stop,
                                          x, y,
                                          res)

                  totals.append(r2result) # added /np.mean(raw)
              
        totals = np.asarray(totals)
                
        Area_poly = np.power((np.sum(totals)/totals.size),0.5)

        areas_local.append(Area_poly)

    areas_global.append((areas_local))    
    
    global_results.append(np.mean(local_results))
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))

g1 = global_results.copy()

axs[0,0].scatter(extensions, global_results, marker = '.', 
               c='black', s=10,)

axs[0,0].scatter(extensions, global_p5, marker = '_', 
               c='gray', s=10,)

axs[0,0].scatter(extensions, global_p95, marker = '_',
               c='gray', s=10)
axs[0,0].set_xlabel('Radius multiplier')
axs[0,0].set_ylabel('Root Mean Square')
axs[0,0].grid()

axs[0,0].set_axisbelow(True)
axs[0,0].title.set_text('RadiusNeighbors\nRegressor,\nweights=distance')
axs[0,0].axhline(y=noremean, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\naverage')
axs[0,0].axhline(y=nore95, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
axs[0,0].axhline(y=nore5, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')
axs[0,0].legend()
print()

carlo = 10*sto

global_results = []
global_p95 = []
global_p5 = []



for i in extensions:
    print("$", end="")
    local_results = []
    areas_local = []
    for j in range(carlo):
        start = 0
        stop = 10

        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = sawtooth(x)
        
        
        max_step = np.max(np.diff(x))
        extension = i

        reg = RadiusNeighborsRegressor(radius=max_step*extension,
                                       weights='uniform',
                                       algorithm = 'auto')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(start, stop, samples)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        

        y_uniform = sawtooth(x_uniform)        
        delta_sq = np.power((y_uniform[2:-2]-y_reg[2:-2]),2)
        diff = np.power(np.sum(delta_sq)/len(delta_sq),0.5)
        result = diff
        
        local_results.append(result)
        totals = []
        newdata = np.rot90([x_uniform,y_reg])
        
        for j in range(1,len(newdata)):
                  x_start = newdata[j-1][0]
                  y_start = newdata[j-1][1]
                  x_stop = newdata[j][0]
                  y_stop = newdata[j][1]
                  r2result = myr2multi(x_start, y_start,
                                          x_stop, y_stop,
                                          x, y,
                                          res)

                  totals.append(r2result) # added /np.mean(raw)
              
        totals = np.asarray(totals)
                
        Area_poly = np.power((np.sum(totals)/totals.size),0.5)

        areas_local.append(Area_poly)

    areas_global.append((areas_local))    
        
    global_results.append(np.mean(local_results))
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))

g2 = global_results.copy()
axs[0,1].scatter(extensions, global_results, marker = '.', label='RNR',
               c='black', s=10,)

axs[0,1].scatter(extensions, global_p5, marker = '_', label='RNR',
               c='gray', s=10,)

axs[0,1].scatter(extensions, global_p95, marker = '_', label='RNR',
               c='gray', s=10)
axs[0,1].set_xlabel('Radius multiplier')

axs[0,1].grid()
axs[0,1].set_axisbelow(True)
axs[0,1].title.set_text('RadiusNeighbors\nRegressor,\nweights=uniform')
axs[0,1].axhline(y=noremean, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\naverage')
axs[0,1].axhline(y=nore95, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
axs[0,1].axhline(y=nore5, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')
print()

carlo=10*sto
global_p95 = []
global_p5 = []
extensions = np.arange(1,51,1)
global_results = []
for i in extensions:
    print(",", end="")
    local_results = []
    areas_local = []
    for j in range(carlo):
        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = sawtooth(x)
        

        extension = i
        

        reg = KNeighborsRegressor(n_neighbors = i,
                                  weights='distance')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(start, stop , samples)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        
        y_uniform = sawtooth(x_uniform)        
        diff = y_uniform[2:-2]-y_reg[2:-2]
        delta_sq = np.power((y_uniform[2:-2]-y_reg[2:-2]),2)
        diff = np.power(np.sum(delta_sq)/len(delta_sq),0.5)
        result = diff
        
        local_results.append(result)
        totals = []
        newdata = np.rot90([x_uniform,y_reg])
        for j in range(1,len(newdata)):
                  x_start = newdata[j-1][0]
                  y_start = newdata[j-1][1]
                  x_stop = newdata[j][0]
                  y_stop = newdata[j][1]
                  r2result = myr2multi(x_start, y_start,
                                          x_stop, y_stop,
                                          x, y,
                                          res)

                  totals.append(r2result) # added /np.mean(raw)
              
        totals = np.asarray(totals)
                
        Area_poly = np.power((np.sum(totals)/totals.size),0.5)

        areas_local.append(Area_poly)

    areas_global.append((areas_local))    
        
    global_results.append(np.mean(local_results))
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))
    
g3 = global_results.copy()   
axs[1,0].scatter(extensions, global_results, marker = '.', #label='mean',
               c='black', s=10)

axs[1,0].scatter(extensions, global_p5, marker = '_', #label='5th/95th percentile',
               c='gray', s=10)

axs[1,0].scatter(extensions, global_p95, marker = '_',
               c='gray', s=10)
axs[1,0].set_xlabel('Neighbour count')

axs[1,0].grid()
axs[1,0].set_ylabel('Root Mean Square')
axs[1,0].set_axisbelow(True)
axs[1,0].title.set_text('KNeighbors\nRegressor,\nweights=distance')
axs[1,0].axhline(y=noremean, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\naverage')
axs[1,0].axhline(y=nore95, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
axs[1,0].axhline(y=nore5, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')
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
    areas_local = []
    for j in range(carlo):
        x = np.random.uniform(start, stop, samples)
        x = np.sort(x)
        y = sawtooth(x)
        

        extension = i
        

        reg = KNeighborsRegressor(n_neighbors = i,
                                  weights='uniform')
        reg.fit(x.reshape(-1,1),y)
        
        x_uniform = np.linspace(start, stop , samples)
        
        y_reg = reg.predict(x_uniform.reshape(-1,1))
        
        y_uniform = sawtooth(x_uniform)        
        delta_sq = np.power((y_uniform[2:-2]-y_reg[2:-2]),2)
        diff = np.power(np.sum(delta_sq)/len(delta_sq),0.5)
        result = diff
        
        local_results.append(result)
        
        totals = []
        newdata = np.rot90([x_uniform,y_reg])
        for j in range(1,len(newdata)):
                  x_start = newdata[j-1][0]
                  y_start = newdata[j-1][1]
                  x_stop = newdata[j][0]
                  y_stop = newdata[j][1]
                  r2result = myr2multi(x_start, y_start,
                                          x_stop, y_stop,
                                          x, y,
                                          res)

                  totals.append(r2result) # added /np.mean(raw)
              
        totals = np.asarray(totals)
                
        Area_poly = np.power((np.sum(totals)/totals.size),0.5)

        areas_local.append(Area_poly)

    areas_global.append((areas_local))    
        
    global_results.append(np.mean(local_results))
    global_p5.append(np.percentile(local_results,5))
    global_p95.append(np.percentile(local_results,95))
    
g4 = global_results.copy()   
axs[1,1].scatter(extensions, global_results, marker = '.', label='mean',
               c='black', s=10)

axs[1,1].scatter(extensions, global_p5, marker = '_', label='5th/95th percentile',
               c='gray', s=10)

axs[1,1].scatter(extensions, global_p95, marker = '_',
               c='gray', s=10)
axs[1,1].set_xlabel('Neighbour count')

axs[1,1].grid()

axs[1,1].set_axisbelow(True)
axs[1,1].title.set_text('KNeighbors\nRegressor,\nweights=uniform')
axs[1,1].axhline(y=noremean, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\naverage')
axs[1,1].axhline(y=nore95, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
axs[1,1].axhline(y=nore5, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')
#axs[1,1].legend()


plt.tight_layout()
plt.savefig('resampling_tri.pdf')
plt.show()
#%%

backup = areas_global.copy()
backup = np.asarray(backup)

#%%
areas_global = []
areas_5 = []
areas_95 =[]
for row in backup:
    areas_global.append(np.nanmean(row))
    areas_5.append(np.nanpercentile(row,5))
    areas_95.append(np.nanpercentile(row,95))

fig, axs = plt.subplots(2,2, figsize = (5,6), sharey=True)
axs[0,0].scatter(np.linspace(0,10,100),areas_global[:100], s=5, c='black',
                 label='mean')
axs[0,1].scatter(np.linspace(0,10,100),areas_global[100:200], s=5, c='black')
axs[1,0].scatter(np.linspace(1,50,50),areas_global[200:250], s=5, c='black')
axs[1,1].scatter(np.linspace(1,50,50),areas_global[250:], s=5, c='black')

axs[0,0].scatter(np.linspace(0,10,100),areas_5[:100], s=2, c='grey',
                  label='5/95 percentile')
axs[0,1].scatter(np.linspace(0,10,100),areas_5[100:200], s=2, c='grey')
axs[1,0].scatter(np.linspace(1,50,50),areas_5[200:250], s=2, c='grey')
axs[1,1].scatter(np.linspace(1,50,50),areas_5[250:], s=2, c='grey')

axs[0,0].scatter(np.linspace(0,10,100),areas_95[:100], s=2, c='grey')
axs[0,1].scatter(np.linspace(0,10,100),areas_95[100:200], s=2, c='grey')
axs[1,0].scatter(np.linspace(1,50,50),areas_95[200:250], s=2, c='grey')
axs[1,1].scatter(np.linspace(1,50,50),areas_95[250:], s=2, c='grey')


axs[0,0].set_xlabel('Radius multiplier')
axs[0,0].set_ylabel('Root Mean Riemann Squared')
axs[0,0].grid()
axs[0,0].set_axisbelow(True)
axs[0,0].title.set_text('RadiusNeighbors\nRegressor,\nweights=distance')

axs[0,0].set_axisbelow(True)


axs[0,1].grid()
axs[0,1].set_xlabel('Radius multiplier')
axs[0,1].set_axisbelow(True)
axs[0,1].title.set_text('RadiusNeighbors\nRegressor,\nweights=uniform')

axs[1,0].set_ylabel('Root Mean Riemann Squared')
axs[1,0].set_axisbelow(True)
axs[1,0].title.set_text('KNeighbors\nRegressor,\nweights=distance')
axs[1,0].grid()
axs[1,0].set_xlabel('Neighbour count')




axs[1,1].grid()
axs[1,1].set_axisbelow(True)
axs[1,1].title.set_text('KNeighbors\nRegressor,\nweights=uniform')

axs[1,1].set_xlabel('Neighbour count')


axs[0,0].set_ylim(0,np.max(areas_global))
#axs[1,0].set_ylim(0,10)
axs[0,0].legend()
for k in [0,1]:
    for m in [0,1]:
        axs[k,m].axhline(y=noremean, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\nmean')
        axs[k,m].axhline(y=nore95, xmin=0, xmax=1, color='black', linewidth=1,
                 label='no resampling,\n5th/95th percentile', linestyle = '--')
        axs[k,m].axhline(y=nore5, xmin=0, xmax=1, color='black', linewidth=1, linestyle = '--')

axs[1,0].legend()
plt.tight_layout()
plt.savefig('resampling RMRS_tri.pdf')
#%%
from sklearn.metrics import r2_score
fig, axs = plt.subplots(2,2, figsize = (5,6), sharey=True)
axs[0,0].scatter(g1,areas_global[:100])
axs[0,1].scatter(g2,areas_global[100:200])
axs[1,0].scatter(g3,areas_global[200:250])
axs[1,1].scatter(g4,areas_global[250:])

print(r2_score(g1,areas_global[:100]))
print(r2_score(g2,areas_global[100:200]))
print(r2_score(g3,areas_global[200:250]))
print(r2_score(g4,areas_global[250:]))
#%%
np.random.seed(3)
start = 0
stop = 10
samples = 50
x = np.random.uniform(start, stop, samples)
x = np.sort(x)
y = sawtooth(x)


xtrue = np.linspace(start, stop , 1000)
ytrue = sawtooth(xtrue)
reg = KNeighborsRegressor(n_neighbors = 3,
                          weights='distance')
reg.fit(x.reshape(-1,1),y)

x_uniform = np.linspace(start, stop , samples)

y_reg = reg.predict(x_uniform.reshape(-1,1))

fig,ax = plt.subplots(1, figsize=(5,4))

#plt.scatter(x_uniform, y, s=10, c='blue', marker='v', label='raw, unindexed')
plt.plot(x_uniform, y, linewidth=0.5, c='red', alpha=0.5,  marker='v',
         label='raw\nequidistant', markersize=5)
plt.scatter(x_uniform, y_reg, s=30, marker='X', c='green', label='resampled,\nequidistant')

plt.scatter(x, y, s=5, marker='o', c='black', label='raw\nx-indexed')
plt.plot(xtrue,ytrue, c='black', linestyle='--', linewidth=1, alpha=0.5, label='true signal')

handles,labels = ax.get_legend_handles_labels()

handles = [handles[1], handles[3], handles[0], handles[2]]
labels = [labels[1], labels[3], labels[0], labels[2]]
#plt.legend(handles, labels, loc='lower left')
plt.legend()
plt.grid()
plt.xlabel('x\n[index]')
plt.ylabel('y=sin(x)')
plt.tight_layout()
plt.savefig('resampling_theory_tri.pdf')

#%%

local_results = []
for j in range(1000):
    start = 0
    stop = 10
    samples = 100
    x = np.random.uniform(start, stop, samples)
    x = np.sort(x)
    y = sawtooth(x)
    
    
    max_step = np.max(np.diff(x))


    
    x_uniform = np.linspace(start, stop, samples)
    
    y_uniform = sawtooth(x_uniform)
    
    
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