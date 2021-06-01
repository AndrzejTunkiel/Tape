#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:55:24 2021

@author: llothar
"""

from sens_tape import tape
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])

from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.ops import unary_union, polygonize
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from statistics_module import stats

def myr2multi(x_start, y_start, x_stop, y_stop, data_x, data_y, res):
  try:
      res = 10
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
    return 0
    print('oops')




def minfinder(h):
    #print('infunction')
    
    data = pd.read_csv('f9ad.csv')
    dfs = data.iloc[2000:10000]


    index = 'Measured Depth m'

    global_mins = {}
    res = 10
    full_log = []
    for target in list(data):
        
        try:
            print(target)
            dfs = data
            
            
            index_dr = np.diff(dfs[index])
    
            index_mean = np.mean(index_dr)
            index_std = np.std(index_dr)
            index_maxgap = np.max(index_dr)

            data_x = np.arange(np.min(dfs[index].to_numpy()),
                          np.max(dfs[index].to_numpy()),
                          index_maxgap*h)
            
            

            
            samples = np.arange(1,31,1)
            weightss = ['uniform', 'distance']
            local_mins = {}
            local_mins_n = {}
            for weights in weightss:
                areas = []
                
                for i in samples:
                    reg = RadiusNeighborsRegressor(radius=index_maxgap*i, weights=weights)
                    raw = dfs[target].interpolate().ffill().bfill().to_numpy()
                    reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
                    data_y = reg.predict(data_x.reshape(-1,1))

                    totals = []
                    newdata = np.rot90([data_x,data_y])
                    
                    for i in range(1,len(newdata)):
                        x_start = newdata[i-1][0]
                        y_start = newdata[i-1][1]
                        x_stop = newdata[i][0]
                        y_stop = newdata[i][1]
                        result = myr2multi(x_start, y_start,
                                                x_stop, y_stop,
                                                dfs[index].to_numpy(), raw,
                                                res)

                        totals.append(result)
                
                    totals = np.asarray(totals)
                    
                    if totals.size == 0:
                        Area_poly = float('inf') 
                    else:
                        Area_poly = np.power((np.sum(totals)/totals.size),0.5)
                        
                    areas.append(Area_poly)
                    
                local_mins[f'RNR {weights}']  = np.min(areas)
                
                local_mins_n[f'RNR {weights}']  = samples[np.argmin(areas)]
                # plt.plot(samples,areas, label=f'RNR, {weights}')
            
            ks = np.arange(1,31,1)
            for weights in weightss:
                areas = []
                
                for i in ks:
                    reg = KNeighborsRegressor(n_neighbors=i, weights=weights)
                    raw = dfs[target].interpolate().ffill().bfill().to_numpy()
                    reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
                    data_y = reg.predict(data_x.reshape(-1,1))
                    
                    totals = []
                    newdata = np.rot90([data_x,data_y])
                    
                    for i in range(1,len(newdata)):
                        x_start = newdata[i-1][0]
                        y_start = newdata[i-1][1]
                        x_stop = newdata[i][0]
                        y_stop = newdata[i][1]
                        result = myr2multi(x_start, y_start,
                                                x_stop, y_stop,
                                                dfs[index].to_numpy(), raw,
                                                res)

                        totals.append(result)
                
                    totals = np.asarray(totals)
                    
                    if totals.size == 0:
                        Area_poly = float('inf') 
                    else:
                        Area_poly = np.power((np.sum(totals)/totals.size),0.5)

                    areas.append(Area_poly)
                local_mins[f'KNN {weights}'] = np.min(areas)
                
                local_mins_n[f'KNN {weights}']  = ks[np.argmin(areas)]


            loopmin = (min(local_mins, key=local_mins.get))
            
            loopminval = local_mins_n[loopmin]
            
            if loopminval == 0 or loopminval == float('inf'):
                print('shit')
            else:
                full_log.append([loopmin, loopminval, target])

            
            
        except:
            print('Error, single row')
            pass
    

    return full_log
#%%
global_results_ave = np.nanmean(global_results, axis=0)

plt.plot(np.rot90(global_results_ave,3))
plt.ylim(1,2)
plt.show()
#%%
h=5

hs = [1,2,3,4,5,6,8,10,12,15,20,30]

for h in hs:
    res = minfinder(h)
    np.save(f'resh{h}.npy',res,  allow_pickle=True)

#%%

data = np.vstack(res)

    

plt.figure(figsize=(4,2.5))

df = pd.DataFrame(data=data, columns=["method", "n", "param"])
df['n'] = df['n'].astype(int)

methods = ['KNN uniform', 'KNN distance', 'RNR uniform', 'RNR distance']
ns = np.arange(1,31,1)
ms = np.arange(0,4,1)
summary = []
for m in ms:
    for n in ns:
        dft = df[df['method'] == methods[m]]
        dft = dft[dft['n'] == n]
        summary.append([m,n,len(dft)])
summary = np.asarray(summary)       

methods_plot = ['KNN\nuniform',
                'KNN\ndistance',
                'RNR\nuniform',
                'RNR\ndistance']

scaler = 1

plt.scatter(x=summary[:,1], y=summary[:,0], s=summary[:,2]*scaler,
            c='steelblue', linewidth=0.5, edgecolors='black')
plt.xticks(ns)
plt.yticks(ms, methods_plot)

sizes = np.arange(100,1001,300)
sizes = np.hstack((1,sizes))
for s in sizes:
    plt.scatter([],[],s=s*scaler, c='steelblue', label=f'{s}\n '
                ,linewidth=0.5, edgecolors='black')

plt.legend(title='winner count', bbox_to_anchor=(1.0, 1), loc='upper left')
plt.xlabel('Neighbor count / Radius multiplier')

plt.savefig('statsh5.pdf')
