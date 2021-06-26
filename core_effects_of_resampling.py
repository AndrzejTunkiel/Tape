#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:43:26 2021

@author: llothar
"""

from sens_tape import tape
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])
data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
                        'MWD Continuous Azimuth dega']

dfs = data.iloc[2000:10000]
index = 'Measured Depth m'
target =  'MWD Continuous Inclination dega',

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8,3))
index_dr = np.diff(dfs[index])

index_mean = np.mean(index_dr)
index_std = np.std(index_dr)
index_maxgap = np.max(index_dr)
h = 5
x = np.arange(np.min(dfs[index].to_numpy()),
              np.max(dfs[index].to_numpy()),
              index_maxgap*h)


from sklearn.neighbors import RadiusNeighborsRegressor


# raw = dfs['MWD Continuous Inclination dega'].interpolate().ffill().bfill().to_numpy()
# reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
# y = reg.predict(x.reshape(-1,1))
# plt.xlim(650,700)
# plt.plot(x,y)
# plt.plot(dfs[index].to_numpy(),raw)

# plt.show()

reg = RadiusNeighborsRegressor(radius=index_maxgap*1, weights='uniform')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[0].plot(x,y, c='blue', linewidth=1, label='r = 1 max step', linestyle="-")

reg = RadiusNeighborsRegressor(radius=index_maxgap*20, weights='uniform')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[0].plot(x,y, c='black', linewidth=1, label='r = 20 max step', linestyle="-")

reg = RadiusNeighborsRegressor(radius=index_maxgap*100, weights='uniform')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[0].plot(x,y, c='black', linewidth=1, label='r = 100 max step', linestyle="--")


raw_x = dfs[index].to_numpy()
axs[0].plot(raw_x,raw, c='red', linestyle=':', label='raw data')
axs[0].grid()
plt.tight_layout()
axs[0].set_xlim(650,690)
plt.ylim(0,60)
axs[0].legend()
axs[0].set_title('Uniform weight')
axs[0].set_ylabel('Rate of Penetration [m/h]')
axs[0].set_xlabel('Measured Depth [m]')


reg = RadiusNeighborsRegressor(radius=index_maxgap*1, weights='distance')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[1].plot(x,y, c='blue', linewidth=1, label='r = 1 max step', linestyle="-")

reg = RadiusNeighborsRegressor(radius=index_maxgap*20, weights='distance')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[1].plot(x,y, c='black', linewidth=1, label='r = 20 max step', linestyle="-")

reg = RadiusNeighborsRegressor(radius=index_maxgap*100, weights='distance')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[1].plot(x,y, c='black', linewidth=1, label='r = 100 max step', linestyle="--")


raw_x = dfs[index].to_numpy()
axs[1].plot(raw_x,raw, c='red', linestyle=':', label='raw data')
axs[1].grid()
plt.tight_layout()
axs[1].set_xlim(650,690)
plt.ylim(0,60)

axs[1].legend()
axs[1].set_title('Distance weight')
axs[1].set_xlabel('Measured Depth [m]')


plt.savefig('resampling_radius_rnr.pdf')
plt.show()
#%%
data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
                        'MWD Continuous Azimuth dega']

dfs = data.iloc[2000:10000]
index = 'Measured Depth m'
target =  'MWD Continuous Inclination dega',

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8,3))
index_dr = np.diff(dfs[index])

index_mean = np.mean(index_dr)
index_std = np.std(index_dr)
index_maxgap = np.max(index_dr)
h = 5
x = np.arange(np.min(dfs[index].to_numpy()),
              np.max(dfs[index].to_numpy()),
              index_maxgap*h)


from sklearn.neighbors import KNeighborsRegressor


# raw = dfs['MWD Continuous Inclination dega'].interpolate().ffill().bfill().to_numpy()
# reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
# y = reg.predict(x.reshape(-1,1))
# plt.xlim(650,700)
# plt.plot(x,y)
# plt.plot(dfs[index].to_numpy(),raw)

# plt.show()

reg = KNeighborsRegressor(n_neighbors=1, weights='uniform')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[0].plot(x,y, c='blue', linewidth=1, label='K = 1', linestyle="-")

reg = KNeighborsRegressor(n_neighbors=20, weights='uniform')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[0].plot(x,y, c='black', linewidth=1, label='K = 20', linestyle="-")

reg = KNeighborsRegressor(n_neighbors=100, weights='uniform')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[0].plot(x,y, c='black', linewidth=1, label='K = 100', linestyle="--")


raw_x = dfs[index].to_numpy()
axs[0].plot(raw_x,raw, c='red', linestyle=':', label='raw data')
axs[0].grid()
plt.tight_layout()
axs[0].set_xlim(650,690)
plt.ylim(0,60)
axs[0].legend()
axs[0].set_title('Uniform weight')
axs[0].set_ylabel('Rate of Penetration [m/h]')
axs[0].set_xlabel('Measured Depth [m]')


reg = KNeighborsRegressor(n_neighbors=1, weights='distance')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[1].plot(x,y, c='blue', linewidth=1, label='K = 1', linestyle="-")

reg = KNeighborsRegressor(n_neighbors=20, weights='distance')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[1].plot(x,y, c='black', linewidth=1, label='K = 20', linestyle="-")

reg = KNeighborsRegressor(n_neighbors=100, weights='distance')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[1].plot(x,y, c='black', linewidth=1, label='K = 100', linestyle="--")


raw_x = dfs[index].to_numpy()
axs[1].plot(raw_x,raw, c='red', linestyle=':', label='raw data')
axs[1].grid()
plt.tight_layout()
axs[1].set_xlim(650,690)
plt.ylim(0,60)

axs[1].legend()
axs[1].set_title('Distance weight')
axs[1].set_xlabel('Measured Depth [m]')


plt.savefig('resampling_radius_knn.pdf')
plt.show()
#%%
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.ops import unary_union, polygonize
data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0',
         'Pass Name unitless',
'MWD Magnetic Toolface dega',
'nameWellbore',
'name',
'IMP/ARC Attenuation Conductivity 40-in. at 2 MHz mS/m',
'ARC Annular Pressure kPa',
'MWD Collar RPM rpm',
'IMP/ARC Non-BHcorr Phase-Shift Resistivity 28-in. at 2 MHz ohm.m',
'IMP/ARC Phase-Shift Conductivity 40-in. at 2 MHz mS/m',
'Annular Temperature degC',
'IMP/ARC Non-BHcorr Phase-Shift Resistivity 40-in. at 2 MHz ohm.m',
'ARC Gamma Ray (BH corrected) gAPI',
'IMP/ARC Non-BHcorr Attenuation Resistivity 40-in. at 2 MHz ohm.m',
'MWD Stick-Slip PKtoPK RPM rpm',
'IMP/ARC Non-BHcorr Attenuation Resistivity 28-in. at 2 MHz ohm.m',
'IMP/ARC Phase-Shift Conductivity 28-in. at 2 MHz mS/m'  
    ]

data = data.drop(drops, axis=1)
dfs = data.iloc[2000:10000]
index = 'Measured Depth m'
target =  'Rate of Penetration m/h' #'MWD Continuous Inclination dega' 

index_dr = np.diff(dfs[index])

index_mean = np.mean(index_dr)
index_std = np.std(index_dr)
index_maxgap = np.max(index_dr)
h = 5
data_x = np.arange(np.min(dfs[index].to_numpy()),
              np.max(dfs[index].to_numpy()),
              index_maxgap*h)

#%%
for target in list(data):
    # try:
        areas = []
        samples = np.arange(1,200,10)
        weightss = ['uniform', 'distance']
        
        for weights in weightss:
            areas = []
            for i in samples:
                reg = RadiusNeighborsRegressor(radius=index_maxgap*i, weights=weights)
                raw = dfs[target].interpolate().ffill().bfill().to_numpy()
                reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
                data_y = reg.predict(data_x.reshape(-1,1))
                
                x_y_curve1 = np.rot90([data_x,data_y])
                x_y_curve2 = np.rot90([dfs[index].to_numpy(), raw])
                
                
                polygon_points = [] #creates a empty list where we will append the points to create the polygon
                
                for xyvalue in x_y_curve1:
                    polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1
                
                for xyvalue in x_y_curve2[::-1]:
                    polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)
                
                for xyvalue in x_y_curve1[0:1]:
                    polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon
                
                polygon = Polygon(polygon_points)
                area = polygon.area
                
                x,y = polygon.exterior.xy
                    # original data
                ls = LineString(np.c_[x, y])
                    # closed, non-simple
                lr = LineString(ls.coords[:] + ls.coords[0:1])
                lr.is_simple  # False
                mls = unary_union(lr)
                mls.geom_type  # MultiLineString'
                
                Area_cal =[]
                
                for polygon in polygonize(mls):
                    Area_cal.append(polygon.area)
                    Area_poly = (np.asarray(Area_cal).sum())
                areas.append(Area_poly)
                
            plt.plot(samples,areas, label=f'RNR, {weights}')
        
        from sklearn.neighbors import KNeighborsRegressor
        
        ks = np.arange(1,200,10)
        for weights in weightss:
            areas = []
            for i in ks:
                reg = KNeighborsRegressor(n_neighbors=i, weights=weights)
                raw = dfs[target].interpolate().ffill().bfill().to_numpy()
                reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
                data_y = reg.predict(data_x.reshape(-1,1))
                
                x_y_curve1 = np.rot90([data_x,data_y])
                x_y_curve2 = np.rot90([dfs[index].to_numpy(), raw])
                
                
                polygon_points = [] #creates a empty list where we will append the points to create the polygon
                
                for xyvalue in x_y_curve1:
                    polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1
                
                for xyvalue in x_y_curve2[::-1]:
                    polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)
                
                for xyvalue in x_y_curve1[0:1]:
                    polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon
                
                polygon = Polygon(polygon_points)
                area = polygon.area
    
                x,y = polygon.exterior.xy
                    # original data
                ls = LineString(np.c_[x, y])
                    # closed, non-simple
                lr = LineString(ls.coords[:] + ls.coords[0:1])
                lr.is_simple  # False
                mls = unary_union(lr)
                mls.geom_type  # MultiLineString'
                
                Area_cal =[]
                
                for polygon in polygonize(mls):
                    Area_cal.append(polygon.area)
                    Area_poly = (np.asarray(Area_cal).sum())
                areas.append(Area_poly)
    
               
            plt.plot(ks,areas, label=f'KNN, {weights}')
        
        plt.legend()
        plt.title(target)
        plt.grid()
        plt.show()
    # except:
    #     print(f'{target} failed for some reason')


#%%

# no poly version
def myr2(x,y,data_x, data_y):
  try:
    x1 = np.max(data_x[data_x < x])
    x2 = np.min(data_x[data_x > x])
    
    loc1 = np.where(data_x == x1)
    loc2 = np.where(data_x == x2)

    y1 = data_y[loc1][-1]
    y2 = data_y[loc2][0]



    m = (y1-y2)/(x1-x2)
    b = (x1*y2 - x2*y1)/(x1-x2)

    
    y_inter = m * x + b

    return np.power(y-y_inter, 2)
  except:
    return 0

n = 0
for target in list(data):
    # try:
        plt.figure(figsize=(5,5))
        areas = []
        samples = np.arange(1,31,1)
        weightss = ['uniform', 'distance']
        
        for weights in weightss:
            areas = []
            for i in samples:
                reg = RadiusNeighborsRegressor(radius=index_maxgap*i, weights=weights)
                raw = dfs[target].interpolate().ffill().bfill().to_numpy()
                reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
                data_y = reg.predict(data_x.reshape(-1,1))
                
                totals = []
                for row in np.rot90([data_x,data_y]):
                  x = row[0]
                  y = row[1]
                  totals.append(myr2(x,y,dfs[index].to_numpy(), raw))
              
              
                Area_poly = np.power((np.sum(totals)/len(totals)),0.5)
                areas.append(Area_poly)
                
            plt.plot(samples,areas, label=f'RNR, {weights}')
        
        from sklearn.neighbors import KNeighborsRegressor
        
        ks = np.arange(1,31,1)
        for weights in weightss:
            areas = []
            for i in ks:
                reg = KNeighborsRegressor(n_neighbors=i, weights=weights)
                raw = dfs[target].interpolate().ffill().bfill().to_numpy()
                reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
                data_y = reg.predict(data_x.reshape(-1,1))
                
                
                totals = []
                for row in np.rot90([data_x,data_y]):
                  x = row[0]
                  y = row[1]
                  totals.append(myr2(x,y,dfs[index].to_numpy(), raw))
              
              
                Area_poly = np.power((np.sum(totals)/len(totals)),0.5)
                areas.append(Area_poly)
               
            plt.plot(ks,areas, label=f'KNN, {weights}')
        
        plt.xlabel('K \ radius multiplier')
        plt.ylabel('Error [RMS]')
        plt.legend()
        plt.title(target)
        plt.grid()
        plt.yscale('log')
        plt.savefig(f'{n}.pdf')
        n += 1
        plt.show()
    # except:
    #     print(f'{target} failed for some reason')

#%%
# no poly version, Riemann squared
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
    return 0
    print('oops')

n = 0
res = 10

global_results = []
colors = ['red', 'green', 'blue', 'black']
linestyles = ['-','--', '-.', ':']


for target in list(data):
    # try:
        c = 0
        local_result = [[],[],[],[]]
        plt.figure(figsize=(4,4))
        areas = []
        samples = np.arange(1,31,1)
        weightss = ['uniform', 'distance']

        plotno = 0
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

                  totals.append(result) # added /np.mean(raw)
              
                totals = np.asarray(totals)
                
                Area_poly = np.power((np.sum(totals)/totals.size),0.5)

                areas.append(Area_poly)
                
            plt.plot(samples,areas, label=f'RNR\n{weights}',
                     c = colors[c], linestyle = linestyles[c],linewidth=1.5 )
            c += 1
            local_result[plotno] = areas
            plotno += 1
        
        from sklearn.neighbors import KNeighborsRegressor
        
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
                  totals.append(myr2multi(x_start, y_start,
                                          x_stop, y_stop,
                                          dfs[index].to_numpy(), raw,
                                          res))
              
                totals = np.asarray(totals)
                Area_poly = np.power((np.sum(totals)/totals.size),0.5)
                areas.append(Area_poly)
               
            plt.plot(ks,areas, label=f'KNN\n{weights}',
                     c = colors[c], linestyle = linestyles[c],linewidth=1.5 )
            c += 1
            local_result[plotno] = areas
            plotno += 1
            
        local_result = local_result/np.min(local_result)
        global_results.append(local_result)
        
        plt.xlabel('neigbor count / radius multiplier')
        plt.ylabel('error [RMRS]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(target)
        plt.grid()
        plt.yscale('log')
        plt.savefig(f'multi_{n}.pdf')
        n += 1
        plt.show()
        
        plt.plot(local_result[0])
        plt.plot(local_result[1])
        plt.plot(local_result[2])
        plt.plot(local_result[3])
        plt.show()
    # except:
    #     print(f'{target} failed for some reason')

np.save('global_results.npy', global_results)
#%%

global_results = np.load('global_results.npy') 
plt.figure(figsize=(4,4))
global_results = np.asarray(global_results)

methods_plot = [
    'RNR\nuniform',
    'RNR\ndistance',
    'KNN\nuniform',
    'KNN\ndistance'
    ]


colors = ['red', 'green', 'blue', 'black']
linestyles = ['-','--', '-.', ':']
for i in range(4):
    plt.plot(np.nanmean(global_results[:,i,:], axis=0), label=methods_plot[i],
             c=colors[i], linewidth=1.5, linestyle = linestyles[i])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.yscale('log')
ymax = 3.1
plt.yticks(np.arange(1,ymax,0.2), np.arange(100,ymax*100,20).astype(int))
plt.grid()
plt.xlabel('neighbor count / radius multiplier')
plt.ylabel('RMRS error\ncompared to best selection [%]')
plt.ylim(1,3)
plt.xticks(np.arange(-1,31,5), np.arange(0,32,5))
plt.savefig('algocompare.pdf')

#%%

plt.figure(figsize=(5,4))
plt.rc('axes', axisbelow=True)
plt.grid(linewidth=1, color='gray')
x = np.arange(1,101,1)
y = 1/x

import matplotlib

cmap = matplotlib.cm.get_cmap('hsv')

n = 15


for i in range(n+1):
    for j in range(i):
        
        if i == n:
            plt.bar(x=i,
                    height=y[j]/np.sum(y[:i]),
                    bottom=np.sum(y[:j])/np.sum(y[:i]),
                    color = cmap(j/(n+1)),
                    label=f'd = {j+1}',
                    edgecolor='black')
        else:
            plt.bar(x=i,
                    height=y[j]/np.sum(y[:i]),
                    bottom=np.sum(y[:j])/np.sum(y[:i]),
                    color = cmap(j/(n+2)),
                    edgecolor='black')
            
            
plt.xlim(0,n+1)
plt.xticks(np.arange(1,n+1,1), rotation=90)
plt.yticks(np.linspace(0,1,11), np.linspace(0,100,11).astype(int))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Radius Neighbour Regressor radius limit')
plt.ylabel('Datapoint weights, percent')
plt.tight_layout()

#plt.grid()
plt.savefig('Cumulative weights.pdf')

#%%

plt.figure(figsize=(5,4))
plt.rc('axes', axisbelow=True)
plt.grid(linewidth=1, color='gray')
x = np.ones(100)
y = x

import matplotlib

cmap = matplotlib.cm.get_cmap('hsv')

n = 15


for i in range(n+1):
    for j in range(i):
        
        if i == n:
            plt.bar(x=i,
                    height=y[j]/np.sum(y[:i]),
                    bottom=np.sum(y[:j])/np.sum(y[:i]),
                    color = cmap(j/(n+1)),
                    label=f'd = {j+1}',
                    edgecolor='black')
        else:
            plt.bar(x=i,
                    height=y[j]/np.sum(y[:i]),
                    bottom=np.sum(y[:j])/np.sum(y[:i]),
                    color = cmap(j/(n+2)),
                    edgecolor='black')
            
            
plt.xlim(0,n+1)
plt.xticks(np.arange(1,n+1,1), rotation=90)
plt.yticks(np.linspace(0,1,11), np.linspace(0,100,11).astype(int))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Radius Neighbour Regressor radius limit')
plt.ylabel('Datapoint weights, percent')
plt.tight_layout()


plt.savefig('Cumulative weights2.pdf')

#%%
import glob
filelist = (glob.glob("full_log*.npy"))

data_array = []
for file in filelist:
    data = np.load(file, allow_pickle=True)
    if len(data) > 0:
        data_array.append(data)
    
data = np.vstack(data_array)

    

plt.figure(figsize=(4,2.5))

df = pd.DataFrame(data=data, columns=["method", "n", "param"])
df['n'] = df['n'].astype(int)

methods = ['KNN uniform', 'KNN distance', 'RNR uniform', 'RNR distance']
ns = np.arange(1,11,1)
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

scaler = 1.5

plt.scatter(x=summary[:,1], y=summary[:,0], s=summary[:,2]*scaler, c='steelblue')
plt.xticks(ns)
plt.yticks(ms, methods_plot)

sizes = np.arange(100,401,100)
sizes = np.hstack((1,sizes))
for s in sizes:
    plt.scatter([],[],s=s*scaler, c='steelblue', label=f'{s}\n ')

plt.legend(title='winner count', bbox_to_anchor=(1.0, 1), loc='upper left')
plt.xlabel('Neighbor count / Radius multiplier')


#%%
import glob
filelist = (glob.glob("simann*.npy"))

data_array = []
for file in filelist:
    data = np.load(file, allow_pickle=True)
    if len(data) > 0:
        data_array.append(data)
    
data = np.vstack(data_array)

    

plt.figure(figsize=(4,2.5))

df = pd.DataFrame(data=data, columns=["method", "n", "param"])
df['n'] = df['n'].astype(int)

methods = ['KNN uniform', 'KNN distance', 'RNR uniform', 'RNR distance']
ns = np.arange(1,11,1)
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
plt.savefig('riemann.pdf')


#%%

hs = [1,2,3,4,5,6,8,10,12,15,20,30]

for h in hs:
    
    res = np.load(f'resh{h}.npy',  allow_pickle=True)
    
    data = np.vstack(res)

    

    plt.figure(figsize=(6,2.5))
    
    df = pd.DataFrame(data=data, columns=["method", "n", "param"])
    df['n'] = df['n'].astype(int)
    
    methods = ['KNN distance', 'KNN uniform', 'RNR distance', 'RNR uniform']
    ns = np.arange(1,31,1)
    ms = np.arange(0,4,1)
    summary = []
    for m in ms:
        for n in ns:
            dft = df[df['method'] == methods[m]]
            dft = dft[dft['n'] == n]
            summary.append([m,n,len(dft)])
    summary = np.asarray(summary)       
    
    
    xsum = []
    
    for i in range(4):
        xsum.append(np.sum(summary[summary[:,0]==i][:,2]))
    
    
    methods_plot = [f'KNN\ndistance ({xsum[0]})',
                    f'KNN\nuniform ({xsum[1]})',
                    f'RNR\ndistance ({xsum[2]})',
                    f'RNR\nuniform ({xsum[3]})']
    
    scaler = 4
    
    plt.scatter(x=summary[:,1], y=summary[:,0], s=summary[:,2]*scaler,
                c='steelblue', linewidth=0.5, edgecolors='black')
    plt.xticks(ns)
    plt.yticks(ms, methods_plot)
    plt.ylim(-0.5,3.5)
    sizes = np.arange(10,51,10)
    sizes = np.hstack((1,sizes))
    for s in sizes:
        plt.scatter([],[],s=s*scaler, c='steelblue', label=f'{s}'
                    ,linewidth=0.5, edgecolors='black')
    
    plt.legend(title='winner count', bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.xlabel('Neighbor count / Radius multiplier')
    #plt.title(f'h-step {h}')
    plt.xticks(ns[::1],ns[::1],rotation=90)
    plt.xlim(0,20)
    plt.grid()
    plt.savefig(f'h-step {h}.pdf')
    plt.show()
    
#%%

res = np.load(f'resh{5}.npy',  allow_pickle=True)
data = np.vstack(res)



plt.figure(figsize=(6,2.5))

df = pd.DataFrame(data=data, columns=["method", "n", "param"])
df['n'] = df['n'].astype(int)

methods = ['KNN distance', 'KNN uniform', 'RNR distance', 'RNR uniform']
ns = np.arange(1,31,1)
ms = np.arange(0,4,1)
summary = []
for m in ms:
    for n in ns:
        dft = df[df['method'] == methods[m]]
        dft = dft[dft['n'] == n]
        summary.append([m,n,len(dft)])
summary = np.asarray(summary)       
    
a = summary
print(a[a[:, 2].argsort()])