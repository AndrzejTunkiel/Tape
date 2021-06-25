#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:44:31 2021

@author: atunkiel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:43:26 2021

@author: llothar
"""
start =660
stop = 720

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
axs[0].set_xlim(start,stop)
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
axs[1].set_xlim(start,stop)
plt.ylim(0,60)

axs[1].legend()
axs[1].set_title('Distance weight')
axs[1].set_xlabel('Measured Depth [m]')


plt.savefig('resampling_radius_rnr_phd.pdf')
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
axs[0].set_xlim(start,stop)
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
axs[1].set_xlim(start,stop)
plt.ylim(0,60)

axs[1].legend()
axs[1].set_title('Distance weight')
axs[1].set_xlabel('Measured Depth [m]')


plt.savefig('resampling_radius_knn_phd.pdf')
plt.show()

#%%
# no poly version, Riemann squared

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
colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']
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
        plt.savefig(f'multi_{n}_phd.pdf')
        n += 1
        plt.show()
        
        plt.plot(local_result[0])
        plt.plot(local_result[1])
        plt.plot(local_result[2])
        plt.plot(local_result[3])
        plt.show()
    # except:
    #     print(f'{target} failed for some reason')


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

colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']
linestyles = ['-','--', '-.', ':']
for i in range(4):
    plt.plot(np.nanmean(global_results[:,i,:], axis=0), label=methods_plot[i],
             c=colors[i], linewidth=1.5, linestyle = linestyles[i])

plt.legend(bbox_to_anchor=(1.05, 0.75), loc='upper left')

plt.yscale('log')
ymax = 3.1
plt.yticks(np.arange(1,ymax,0.2), np.arange(100,ymax*100,20).astype(int))
plt.grid()
plt.xlabel('neighbor count / radius multiplier')
plt.ylabel('average RMRS error\ncompared to best selection [%]')
plt.ylim(1,3)
plt.xticks(np.arange(-1,31,5), np.arange(0,32,5))
plt.savefig('algocompare_phd.pdf')