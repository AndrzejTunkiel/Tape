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

data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
                        'MWD Continuous Azimuth dega']

dfs = data.iloc[2000:10000]
index = 'Measured Depth m'
target =  'MWD Continuous Inclination dega',
#%%
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

axs[0].plot(x,y, c='blue', linewidth=1, label='radius = 1 min_g', linestyle="-")

reg = RadiusNeighborsRegressor(radius=index_maxgap*10, weights='uniform')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[0].plot(x,y, c='black', linewidth=1, label='radius = 10 min_g', linestyle="-")

reg = RadiusNeighborsRegressor(radius=index_maxgap*100, weights='uniform')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[0].plot(x,y, c='black', linewidth=1, label='radius = 100 min_g', linestyle="--")


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

axs[1].plot(x,y, c='blue', linewidth=1, label='radius = 1 min_g', linestyle="-")

reg = RadiusNeighborsRegressor(radius=index_maxgap*10, weights='distance')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[1].plot(x,y, c='black', linewidth=1, label='radius = 10 min_g', linestyle="-")

reg = RadiusNeighborsRegressor(radius=index_maxgap*100, weights='distance')
raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

axs[1].plot(x,y, c='black', linewidth=1, label='radius = 100 min_g', linestyle="--")


raw_x = dfs[index].to_numpy()
axs[1].plot(raw_x,raw, c='red', linestyle=':', label='raw data')
axs[1].grid()
plt.tight_layout()
axs[1].set_xlim(650,690)
plt.ylim(0,60)

axs[1].legend()
axs[1].set_title('Distance weight')
axs[1].set_xlabel('Measured Depth [m]')


plt.savefig('resampling_radius.pdf')
plt.show()