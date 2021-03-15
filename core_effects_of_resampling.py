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
plt.figure(figsize=(5,5))
index_dr = np.diff(dfs[index])

index_mean = np.mean(index_dr)
index_std = np.std(index_dr)
index_maxgap = np.max(index_dr)
h = 5
x = np.arange(np.min(dfs[index].to_numpy()),
              np.max(dfs[index].to_numpy()),
              index_maxgap*h)


from sklearn.neighbors import RadiusNeighborsRegressor
reg = RadiusNeighborsRegressor(radius=index_maxgap*1, weights='distance')

# raw = dfs['MWD Continuous Inclination dega'].interpolate().ffill().bfill().to_numpy()
# reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
# y = reg.predict(x.reshape(-1,1))
# plt.xlim(650,700)
# plt.plot(x,y)
# plt.plot(dfs[index].to_numpy(),raw)

# plt.show()



raw = dfs['Rate of Penetration m/h'].interpolate().ffill().bfill().to_numpy()
reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
y = reg.predict(x.reshape(-1,1))

plt.plot(x,y)


raw_x = dfs[index].to_numpy()
plt.plot(raw_x,raw)
plt.grid()
plt.tight_layout()
plt.xlim(660,690)
plt.show()