# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:17:34 2021

@author: atunkiel
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics_module
import r2_module
#%%

data = pd.read_csv('f9ad.csv')

data = data.select_dtypes([np.number])
#%%

s,m, per = statistics_module.stats(data)

from prettytable import PrettyTable
x = PrettyTable()

x.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
x.add_rows(
    [
        ["Adelaide", 1295, 1158259, 600.5],
        ["Brisbane", 5905, 1857594, 1146.4],
        ["Darwin", 112, 120900, 1714.7],
        ["Hobart", 1357, 205556, 619.5],
        ["Sydney", 2058, 4336374, 1214.8],
        ["Melbourne", 1566, 3806092, 646.9],
        ["Perth", 5386, 1554769, 869.4],
    ]
)

x.align = "r"

print(x)

x.align["City name"] = "l"
x.align["Area"] = "c"
x.align["Population"] = "r"
x.align["Annual Rainfall"] = "c"
print(x)

#%%

for i in range(len(list(data))):
    x_label = np.asarray(per[i]['gap_sizes'], dtype=int)
    x = np.arange(0, len(x_label),1)
    y = per[i]['percentage_cells_occupied']
    plt.xticks(x, x_label, rotation=90)
    plt.bar(x,y)
    plt.title(list(data)[i])

    plt.show()
    
#%%

plt.bar(x,y)

#%%

series = data.to_numpy()[:,10].flatten()

series = np.asarray(series, dtype=float)

forin = series[~np.isnan(series)]

#%%
result = r2_module.pred3(forin,1)

#%%
cutoff = 20
m_plot = m.copy()
m_plot_inv = m.copy()

for i in range(m_plot.shape[1]):
    m_plot[:,i] = np.where(m_plot[:,i] <= cutoff, i, np.nan)

for i in range(m_plot.shape[1]):
    m_plot_inv[:,i] = np.where(m_plot_inv[:,i] > cutoff, i, np.nan)

m_plot = m_plot.T
m_plot_inv = m_plot_inv.T

x = np.arange(0,m_plot.shape[1],1)

plt.figure(figsize=(16,9))

for i in range(m_plot.shape[0]):
    plt.scatter(x, m_plot[i], s=17, c='green', linewidth=1, marker=3)

    plt.scatter(x, m_plot_inv[i], s=17, c='red', linewidth=1, marker=3)
plt.savefig('test.png', dpi=300)
plt.show()

#%%

import overlap_plot_module


#%% commented out for speed
# for i in range(100):
#     plt.figure(figsize=(19,9))
#     overlap_plot_module.overlap_plot(m, i)
#     plt.title(f'Filling gaps length {i}')
    
#     plt.savefig(f'{str(i).zfill(2)}.png', dpi=300)
#     plt.show()

#%%

# filling in only cells that match prescribed cutoff. 

cutoff = 20
m_kill = m.copy()
m_kill = np.where(m_kill <= cutoff, 1, np.nan)

data_k = pd.DataFrame(data)
data_k = data_k.interpolate()
data_k = data_k.to_numpy()
data_k = np.multiply(data_k,m_kill)

#%%

x = np.arange(0, len(m),1)
y = m[:,7]
plt.scatter(x,y, s=1)
plt.ylim(0,30)
plt.title('Plotting gap distribution row-based')
#plt.xlim(2500,7500)
#%%

series = data.to_numpy()[:,7].flatten()

series = np.asarray(series, dtype=float)

forin = series[~np.isnan(series)]
plt.plot(forin)

result = r2_module.pred3(forin,1)
plt.title(f'Predicted R2 STUCK_RT for [:,7], all: {result}')
plt.show()
print (f'Predicted R2 STUCK_RT for [:,7], all: {result}')


series = data.to_numpy()[2500:10000,7].flatten()

series = np.asarray(series, dtype=float)

forin = series[~np.isnan(series)]
plt.plot(forin)

result = r2_module.pred3(forin,1)
plt.title(f'Predicted R2 STUCK_RT for [2.5k:10k,7], all: {result}')
plt.show()
print (f'Predicted R2 STUCK_RT for [2.5k:10k,7], all: {result}')


series = data.to_numpy()[12500:17500,7].flatten()

series = np.asarray(series, dtype=float)

forin = series[~np.isnan(series)]
plt.plot(forin)

result = r2_module.pred3(forin,1)
plt.title((f'Predicted R2 STUCK_RT for [12.5k:17.5:,7], all: {result}'))
plt.show()
print (f'Predicted R2 STUCK_RT for [12.5k:17.5:,7], all: {result}')
#%%

plt.plot(data['Measured Depth m'].diff())
plt.ylim(0, 0.2)

#%%

series = data.to_numpy()[:,6].flatten()

series = np.asarray(series, dtype=float)

forin = series[~np.isnan(series)]
plt.plot(forin)
plt.show()
result = r2_module.pred3(forin,1)

print (f'Predicted R2 average rotary speed for [:,7], all: {result}')

#%%

series = data.to_numpy()[:,8].flatten()

series = np.asarray(series, dtype=float)

forin = series[~np.isnan(series)]
plt.plot(forin)
plt.show()
result = r2_module.pred3(forin,1)

print (f'Predicted R2 corr surf WOB for [:,7], all: {result}')

#%%
x = np.arange(0, len(m),1)
y = m[:,8]
plt.scatter(x,y, s=1)
plt.ylim(0,10)
#plt.xlim(2500,7500)