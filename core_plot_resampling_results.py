#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:54:29 2021

@author: llothar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('resampling_study.csv')

resampling = ['radius', 'knn']
weight = ['uniform', 'distance']

fig, axs = plt.subplots(1, 4, sharey=True, figsize=(9,3))
i=0
for r in resampling:
    for w in weight:
        dft = df.loc[(df['resampling'] == r) & (df['weight'] == w)]
        
        x = dft['K'].astype(int)
        y = dft['MAE']
        
        axs[i].scatter(x,y, c='black', s=10, marker='x')
        
        dft = df.loc[(df['resampling'] == 'no')]
        y_no = np.mean(dft['MAE'])
        axs[i].hlines(y_no, 0, 100, label='no resampling', color='black',
                   linewidth=1, linestyle="--")
        # y_no = np.min(dft['MAE'])
        # axs[i].hlines(y_no, 0, 100, label='no resampling min/max', color='black',
        #            linewidth=0.7, linestyle='--')
        # y_no = np.max(dft['MAE'])
        # axs[i].hlines(y_no, 0, 100, color='black',
        #            linewidth=0.7, linestyle='--')
        #axs[i].set_ylim(0.75,1.4)
        #axs[i].legend()
        axs[i].grid()
        axs[i].set_xlabel(f'{r} n {w}')
        
        i += 1
        
axs[0].set_ylabel('Mean Absolute Error\n[degrees]')


axs[0].set_title(f'Fixed Radius\nuniform weight')
axs[1].set_title(f'Fixed Radius\ndistance weight')
axs[2].set_title(f'K-Nearest Neighbors\nuniform weight')
axs[3].set_title(f'K-Nearest Neighbors\ndistance weight')


axs[0].set_xlabel(f'radius multiplier n\n (r = n * max_step)')
axs[1].set_xlabel(f'radius multiplier n\n (r = n * max_step)')
axs[2].set_xlabel(f'K, neighbor count')
axs[3].set_xlabel(f'K, neighbor count')


for i in range(4):
    axs[i].set_xticks(np.linspace(0,100,6).astype(int))

axs[3].legend()
plt.tight_layout()
plt.savefig('resampling mae.pdf')
plt.show()

fig, axs = plt.subplots(1, 4, sharey=True, figsize=(9,3))
i=0
for r in resampling:
    for w in weight:
        dft = df.loc[(df['resampling'] == r) & (df['weight'] == w)]
        
        x = dft['K'].astype(int)
        y = dft['usable']
        
        
        axs[i].scatter(x,y, c='black', s=10, marker='x')
        
        dft = df.loc[(df['resampling'] == 'no')]
        y_no = np.mean(dft['usable'])
        axs[i].hlines(y_no, 0, 100, label='no resampling', color='black',
                   linewidth=1, linestyle="--")
        # y_no = np.min(dft['usable'])
        # axs[i].hlines(y_no, 0, 100, label='no resampling min/max', color='black',
        #            linewidth=0.7, linestyle='--')
        # y_no = np.max(dft['usable'])
        # axs[i].hlines(y_no, 0, 100, color='black',
        #            linewidth=0.7, linestyle='--')
        #axs[i].set_ylim(0.75,1.4)
        #axs[i].legend()
        axs[i].grid()
        axs[i].set_xlabel(f'{r} n {w}')
        
        i += 1
        
axs[0].set_ylabel('Prediction area under\n 1.2 deg. error')

axs[0].set_title(f'Fixed Radius\nuniform weight')
axs[1].set_title(f'Fixed Radius\ndistance weight')
axs[2].set_title(f'K-Nearest Neighbors\nuniform weight')
axs[3].set_title(f'K-Nearest Neighbors\ndistance weight')


axs[0].set_xlabel(f'radius multiplier n\n (r = n * max_step)')
axs[1].set_xlabel(f'radius multiplier n\n (r = n * max_step)')
axs[2].set_xlabel(f'K, neighbor count')
axs[3].set_xlabel(f'K, neighbor count')

for i in range(4):
    axs[i].set_xticks(np.linspace(0,100,6).astype(int))

axs[3].legend()
plt.tight_layout()
plt.savefig('resampling useful.pdf')
plt.show()

#%%
# data = pd.read_csv('f9ad.csv')
# np.max(data.iloc[2000:10000]['Measured Depth m'].ffill().diff())
plt.figure(figsize=(4,3))
import seaborn as sns
df = pd.read_csv('hstep_extension_study.csv')

results = []
maxstep = int(np.max(df['hstep_extension']))
for i in range(0,maxstep):
    results.append(np.average(df[df['hstep_extension'] == i+1]['MAE']))
    
plt.grid
plt.scatter(x = np.arange(1,maxstep+1,1), y=results, marker="x", c='black', s=20)
plt.grid()
n = 10
xticklabels = np.linspace(0.153, 0.153*maxstep,n)
xticklabels = np.round(xticklabels, 2)

plt.xticks(np.linspace(1,maxstep+1,n), xticklabels, rotation=90)
plt.xlabel('Resampling step length [m]')
plt.ylabel('Mean Absolute Error [deg]')
plt.tight_layout()
plt.savefig('hstep_extension.pdf')

#%%
plt.figure(figsize=(4,3))
import seaborn as sns
df = pd.read_csv('filling_study.csv')





sns.boxplot(x=df['smartfill'], y=df['MAE'],color='gray' )
labels = np.round(np.linspace(0,1,11),1).astype(str)

labels[0] = 'FF only'
labels[-1]= 'LI only'
plt.xticks(np.linspace(0,10,11),
           labels,
           rotation=90)
plt.xlabel('smartfil threshold')
#plt.xticks(np.linspace(0,11,11), np.linspace(0,1,11))

plt.ylabel('Mean Absolute Error [deg]')
plt.tight_layout()

plt.savefig('smartfill_study.pdf')
