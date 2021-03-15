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
        
axs[0].set_ylabel('Mean Absolute Error')


axs[0].set_xlabel(f'radius\nuniform weight')
axs[1].set_xlabel(f'radius\ndistance weight')
axs[2].set_xlabel(f'neighbors\nuniform weight')
axs[3].set_xlabel(f'neighbors\ndistance weight')
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
        
axs[0].set_ylabel('Prediction area under 1.2 deg. error')


axs[0].set_xlabel(f'radius\nuniform weight')
axs[1].set_xlabel(f'radius\ndistance weight')
axs[2].set_xlabel(f'neighbors\nuniform weight')
axs[3].set_xlabel(f'neighbors\ndistance weight')
axs[3].legend()
plt.tight_layout()
plt.savefig('resampling useful.pdf')
plt.show()