#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:36:28 2021

@author: llothar
"""

from statistics_module import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('f9ad.csv')
#%%
s, m, per = stats(df)
target =  'MWD Continuous Inclination dega'

#plt.style.use(['science','no-latex'])
## Gap statistics for target
#
# This chart will show the percentage of dataset occupied by gaps of a certain
# size. Gaps are normal in drilling logs and nothing to be afraid of

x_label = per[target]['gap_sizes']
x = np.arange(0, len(x_label),1)

my_figsize = (7,2.5)

plt.figure(figsize=my_figsize)
y = per[target]['percentage_cells_occupied'] 
plt.xticks(x, x_label, rotation=90)
plt.bar(x,y, color='gray')
#plt.title(f'Gap distribution in:\n {target}')
plt.xlabel('Gap length [rows]')
plt.ylabel('Dataset occupied [%]')
plt.xlim(-1,51)
x_labels = x.tolist()
x_labels[0] = 'data'
plt.xticks(x, x_labels)
plt.grid()
plt.tight_layout()
plt.savefig('raw_gaps_stats.pdf')
plt.show() 



## Outlier detection

outlier_cutoff = 0.005 #arbitrarily selected

# calculation that penalizes long, rare, continuous gaps
out_coef = per[target]['gap_sizes'] / (per[target]['gap_counts'] * len(df))

x = np.arange(0,len(per[target]['gap_sizes']),1)
x_label = per[target]['gap_sizes']
x = np.arange(0, len(x_label),1)
plt.figure(figsize=my_figsize)
plt.xticks(x, x_label, rotation=90)
plt.bar(x,out_coef, color='gray')
#plt.ylim(0,0.005)
plt.plot([-1,51],[outlier_cutoff]*2, color='black', label='cutoff',
         linestyle='--')
plt.legend()
x_labels = x.tolist()
x_labels[0] = 'data'
plt.xticks(x, x_labels)
plt.xlim(-1,51)
#plt.title(f'Gap coefficient in: {target}')
plt.xlabel('Gap length [rows]')
plt.ylabel('Gap coefficient')
plt.grid()
plt.tight_layout()
plt.savefig('proc_gaps_stats.pdf')
plt.show() 