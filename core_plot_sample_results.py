#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:37:07 2021

@author: llothar
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])
diffs = np.load('r1d.npy')


plt.figure(figsize=(4,4))
sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis',
            cbar_kws={'label': 'Mean Absolute Error [degrees]'})
#plt.title(np.average(diffs))
xn = 8
yn = 10
plt.xticks(np.linspace(0,113,xn), np.linspace(0,25,xn).astype(int))
plt.yticks(np.linspace(0,100,yn), np.linspace(15,80,yn).astype(int))
plt.xlabel('Prediction distance [m]')
plt.ylabel('Well drilled [%]')
plt.tight_layout()
plt.savefig('Example results.pdf')