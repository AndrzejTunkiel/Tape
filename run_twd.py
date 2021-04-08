# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:23:34 2021

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

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

splits = np.linspace(0.15, 0.8, 30)

truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    truth, pred, columns, score = tape(data, split=split,
                                      drops=drops,
                                      index = 'Measured Depth m',
                                      target =  'MWD Continuous Inclination dega',
                                      convert_to_diff = [],
                                      lcs_list = ['MWD Continuous Inclination dega'],
                                      plot_samples = False,
                                      resample='radius',
                                      hPcaScaler='mm')
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(np.average(diffs))
plt.show()

truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    truth, pred, columns, score = tape(data, split=split,
                                      drops=drops,
                                      index = 'Measured Depth m',
                                      target =  'MWD Continuous Inclination dega',
                                      convert_to_diff = [],
                                      lcs_list = ['MWD Continuous Inclination dega'],
                                      plot_samples = False,
                                      resample='radius',
                                      hPcaScaler='ss')
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(np.average(diffs))
plt.show()
#%%

truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    truth, pred, columns, score = tape(data, split=split,
                                      drops=drops,
                                      index = 'Measured Depth m',
                                      target =  'MWD Continuous Inclination dega',
                                      convert_to_diff = [],
                                      lcs_list = ['MWD Continuous Inclination dega'],
                                      plot_samples = False,
                                      resample='knn')
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(np.average(diffs))
plt.show()
#%%


truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    truth, pred, columns, score = tape(data, split=split,
                                      drops=drops,
                                      index = 'Measured Depth m',
                                      target =  'MWD Continuous Inclination dega',
                                       asel_choice = 'ppscore',
                                       hAttrCount=6,
                                      convert_to_diff = [],
                                      lcs_list = ['MWD Continuous Inclination dega'],
                                      plot_samples = False)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(np.average(diffs))
plt.show()





truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    truth, pred, columns, score = tape(data, split=split,
                                      drops=drops,
                                      index = 'Measured Depth m',
                                      target =  'MWD Continuous Inclination dega',
                                       asel_choice = 'pearson',
                                       hAttrCount=6,
                                      convert_to_diff = [],
                                      lcs_list = ['MWD Continuous Inclination dega'],
                                      plot_samples = False)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(np.average(diffs))
plt.show()
#%%

truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    truth, pred, columns, score = tape(data, split=split,
                                 drops=drops,
                                 index = 'Measured Depth m',
                                 target =  'MWD Continuous Inclination dega',
                                 convert_to_diff = ['MWD Continuous Inclination dega'],
                                 lcs_list = [],
                                 shift=0.1,
                                 plot_samples = False)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(np.average(diffs))
plt.show()


