#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:07:44 2021

@author: atunkiel
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




splits = np.linspace(0.15, 0.8, 100)
truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    print(split)
    truth, pred, columns, score = tape(data, split=split,
                                      drops=drops,
                                      index = 'Measured Depth m',
                                      target =  'MWD Continuous Inclination dega',
                                      convert_to_diff = [],
                                      lcs_list = ['MWD Continuous Inclination dega'],
                                      resample='knn',
                                      plot_samples = False,
                                      resample_coef=1,
                                      resample_weights='distance')
    
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
plt.savefig('knn1d.pdf')
np.save('knn1d.npy', diffs)

###


truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    print(split)
    truth, pred, columns, score = tape(data, split=split,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  resample='knn',
                                  resample_coef=1,
                                  plot_samples = False,
                                  resample_weights='uniform')
    
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
plt.savefig('knn1u.pdf')
np.save('knn1u.npy', diffs)

###


truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    print(split)
    truth, pred, columns, score = tape(data, split=split,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  resample='knn',
                                  resample_coef=8,
                                  plot_samples = False,
                                  resample_weights='distance')
    
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
plt.savefig('knn8d.pdf')
np.save('knn8d.npy', diffs)


###


truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    print(split)
    truth, pred, columns, score = tape(data, split=split,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  resample='knn',
                                  resample_coef=8,
                                  plot_samples = False,
                                  resample_weights='uniform')
    
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
plt.savefig('knn8u.pdf')
np.save('knn8u.npy', diffs)


truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    print(split)
    truth, pred, columns, score = tape(data, split=split,
                                      drops=drops,
                                      index = 'Measured Depth m',
                                      target =  'MWD Continuous Inclination dega',
                                      convert_to_diff = [],
                                      lcs_list = ['MWD Continuous Inclination dega'],
                                      resample='radius',
                                      plot_samples = False,
                                      resample_coef=1,
                                      resample_weights='distance')
    
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
plt.savefig('r1d.pdf')
np.save('r1d.npy', diffs)

###


truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    print(split)
    truth, pred, columns, score = tape(data, split=split,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  resample='radius',
                                  resample_coef=1,
                                  plot_samples = False,
                                  resample_weights='uniform')
    
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
plt.savefig('r1u.pdf')
np.save('r1u.npy', diffs)

###


truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    print(split)
    truth, pred, columns, score = tape(data, split=split,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  resample='radius',
                                  resample_coef=8,
                                  plot_samples = False,
                                  resample_weights='distance')
    
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
plt.savefig('r8d.pdf')
np.save('r8d.npy', diffs)

###


truth_array = []
pred_array = []
columns_array = []
score_array = []
for split in splits:
    print(split)
    truth, pred, columns, score = tape(data, split=split,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  resample='radius',
                                  resample_coef=8,
                                  plot_samples = False,
                                  resample_weights='uniform')
    
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
plt.savefig('r8u.pdf')
np.save('r8u.npy', diffs)