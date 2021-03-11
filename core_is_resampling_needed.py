#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:17:48 2021

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
                                      resample='no',
                                      plot_samples = False,
                                      resample_coef=1,
                                      resample_weights='distance',
                                      hstep_extension = 20)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(f'''MAE: {np.average(diffs):.2f},
          1.2% area: {np.average(np.asarray(diffs) < 1.2)*100:.1f}%''')

plt.savefig('h20no.pdf')
np.save('h20no.npy', diffs)
plt.clf()

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
                                      resample_weights='uniform',
                                      hstep_extension = 20)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(f'''MAE: {np.average(diffs):.2f},
          1.2% area: {np.average(np.asarray(diffs) < 1.2)*100:.1f}%''')

plt.savefig('h20r1u.pdf')
np.save('h20r1u.npy', diffs)
plt.clf()

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
                                      resample='no',
                                      plot_samples = False,
                                      resample_coef=1,
                                      resample_weights='distance',
                                      hstep_extension = 10)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(f'''MAE: {np.average(diffs):.2f},
          1.2% area: {np.average(np.asarray(diffs) < 1.2)*100:.1f}%''')


plt.savefig('h10no.pdf')
np.save('h10no.npy', diffs)
plt.clf()


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
                                      resample_weights='uniform',
                                      hstep_extension = 10)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(f'''MAE: {np.average(diffs):.2f},
          1.2% area: {np.average(np.asarray(diffs) < 1.2)*100:.1f}%''')
plt.savefig('h10r1u.pdf')
np.save('h10r1u.npy', diffs)
plt.clf()


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
                                      resample='no',
                                      plot_samples = False,
                                      resample_coef=1,
                                      resample_weights='distance',
                                      hstep_extension = 5)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(f'''MAE: {np.average(diffs):.2f},
          1.2% area: {np.average(np.asarray(diffs) < 1.2)*100:.1f}%''')

plt.savefig('h5no.pdf')
np.save('h5no.npy', diffs)
plt.clf()

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
                                      resample_weights='uniform',
                                      hstep_extension = 5)
    
    truth_array.append(truth)
    pred_array.append(pred)
    columns_array.append(columns)
    score_array.append(score)
    
diffs = []

for i in range(len(truth_array)):
    diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))

sns.heatmap(diffs, vmax=1.2, vmin=0, cmap='viridis')
plt.title(f'''MAE: {np.average(diffs):.2f},
          1.2% area: {np.average(np.asarray(diffs) < 1.2)*100:.1f}%''')

plt.savefig('h5r1u.pdf')
np.save('h5r1u.npy', diffs)
plt.clf()
