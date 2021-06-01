#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:54:29 2021

@author: llothar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['science','no-latex'])
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
        axs[i].hlines(y_no, 0, 100, label='no resampling\nmean', color='black',
                   linewidth=1, linestyle="--")
        y_no = np.percentile(dft['MAE'],5)
        axs[i].hlines(y_no, 0, 100, label='no resampling\n$5^{th}$, $95^{th}$ perc.', color='black',
                    linewidth=0.7, linestyle='--')
        y_no = np.percentile(dft['MAE'],95)
        axs[i].hlines(y_no, 0, 100, color='black',
                    linewidth=0.7, linestyle='--')
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
        y_no = np.percentile(dft['usable'],5)
        axs[i].hlines(y_no, 0, 100, label='no res. min/max', color='black',
                    linewidth=0.7, linestyle='--')
        y_no = np.percentile(dft['usable'],95)
        axs[i].hlines(y_no, 0, 100, color='black',
                    linewidth=0.7, linestyle='--')
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
p95 = []
p5 = []
maxstep = int(np.max(df['hstep_extension']))
for i in range(0,maxstep):
    results.append(np.average(df[df['hstep_extension'] == i+1]['MAE']))
    p95.append(np.percentile(df[df['hstep_extension'] == i+1]['MAE'],95))
    p5.append(np.percentile(df[df['hstep_extension'] == i+1]['MAE'],5))
    
#plt.grid
#plt.scatter(x = np.arange(1,maxstep+1,1), y=results, marker="x",
#            c='red', s=20)

plt.plot(np.arange(1,maxstep+1,1), results, c='tab:red',
         linestyle = '-', label='average', linewidth=2)
#plt.plot(np.arange(1,maxstep+1,1), p95, c='red')
#plt.plot(np.arange(1,maxstep+1,1), p5, c='red')

plt.scatter(x = df['hstep_extension'], y=df['MAE'], s=1, alpha=1, c='gray')
plt.grid()
# plt.scatter([],[], marker="x",
#             c='red', s=20,
#             label='Average result')

plt.scatter([],[],
            c='black', s=1,
            label='individual result')
plt.legend()
#plt.grid()
n = 10
xticklabels = np.linspace(0.153, 0.153*maxstep,n)
xticklabels = np.round(xticklabels, 2)

plt.xticks(np.linspace(1,maxstep+1,n), xticklabels, rotation=90)
plt.xlabel('Re-sampling rate [m]')
plt.ylabel('Mean Absolute Error [deg]')
plt.tight_layout()

plt.savefig('hstep_extension.pdf')

#%%

fig, ax = plt.subplots(figsize=(4,3))

df = pd.read_csv('f9ad.csv')

df = df.iloc[2000:10000]

cols = ['Measured Depth m', 'TOFB s', 'AVG_CONF unitless', 'MIN_CONF unitless', 'Average Rotary Speed rpm', 'STUCK_RT unitless', 'Corrected Surface Weight on Bit kkgf', 'Corrected Total Hookload kkgf', 'MWD Turbine RPM rpm', 'MWD Raw Gamma Ray 1/s', 'MWD Continuous Inclination dega', 'Pump 2 Stroke Rate 1/min', 'Rate of Penetration m/h', 'Bit Drill Time h', 'Corrected Hookload kkgf', 'MWD GR Bit Confidence Flag %', 'Pump Time h', 'PowerUP Shock Rate 1/s', 'Total SPM 1/min', 'Average Hookload kkgf', 'Total Hookload kkgf', 'Extrapolated Hole TVD m', 'MWD Gamma Ray (API BH corrected) gAPI', 'EDRT unitless', 'Pump 1 Stroke Rate 1/min', 'Total Bit Revolutions unitless', 'Mud Density In g/cm3.1', 'Weight on Bit kkgf', 'Hole Depth (TVD) m', 'MWD Shock Risk unitless', 'Bit run number unitless', 'Inverse ROP s/m', 'Pump 4 Stroke Rate 1/min', 'Rig Mode unitless', 'MWD Shock Peak m/s2', 'SPN Sp_RigMode 2hz unitless', 'Average Standpipe Pressure kPa', 'Rate of Penetration (5ft avg) m/h', 'AJAM_MWD unitless', '1/2ft ROP m/h', 'Hole depth (MD) m', 'Mud Flow In L/min', 'BHFG unitless', 'MWD DNI Temperature degC', 'Average Surface Torque kN.m', 'Total Downhole RPM rpm', 'SHK3TM_RT min', 'Pump 3 Stroke Rate 1/min', 'Inverse ROP (5ft avg) s/m', 'S1AC kPa', 'S2AC kPa', 'IMWT g/cm3', 'OSTM s']

df = df[cols]
smartfills = np.linspace(0,1.0000001,11)
counts = []
percentages = []

for smartfill in smartfills:
    fill_method = []
    
    for attribute in list(df):
                
        try:
            dropna_diff = np.diff(df[attribute].dropna())
        
            try:
                zeros_p = np.count_nonzero(dropna_diff == 0) / len(dropna_diff)
                
                if zeros_p >= smartfill: # Threshold to check?
                    fill_method.append(1)
                else:
                    fill_method.append(0)
            except:
                pass
                #print(f'{attribute} failed 1')
        except:
            pass
            #print(f'{attribute} failed 2')
            
    #print(smartfill)        
    #print(f'Average {np.mean(fill_method)}')
    #print(f'Count {np.sum(fill_method)}')
    counts.append(np.sum(fill_method))
    percentages.append(np.mean(fill_method))

percentages = np.asarray(percentages)
percentages = percentages*100

#plt.figure(figsize=(4,3))
import seaborn as sns
df = pd.read_csv('filling_study.csv')





ax = sns.boxplot(x=df['smartfill'], y=df['MAE'],color='gray' )
labels = np.round(np.linspace(0,100,11),0).astype(int)

#labels[0] = 'FF only'
#labels[-1]= 'LI only'
plt.xticks(np.linspace(0,10,11),
           labels,
           rotation=90)
plt.xlabel('''Imputation algorithm selection threshold [%]''')
#plt.xticks(np.linspace(0,11,11), np.linspace(0,1,11))

plt.ylabel('Mean Absolute Error [deg]')
plt.tight_layout()
plt.grid()

ax2 = ax.twinx() 

ax2.plot(percentages, color='red', linestyle = '--', label='percent FFilled',
         linewidth=2)
ax2.set_ylabel('Percentage of attributes \n forward-filled', color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.savefig('smartfill_study.pdf')
plt.show()
A = df[np.round(df['smartfill'],1) == 1]['MAE'].to_list()
B = df[np.round(df['smartfill'],1) == 0.2]['MAE'].to_list()
from scipy import stats
t_check=stats.ttest_ind(A,B)
print(t_check[1])
#%%

fig, ax = plt.subplots(figsize=(4,3))

df = pd.read_csv('f9ad.csv')

df = df.iloc[2000:10000]

cols = ['Measured Depth m', 'TOFB s', 'AVG_CONF unitless', 'MIN_CONF unitless', 'Average Rotary Speed rpm', 'STUCK_RT unitless', 'Corrected Surface Weight on Bit kkgf', 'Corrected Total Hookload kkgf', 'MWD Turbine RPM rpm', 'MWD Raw Gamma Ray 1/s', 'MWD Continuous Inclination dega', 'Pump 2 Stroke Rate 1/min', 'Rate of Penetration m/h', 'Bit Drill Time h', 'Corrected Hookload kkgf', 'MWD GR Bit Confidence Flag %', 'Pump Time h', 'PowerUP Shock Rate 1/s', 'Total SPM 1/min', 'Average Hookload kkgf', 'Total Hookload kkgf', 'Extrapolated Hole TVD m', 'MWD Gamma Ray (API BH corrected) gAPI', 'EDRT unitless', 'Pump 1 Stroke Rate 1/min', 'Total Bit Revolutions unitless', 'Mud Density In g/cm3.1', 'Weight on Bit kkgf', 'Hole Depth (TVD) m', 'MWD Shock Risk unitless', 'Bit run number unitless', 'Inverse ROP s/m', 'Pump 4 Stroke Rate 1/min', 'Rig Mode unitless', 'MWD Shock Peak m/s2', 'SPN Sp_RigMode 2hz unitless', 'Average Standpipe Pressure kPa', 'Rate of Penetration (5ft avg) m/h', 'AJAM_MWD unitless', '1/2ft ROP m/h', 'Hole depth (MD) m', 'Mud Flow In L/min', 'BHFG unitless', 'MWD DNI Temperature degC', 'Average Surface Torque kN.m', 'Total Downhole RPM rpm', 'SHK3TM_RT min', 'Pump 3 Stroke Rate 1/min', 'Inverse ROP (5ft avg) s/m', 'S1AC kPa', 'S2AC kPa', 'IMWT g/cm3', 'OSTM s']

df = df[cols]
smartfills = np.linspace(0,1.0000001,11)
counts = []
percentages = []

for smartfill in smartfills:
    fill_method = []
    
    for attribute in list(df):
                
        try:
            dropna_diff = np.diff(df[attribute].dropna())
        
            try:
                zeros_p = np.count_nonzero(dropna_diff == 0) / len(dropna_diff)
                
                if zeros_p >= smartfill: # Threshold to check?
                    fill_method.append(1)
                else:
                    fill_method.append(0)
            except:
                pass
                #print(f'{attribute} failed 1')
        except:
            pass
            #print(f'{attribute} failed 2')
            
    #print(smartfill)        
    #print(f'Average {np.mean(fill_method)}')
    #print(f'Count {np.sum(fill_method)}')
    counts.append(np.sum(fill_method))
    percentages.append(np.mean(fill_method))

percentages = np.asarray(percentages)
percentages = percentages*100

#plt.figure(figsize=(4,3))
import seaborn as sns
df = pd.read_csv('filling_study_rop.csv')





ax = sns.boxplot(x=df['smartfill'], y=df['MAE'],color='gray' )
labels = np.round(np.linspace(0,100,11),0).astype(int)

#labels[0] = 'FF only'
#labels[-1]= 'LI only'
plt.xticks(np.linspace(0,10,11),
           labels,
           rotation=90)
plt.xlabel('''Imputation algorithm selection threshold [%]''')
#plt.xticks(np.linspace(0,11,11), np.linspace(0,1,11))

plt.ylabel('Mean Absolute Error [deg]')
plt.tight_layout()
plt.grid()

ax2 = ax.twinx() 

ax2.plot(percentages, color='red', linestyle = '--', label='percent FFilled',
         linewidth=2)
ax2.set_ylabel('Percentage of attributes \n forward-filled', color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.savefig('smartfill_study_rop.pdf')
plt.show()
A = df[np.round(df['smartfill'],1) == 1]['MAE'].to_list()
B = df[np.round(df['smartfill'],1) == 0.2]['MAE'].to_list()
from scipy import stats
t_check=stats.ttest_ind(A,B)
print(t_check[1])
#%%

df = pd.read_csv('selstrat_study.csv')

df['strategy'] = df['strategy'].replace(['pca'],'PCA norm.')
df['strategy'] = df['strategy'].replace(['pca_ss'],'PCA stand.')

plt.figure(figsize=(5,5))
strategies = ['ppscore', 'pearson', 'PCA norm.', 'PCA stand.']

colors = {'ppscore' : 'tab:green',
          'pearson' : 'tab:orange',
          'PCA norm.' : 'tab:blue',
          'PCA stand.' : 'tab:red'}

labels = {'ppscore' : 'mean ppscore',
          'pearson' : 'mean pearson',
          'PCA norm.' : 'mean PCA norm.',
          'PCA stand.' : 'mean PCA stand.'}

for strategy in strategies:
    averages = []
    unique = np.unique(df[df['strategy'] == strategy]['qty'])
    for qty in unique:
        dft = df[df['strategy'] == strategy]
        dft = dft[dft['qty'] == qty]
        averages.append(np.mean(dft['MAE']))
        
    plt.plot(averages, c=colors[strategy], linestyle='--',
             linewidth=2, label=labels[strategy])


#sns.boxplot(data = df, x='qty', y='MAE', hue = 'strategy')

sns.swarmplot(data = df, x='qty', y='MAE', hue = 'strategy', s=4.5, dodge=False)
plt.grid()

handles, labels = plt.gca().get_legend_handles_labels()

order = [1,0,2,3,6,5,4,7]

plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
           bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
plt.xlabel('Parameter quantity')
plt.ylabel('Mean Absolute Error [degrees]')
plt.tight_layout()
plt.savefig('selstrat_lcs.pdf')
plt.show()
#%%

df = pd.read_csv('selstrat_study_delta.csv')
plt.figure(figsize=(5,5))
df['strategy'] = df['strategy'].replace(['pca'],'PCA')

plt.figure(figsize=(5,5))
strategies = ['ppscore', 'pearson', 'PCA']

colors = {'ppscore' : 'tab:green',
          'pearson' : 'tab:orange',
          'PCA' : 'tab:blue'}

labels = {'ppscore' : 'mean ppscore',
          'pearson' : 'mean pearson',
          'PCA' : 'mean PCA'}
for strategy in strategies:
    averages = []
    unique = np.unique(df[df['strategy'] == strategy]['qty'])
    for qty in unique:
        dft = df[df['strategy'] == strategy]
        dft = dft[dft['qty'] == qty]
        averages.append(np.average(dft['MAE']))
        
    plt.plot(averages, c=colors[strategy], linestyle='--', 
             linewidth=2, label=labels[strategy])

#sns.boxplot(data = df, x='qty', y='MAE', hue = 'strategy')

sns.swarmplot(data = df, x='qty', y='MAE', hue = 'strategy', s=6, dodge=False)
plt.grid()
plt.xlabel('Parameter quantity')
plt.ylabel('Mean Absolute Error [degrees]')
handles, labels = plt.gca().get_legend_handles_labels()

order = [1,0,2,5,4,3]

plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
           bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
plt.tight_layout()
plt.savefig('selstrat_lcs_delta.pdf')

#%%

volve_depth = np.load('resultfile.npy')
plt.figure(figsize=(5,5))
plt.bar([0,1,2,3], volve_depth)
plt.grid()
plt.xlabel('Resampling method with lowest area between lines')
plt.xticks([0,1,2,3], ['KNN\nuniform', 'KNN\ndistance', 'RNR\nuniform', 'RNR\ndistance'])
plt.ylabel('Attribute count')
plt.savefig('betweenlines.pdf')