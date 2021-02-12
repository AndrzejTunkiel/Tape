# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:36:57 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no
"""

import logging

from logging import debug
from logging import info
from logging import warning
from logging import error
from logging import critical

logging.basicConfig(format='%(asctime)s - %(levelname)s:%(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', 
                    #filename='app.log',
                    level=logging.WARNING)


#%%

import warnings
warnings.filterwarnings('ignore')

from numpy.random import seed
seed(0)
import tensorflow as tf

tf.random.set_seed(0)

info(tf.__version__)

import pandas as pd
import numpy as np


#%%

from training_delta_best import train_delta
from training_nominal import train_nominal
from training_nominal_middleVal import train_nominal_middleVal

from supplemental_functions import expandaxis

begin = 500
middle = 800
end = 843

model_array = []
val_loss_array = []
hypers_array = []


percent_drilled = np.arange(15,81,1)

#percent_drilled = [60]
matrix_size = len(percent_drilled)




newdim = np.full((matrix_size,),3)
start = np.full((matrix_size,), begin)
stop = np.full((matrix_size,), end)
inc_layer1 = np.full((matrix_size,), 371) #was256
inc_layer2 = np.full((matrix_size,), 2) #np.full((matrix_size,), 48) 
#gaussian noise now, divided by 1000.
data_layer1 = np.full((matrix_size,), 1) 
data_layer2 = np.full((matrix_size,), 1) #drop2
dense_layer = np.full((matrix_size,), 8) #was139 
                                        #np.arange(139-step, 139+step+1, step) 
range_max =np.full((matrix_size,), 1)  #DISABLED
memory =  np.full((matrix_size,), 100) #np.arange(70, 101, step) 
                                    # np.full((matrix_size,), 200) 
                                    #was86 #np.arange(86-step, 86+step+1, step) 
predictions = np.full((matrix_size,), 100)
drop1 = np.full((matrix_size,), 50)
drop2 = np.full((matrix_size,), 0) #np.random.randint(50,90,size=matrix_size)
lr = np.full((matrix_size,), 40) #was 16
bs = np.full((matrix_size,), 32)
ensemble_count = np.full((matrix_size,),1)

inputs = [newdim, percent_drilled, start, stop, inc_layer1,
                    inc_layer2,data_layer1,data_layer2,dense_layer,
                    range_max, memory, predictions, drop1, drop2, lr, bs, ensemble_count]

for loc, val in enumerate(inputs):
    inputs[loc] = expandaxis(val)
    

hypers = np.hstack(inputs)

ID = np.random.randint(0,999999)

histories = []
val_loss_array = []
aae_array = []
ypred_array = []
ytrue_array = []

for i, val in enumerate(hypers):
    print (f'Evaluating {i+1}/{len(hypers)}')
   
    val_loss, test_loss, aae,ypred, ytrue = train_delta(val)
    aae_array.append(aae)
    ypred_array.append(ypred)
    ytrue_array.append(ytrue)
    
# =============================================================================
#     val_loss_array.append(val_loss)
#     hypers_array.append(i)
#     print (i)
#     print (val_loss)
#     output = np.append(hypers_array, expandaxis(val_loss_array), axis=1)
#     output = pd.DataFrame(output,columns=["PCA dim count",
#                                           "percentage_drilled", "start",
#                                           "stop", "inc_layer1","inc_layer2",
#                                           "data_layer1","data_layer2",
#                                           "dense_layer"," range_max",
#                                           " memory"," predictions","drop1",
#                                           "drop2", "val loss"])
# =============================================================================
# =============================================================================
#     try:
#         #print (output)
#         output.to_csv("FastICA " + str(ID) +".csv")
#     except:
#         print("File opened?")
# 
# =============================================================================

#%%
np.save('delta best ref aae.npy', aae_array)    
np.save('delta best ref ypred.npy', ypred_array)    
np.save('delta best ref ytrue.npy', ytrue_array)    

import matplotlib.pyplot as plt
import seaborn as sns
print(np.average(aae_array))
sns.heatmap(aae_array)
#np.save('delta ave no lottery.npy', aae_array)
#%%

aae_array = np.load('delta best ref aae.npy', allow_pickle=True)
ypred_array = np.load('delta best ref ypred.npy', allow_pickle=True)
ytrue_array = np.load('delta best ref ytrue.npy', allow_pickle=True)
#%%
from sklearn.metrics import r2_score

coor_true_array = []
coor_pred_array = []
x_r2_array = []
y_r2_array = []
dist_array = []

for i in range(len(ypred_array)):
    y_true = ytrue_array[i]
    y_pred = ypred_array[i]
    
    y_true = np.radians(y_true)
    y_pred = np.radians(y_pred)

    coor_true = np.zeros((y_true.shape[0],2))

    for i in range(y_true.shape[0]):
        dx = 0
        dy = 0
        for j in range(y_true.shape[1]):
            dx = dx + np.cos(y_true[i,j])*0.230876
            dy = dy + np.sin(y_true[i,j])*0.230876
        coor_true[i] = [dx,dy]

    coor_pred = np.zeros((y_pred.shape[0],2))

    for i in range(y_pred.shape[0]):
        dx = 0
        dy = 0
        for j in range(y_pred.shape[1]):
            dx = dx + np.cos(y_pred[i,j])*0.230876
            dy = dy + np.sin(y_pred[i,j])*0.230876
        coor_pred[i] = [dx,dy]
    
    dist = np.zeros(coor_true.shape[0])
    for i in range(len(coor_true)):
        dist[i] = np.sqrt( (coor_true[i,0] - coor_pred[i,0])**2 + (coor_true[i,1] - coor_pred[i,1])**2)
    
    dist_array.append(dist*1000)
    coor_true_array.append(coor_true)
    coor_pred_array.append(coor_pred)
    x_r2_array.append(r2_score(coor_true[:,0], coor_pred[:,0]))
    y_r2_array.append(r2_score(coor_true[:,1], coor_pred[:,1]))


#%%    
data_x = []
data_y = []
for per in range(len(percent_drilled)):
    for i in dist_array[per]:
        data_x.append(percent_drilled[per])
        data_y.append(i)

ax = sns.kdeplot(data_y, data_x, shade=True, shade_lowest=True,
                 cmap='viridis')
#%%
for i in range(len(x_r2_array)):
    if x_r2_array[i] < 0:
        x_r2_array[i] = 0

for i in range(len(y_r2_array)):
    if y_r2_array[i] < 0:
        y_r2_array[i] = 0



r2_angle = []
for i in range(len(ytrue_array)):
    r2_angle.append(r2_score(ytrue_array[i],ypred_array[i]))

for i in range(len(r2_angle)):
    if r2_angle[i] < 0:
        r2_angle[i] = 0

x = np.linspace(552,778,66)
myxticks = np.linspace(552,778,10).astype(int)

plt.figure(figsize=(3,3))
plt.scatter(x_r2_array,x, marker="x", color="black", label=r"x-coordinate")
plt.grid()
plt.xticks(np.arange(-2,1.1,0.1), rotation=90)
plt.xlim(0,1)
plt.yticks(myxticks)
plt.title("x-coordinate")
plt.ylabel('measured depth drilled')
plt.xlabel(r'$R^2$')
plt.ylim(778,552)
plt.tight_layout()
plt.savefig('r2x.pdf')
plt.show()

plt.figure(figsize=(3,3))
plt.scatter(y_r2_array,x, marker="x", color="black", label=r"y-coordinate")
plt.grid()
plt.xticks(np.arange(-2,1.1,0.1), rotation=90)
plt.xlim(0,1)
plt.yticks(myxticks)
plt.title("y-coordinate")
plt.ylabel('measured depth drilled')
plt.xlabel(r'$R^2$')
plt.ylim(778,552)
plt.tight_layout()
plt.savefig('r2y.pdf')
plt.show()




plt.figure(figsize=(3,3))
plt.scatter(r2_angle,x, label=r"angle",  marker="x", color="black")
plt.grid()
plt.xticks(np.arange(-2,1.1,0.1), rotation=90)
plt.xlim(0,1)
plt.yticks(myxticks)
plt.title("angle")
plt.xlabel(r'$R^2$')
plt.ylabel('measured depth drilled')
plt.ylim(778,552)
plt.tight_layout()
plt.savefig('r2angle.pdf')
plt.show()

#%%

percent_drilled = np.arange(15,81,1)

plt.figure(figsize=(3,3))
data_x = []
data_y = []
for per in range(len(percent_drilled)):
    for i in dist_array[per]:
        data_x.append(percent_drilled[per])
        data_y.append(i)

ax = sns.kdeplot(data_y, data_x, shade=True, shade_lowest=True, n_levels=10,
                 cmap='viridis')

ax.invert_yaxis()
ax.set_xlim(0,1200)
ax.set_xlabel('Position error at bit, \n23m sensor-bit distance [m]')
ax.set_xticks(np.linspace(0,1200,10))
ax.set_xticklabels(labels=np.round(np.arange(0,1.81,0.2),1), rotation=90)

ax.set_yticks(np.linspace(15,80,10))
ax.set_yticklabels((np.linspace(552,778,10).astype(int)))
ax.set_ylim(80,15)
ax.set_ylabel('Measure Depth drilled [m]')
plt.tight_layout()
plt.savefig('position_error.pdf')
plt.show()

#%%

plt.figure(figsize=(3,3))
data = aae_array

xticks = np.array(np.linspace(0,23,7),dtype=int)
yticks = np.array(np.linspace(552,778,11),dtype=int)


mean = np.round(np.mean(data),2)
sns.heatmap(data, cmap="RdYlGn_r", vmax=1.2, vmin=0,
            linewidths=False, cbar_kws={'label'  : 'Mean Absolute Error [$^{\circ}$]'})

plt.xticks(np.arange(0,101,10), labels=np.round(np.linspace(0,23,11),0).astype(int))

plt.ylabel('Measured Depth drilled [m]')
plt.yticks(np.linspace(0,data.shape[0],10),
           labels=np.round(np.linspace(552,778,10),0).astype(int),
           rotation=0)
plt.xlabel('Prediction distance [m]')
plt.tight_layout()
plt.savefig('MAE angle.pdf')
