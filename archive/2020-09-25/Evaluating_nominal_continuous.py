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

from sklearn.metrics import r2_score
#%%

from training_nominal_ensemble import train_nominal as train_nominal_e
from training_nominal_best import train_nominal as train_nominal_b


from supplemental_functions import expandaxis

begin = 500
middle = 800
end = 843

model_array = []
val_loss_array = []
hypers_array = []


percent_drilled = np.arange(15,81,1)

#percent_drilled = [20]
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
ensemble_count = np.full((matrix_size,),10)

inputs = [newdim, percent_drilled, start, stop, inc_layer1,
                    inc_layer2,data_layer1,data_layer2,dense_layer,
                    range_max, memory, predictions, drop1, drop2, lr, bs, ensemble_count]

for loc, val in enumerate(inputs):
    inputs[loc] = expandaxis(val)
    

hypers = np.hstack(inputs)

ID = np.random.randint(0,999999)
aaes_ticket = []
aaes_ave = []
r2_ticket = []
r2_ave = []

while True:
    aae_array = []
    r2_array = []
    
    for i, val in enumerate(hypers):
        print (f'Evaluating {i+1}/{len(hypers)}')
       
        val_loss, test_loss, aae,ypred, ytrue = train_nominal_b(val)
        aae_array.append(aae)
        r2_array.append(r2_score(ypred, ytrue))
        
    
    aaes_ticket.append(np.mean(aae_array))
    r2_ticket.append(np.mean(r2_array))
    
    
    np.save('nom ticket_cont_results.npy', aaes_ticket)
    np.save('nom ticket_cont_results r2.npy', r2_ticket)
    
    aae_array = []
    r2_array = []
    
    for i, val in enumerate(hypers):
        print (f'Evaluating {i+1}/{len(hypers)}')
       
        val_loss, test_loss, aae,ypred, ytrue = train_nominal_e(val)
        aae_array.append(aae)  
        r2_array.append(r2_score(ypred, ytrue))

    aaes_ave.append(np.mean(aae_array))
    r2_ave.append(np.mean(r2_array))
    
    np.save('nom ave_cont_results.npy', aaes_ave)
    np.save('nom ave_cont_results r2.npy', r2_ticket)