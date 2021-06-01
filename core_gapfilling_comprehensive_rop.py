# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:05:50 2021

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""
import pandas as pd
import numpy as np
from sens_tape import tape

data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
              'Rate of Penetration m/h',
              'Rate of Penetration (5ft avg) m/h']

splits = np.linspace(0.25, 0.8, 50)

df = pd.DataFrame(columns=['smartfill', 'MAE'])

smartfills = np.linspace(0,1.00001,11)
h5prefix = input('Prefix')
while True:
    for smartfill in smartfills:
        truth_array = []
        pred_array = []
        columns_array = []
        score_array = []
        for split in splits:
            print(split)
            truth, pred, columns, score = tape(data, split=split,
                                              drops=drops,
                                              index = 'Measured Depth m',
                                              target =  'Rate of penetration m/h',
                                              convert_to_diff = [],
                                              lcs_list = [],
                                              resample='radius',
                                              plot_samples = False,
                                              resample_coef=5,
                                              resample_weights='distance',
                                              hstep_extension = 5,
                                              smartfill=smartfill,
                                              h5prefix = h5prefix)
            
            truth_array.append(truth)
            pred_array.append(pred)
            columns_array.append(columns)
            score_array.append(score)
            
        diffs = []
        
        for i in range(len(truth_array)):
            diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))
            
        new_row = {'smartfill' : smartfill,
                   'MAE' : np.average(diffs)}
        print(new_row)
        df = df.append(new_row, ignore_index = True)
        df.to_csv('filling_study_rop.csv')

