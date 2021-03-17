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

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
                        'MWD Continuous Azimuth dega']

splits = np.linspace(0.15, 0.8, 50)

df = pd.DataFrame(columns=['smartfill', 'MAE'])

smartfills = np.linspace(0,1,11)

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
                                              target =  'MWD Continuous Inclination dega',
                                              convert_to_diff = [],
                                              lcs_list = ['MWD Continuous Inclination dega'],
                                              resample='radius',
                                              plot_samples = False,
                                              resample_coef=5,
                                              resample_weights='distance',
                                              hstep_extension = 5,
                                              smartfill=smartfill)
            
            truth_array.append(truth)
            pred_array.append(pred)
            columns_array.append(columns)
            score_array.append(score)
            
        diffs = []
        
        for i in range(len(truth_array)):
            diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))
            
        new_row = {'smartfill' : smartfill,
                   'MAE' : np.average(diffs)}
        df = df.append(new_row, ignore_index = True)
        df.to_csv('filling_study.csv')

