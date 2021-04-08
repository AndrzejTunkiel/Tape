# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:05:50 2021

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""
import pandas as pd
import numpy as np
from sens_tape import tape

h5prefix = input('Unique prefix for model save file h5: ')

data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
                        'MWD Continuous Azimuth dega']

splits = np.linspace(0.15, 0.8, 50)




qtys = np.arange(1,11,1)


for iters in range(6):
    for qty in qtys:
   
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
                                              smartfill=0.2,
                                              hAttrCount=qty,
                                              h5prefix=h5prefix,
                                              asel_choice = 'pca',
                                              hPcaScaler = 'ss')
            
            truth_array.append(truth)
            pred_array.append(pred)
            columns_array.append(columns)
            score_array.append(score)
            
        diffs = []
        
        for i in range(len(truth_array)):
            diffs.append(np.average(np.abs(truth_array[i] - pred_array[i]), axis=0))
        
        df = pd.DataFrame(columns=['strategy','qty', 'MAE'])    
        new_row = {'strategy' : 'pca_ss',
                   'qty' : qty,
                   'MAE' : np.average(diffs)}
        df = df.append(new_row, ignore_index = True)
        df.to_csv('selstrat_study.csv', mode='a', header=False, index=False)

