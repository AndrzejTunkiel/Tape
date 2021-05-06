#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:25:08 2021

@author: llothar
"""

from sens_tape import tape
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np

data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
                        'MWD Continuous Azimuth dega','STUCK_RT unitless']

splits = np.linspace(0.15,1,41)
for split in splits:
    truth, pred, columns, score, senstable = tape(data, split=split,
                                      drops=drops,
                                      index = 'Measured Depth m',
                                      target =  'MWD Continuous Inclination dega',
                                      convert_to_diff = [],
                                      plot_samples = False,
                                      lcs_list = ['MWD Continuous Inclination dega'],
                                      asel_choice='pca',
                                      hAttrCount=5,
                                      sensitivity_analysis = True)

    with open('sens_twd.csv','a') as f:
        w = csv.writer(f)
        w.writerow(senstable.keys())
        w.writerow(senstable.values())