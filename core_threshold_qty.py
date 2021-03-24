#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:13:35 2021

@author: llothar
"""

import pandas as pd
import numpy as np

df = pd.read_csv('f9ad.csv')

df = df.iloc[2000:10000]

cols = ['Measured Depth m', 'TOFB s', 'AVG_CONF unitless', 'MIN_CONF unitless', 'Average Rotary Speed rpm', 'STUCK_RT unitless', 'Corrected Surface Weight on Bit kkgf', 'Corrected Total Hookload kkgf', 'MWD Turbine RPM rpm', 'MWD Raw Gamma Ray 1/s', 'MWD Continuous Inclination dega', 'Pump 2 Stroke Rate 1/min', 'Rate of Penetration m/h', 'Bit Drill Time h', 'Corrected Hookload kkgf', 'MWD GR Bit Confidence Flag %', 'Pump Time h', 'PowerUP Shock Rate 1/s', 'Total SPM 1/min', 'Average Hookload kkgf', 'Total Hookload kkgf', 'Extrapolated Hole TVD m', 'MWD Gamma Ray (API BH corrected) gAPI', 'EDRT unitless', 'Pump 1 Stroke Rate 1/min', 'Total Bit Revolutions unitless', 'Mud Density In g/cm3.1', 'Weight on Bit kkgf', 'Hole Depth (TVD) m', 'MWD Shock Risk unitless', 'Bit run number unitless', 'Inverse ROP s/m', 'Pump 4 Stroke Rate 1/min', 'Rig Mode unitless', 'MWD Shock Peak m/s2', 'SPN Sp_RigMode 2hz unitless', 'Average Standpipe Pressure kPa', 'Rate of Penetration (5ft avg) m/h', 'AJAM_MWD unitless', '1/2ft ROP m/h', 'Hole depth (MD) m', 'Mud Flow In L/min', 'BHFG unitless', 'MWD DNI Temperature degC', 'Average Surface Torque kN.m', 'Total Downhole RPM rpm', 'SHK3TM_RT min', 'Pump 3 Stroke Rate 1/min', 'Inverse ROP (5ft avg) s/m', 'S1AC kPa', 'S2AC kPa', 'IMWT g/cm3', 'OSTM s']

df = df[cols]
smartfills = np.linspace(0,1,11)
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
