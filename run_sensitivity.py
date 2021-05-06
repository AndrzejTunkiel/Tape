# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:23:34 2021

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""

from sens_tape import tape
import pandas as pd
import matplotlib.pyplot as plt
import csv

data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
                        'MWD Continuous Azimuth dega','STUCK_RT unitless']
#%%
truth, pred, columns, score, senstable = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  plot_samples = False,
                                  lcs_list = ['MWD Continuous Inclination dega'])



with open('sens1.csv','w') as f:
    w = csv.writer(f)
    w.writerow(senstable.keys())
    w.writerow(senstable.values())


truth, pred, columns, score, senstable = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = ['MWD Continuous Inclination dega'],
                                  plot_samples = False,
                                  lcs_list = [])

with open('sens2.csv','w') as f:
    w = csv.writer(f)
    w.writerow(senstable.keys())
    w.writerow(senstable.values())
    
    
#%%
truth, pred, columns, score, senstable = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  plot_samples = False,
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  hAttrCount=5)



with open('PCA.csv','w') as f:
    w = csv.writer(f)
    w.writerow(senstable.keys())
    w.writerow(senstable.values())


truth, pred, columns, score, senstable = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  plot_samples = False,
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  asel_choice='ppscore',
                                  hAttrCount=5)

with open('ppscore.csv','w') as f:
    w = csv.writer(f)
    w.writerow(senstable.keys())
    w.writerow(senstable.values())
    
truth, pred, columns, score, senstable = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  plot_samples = False,
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  asel_choice='pearson',
                                  hAttrCount=5)

with open('pearson.csv','w') as f:
    w = csv.writer(f)
    w.writerow(senstable.keys())
    w.writerow(senstable.values())
#%%



truth, pred, columns, score, senstable = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  hstep_extension = 10)
#%%
with open('senstest.csv','w') as f:
    w = csv.writer(f)
    w.writerow(senstable.keys())
    w.writerow(senstable.values())
#%%

plt.bar(range(len(senstable)), senstable.values())
plt.xticks(range(len(senstable)),senstable.keys(), rotation=90)

#%%

truth, pred, columns, score, senstable  = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'Rate of Penetration m/h',
                                  convert_to_diff = [],
                                  lcs_list = [])

#%%
truth, pred, columns, score, senstable = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  hAttrCount=6)

#%%


drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
             'nameWellbore', 'name','RGX_RT unitless',
                       'MWD Continuous Azimuth dega']

truth, pred, columns, score, senstable = tape(data, split=1,
                                 drops=drops,
                                 index = 'Measured Depth m',
                                 target =  'MWD Continuous Inclination dega',
                                 asel_choice = 'ppscore',
                                 hAttrCount=6,
                                 convert_to_diff = [],
                                 lcs_list = ['MWD Continuous Inclination dega'])


drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
             'nameWellbore', 'name','RGX_RT unitless',
                       'MWD Continuous Azimuth dega']

truth, pred, columns, score, senstable = tape(data, split=1,
                                 drops=drops,
                                 index = 'Measured Depth m',
                                 target =  'MWD Continuous Inclination dega',
                                 asel_choice = 'pearson',
                                 hAttrCount=6,
                                 convert_to_diff = [],
                                 lcs_list = ['MWD Continuous Inclination dega'])

#%%
# cumsum or no cumsum

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
             'nameWellbore', 'name','RGX_RT unitless',
                       'MWD Continuous Azimuth dega']

truth, pred, columns, score, senstable = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'])
#%%

truth, pred, columns, score, senstable = tape(data, split=1,
                                 drops=drops,
                                 index = 'Measured Depth m',
                                 target =  'MWD Continuous Inclination dega',
                                 convert_to_diff = ['MWD Continuous Inclination dega'],
                                 lcs_list = [],
                                 shift=0.5)

#%%


drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
             'nameWellbore', 'name','RGX_RT unitless',
                       'MWD Continuous Azimuth dega']

truth, pred, columns, score, senstable  = tape(data, split=1,hMemoryMeters=25, imagination_meters=25,
                                 drops=drops,hAttrCount=6,
                                 index = 'Measured Depth m',
                                 target =  'Corrected Hookload kkgf')

#%%

df = pd.read_csv("http://www.ux.uis.no/~atunkiel/lerwickdata.csv", delim_whitespace=True)

df['epoch'] = range(1, len(df) + 1)

truth, pred, columns, score, senstable = tape(df, split=1,hMemoryMeters=12, imagination_meters=12,
                                 drops=['yyyy'],hAttrCount=5,
                                 index = 'epoch',
                                 target =  'rain[mm]',
                                 hstep_extension = 1,
                                 clean=True)