# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:23:34 2021

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""

from sens_tape import tape
import pandas as pd

data = pd.read_csv('f9ad.csv')

drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
              'nameWellbore', 'name','RGX_RT unitless',
                        'MWD Continuous Azimuth dega']
#%%
truth, pred, columns, score = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  plot_samples = True,
                                  lcs_list = ['MWD Continuous Inclination dega'])


#%%



truth, pred, columns, score = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'],
                                  hstep_extension = 10)

#%%

truth, pred, columns, score  = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'Rate of Penetration m/h',
                                  convert_to_diff = [],
                                  lcs_list = [])

#%%
truth, pred, columns, score = tape(data, split=1,
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

truth, pred, columns, score = tape(data, split=1,
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

truth, pred, columns, score = tape(data, split=1,
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

truth, pred, columns, score = tape(data, split=1,
                                  drops=drops,
                                  index = 'Measured Depth m',
                                  target =  'MWD Continuous Inclination dega',
                                  convert_to_diff = [],
                                  lcs_list = ['MWD Continuous Inclination dega'])
#%%

truth, pred, columns, score = tape(data, split=1,
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

truth, pred, columns, score  = tape(data, split=1,hMemoryMeters=25, imagination_meters=25,
                                 drops=drops,hAttrCount=6,
                                 index = 'Measured Depth m',
                                 target =  'Corrected Hookload kkgf')

#%%

df = pd.read_csv("http://www.ux.uis.no/~atunkiel/lerwickdata.csv", delim_whitespace=True)

df['epoch'] = range(1, len(df) + 1)

truth, pred, columns, score = tape(df, split=1,hMemoryMeters=12, imagination_meters=12,
                                 drops=['yyyy'],hAttrCount=5,
                                 index = 'epoch',
                                 target =  'rain[mm]',
                                 hstep_extension = 1,
                                 clean=True)