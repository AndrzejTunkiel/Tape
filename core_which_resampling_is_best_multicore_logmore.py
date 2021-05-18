#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:55:24 2021

@author: llothar
"""

from sens_tape import tape
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])

from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.ops import unary_union, polygonize
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from statistics_module import stats

def myr2(x,y,data_x, data_y):
  try:
    x1 = np.max(data_x[data_x < x])
    x2 = np.min(data_x[data_x > x])
    
    loc1 = np.where(data_x == x1)
    loc2 = np.where(data_x == x2)

    y1 = data_y[loc1][-1]
    y2 = data_y[loc2][0]



    m = (y1-y2)/(x1-x2)
    b = (x1*y2 - x2*y1)/(x1-x2)

    
    y_inter = m * x + b

    return np.power(y-y_inter, 2)
  except:
    return 0


def minfinder(filename):
    #print('infunction')
    RNR_u = 0
    RNR_d = 0
    KNN_u = 0
    KNN_d = 0
    
    data = pd.read_csv(filename)
    index = 'Measured Depth m'
  
    

    
    global_mins = {}
    
    full_log = []
    for target in list(data):
        print(target)
        try:
            
            df = data
            s, m, per = stats(df)
            
            #%%
            
            ## Gap statistics for target
            #
            # This chart will show the percentage of dataset occupied by gaps of a certain
            # size. Gaps are normal in drilling logs and nothing to be afraid of
            
            x_label = per[target]['gap_sizes']
            x = np.arange(0, len(x_label),1)
            
            y = per[target]['percentage_cells_occupied'] 
            # plt.xticks(x, x_label, rotation=90)
            # plt.bar(x,y)
            # plt.title(f'Gap distribution in:\n {target}')
            # plt.xlabel('Gap length')
            # plt.ylabel('Percentage of dataset occupied')
            
            x_labels = x.tolist()
            x_labels[0] = 'data'
            # plt.xticks(x, x_labels)
            # plt.grid()
            # plt.show() 
            
            #%%
            
            ## Outlier detection
            
            outlier_cutoff = 0.005 #arbitrarily selected
            
            # calculation that penalizes long, rare, continuous gaps
            out_coef = per[target]['gap_sizes'] / (per[target]['gap_counts'] * len(df))
            
            x = np.arange(0,len(per[target]['gap_sizes']),1)
            x_label = per[target]['gap_sizes']
            x = np.arange(0, len(x_label),1)
            
            # plt.xticks(x, x_label, rotation=90)
            # plt.bar(x,out_coef)
            # #plt.ylim(0,0.005)
            # plt.plot(x,[outlier_cutoff]*len(x), color='red', label='cutoff')
            # plt.legend()
            x_labels = x.tolist()
            x_labels[0] = 'data'
            # plt.xticks(x, x_labels)
            # plt.title(f'Gap occupancy = gap size / (relative gap quantity) \n in:{target}')
            # plt.xlabel('Gap length')
            # plt.ylabel('Gap occupancy')
            # plt.grid()
            # plt.show() 
            
            #%%
            
            ## Automatic proposal of useful area, part 1
            
            # find the smallest outlier gap
            # outlier coefficients - True when above outlier cutoff
            cutoff_map = (out_coef >= outlier_cutoff)
            
            # Using map created above, what is the smallest outlier?
            lower_cutoff = np.min(np.asarray(per[target]['gap_sizes'])[cutoff_map])
            
            # This is done to quickly mark gaps bigger than cutoff with zero and
            # other with NaN. This makes a good chart.
            from functools import partial
            
            def _cutoff(x, lower_cutoff=0):
                if x >= lower_cutoff:
                    return 0
                else:
                    return np.nan
            
            _cutoff_par = partial(_cutoff, lower_cutoff=lower_cutoff)
            mapped_outliers = list(map(_cutoff_par, m[target]))
            
            # plt.scatter(df[index], df[target], s=1, c='black', label='data')
            # plt.plot(df[index], mapped_outliers, c='red', label='Unusuable range') 
            #                                                 # has to be plot to avoid index
            #                                                 # discontinuities
                                                            
            # plt.grid()
            # plt.legend()
            # plt.title('Useful range analysis')
            # plt.xlabel(f'{index}')
            # plt.ylabel(f'{target}')
            # plt.show()
            #%%
            
            ## Automatic proposal of useful area, part 2
            
            # Simply finds the biggest area with acceptable gaps
            
            # TODO - check if the algorithm will detect a stride at the end of the dataset
            #        because I have a feeling it won't!
            
            strides = []
            
            s_start = -1
            s_stop = -1
            
            for i in range(len(df)):
                if mapped_outliers[i] != 0 and s_start == -1:
                    s_start = i
                elif mapped_outliers[i] == 0 and s_start != -1:
                    s_stop = i
                    strides.append([s_start, s_stop, s_stop - s_start ])
                    
                    s_start = -1
                    s_stop = -1
            
            strides = np.asarray(strides)
            strides = strides[strides[:,2].argsort()][::-1] # sort by length [2] 
                                                            # and reverse
            # print(f'''Proposed range to use is row {strides[0,0]} to row {strides[0,1]}
            # for total of {strides[0,0]} rows
            #       ''')
            # print(f'All found strides are: [start, stop, length]')
            # print(strides)
            
            s_start = strides[0,0]
            s_stop = strides[0,1]

            
            ## cut the dataframe for the selected stride, redo the stats
            #  From now on dfs is used (dataframe stride)
            margin_percent = 1  # since the edges can be a bit unpredictable, margin is
                                # removed
            
            s_start = s_start + int((s_stop - s_start) * 0.01*margin_percent)
            s_stop = s_stop - int((s_stop - s_start) * 0.01*margin_percent)
            dfs = df.iloc[s_start:s_stop] # dfs = DataFrameStride
            
            
            index_dr = np.diff(dfs[index])
    
            index_mean = np.mean(index_dr)
            index_std = np.std(index_dr)
            index_maxgap = np.max(index_dr)
            h = 5
            data_x = np.arange(np.min(dfs[index].to_numpy()),
                          np.max(dfs[index].to_numpy()),
                          index_maxgap*h)
            
            

            
            samples = np.arange(1,11,1)
            weightss = ['uniform', 'distance']
            local_mins = {}
            local_mins_n = {}
            for weights in weightss:
                areas = []
                
                for i in samples:
                    reg = RadiusNeighborsRegressor(radius=index_maxgap*i, weights=weights)
                    raw = dfs[target].interpolate().ffill().bfill().to_numpy()
                    reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
                    data_y = reg.predict(data_x.reshape(-1,1))

                    totals = []
                    for row in np.rot90([data_x,data_y]):
                        x = row[0]
                        y = row[1]
                        totals.append(myr2(x,y,dfs[index].to_numpy(), raw))
                
                
                    Area_poly = (np.sum(totals))
                        
                    areas.append(Area_poly)
                local_mins[f'RNR {weights}']  = np.min(areas)
                local_mins_n[f'RNR {weights}']  = samples[np.argmin(areas)]
                # plt.plot(samples,areas, label=f'RNR, {weights}')
            
            ks = np.arange(1,11,1)
            for weights in weightss:
                areas = []
                
                for i in ks:
                    reg = KNeighborsRegressor(n_neighbors=i, weights=weights)
                    raw = dfs[target].interpolate().ffill().bfill().to_numpy()
                    reg.fit(dfs[index].to_numpy().reshape(-1,1),raw)
                    data_y = reg.predict(data_x.reshape(-1,1))
                    
                    totals = []
                    for row in np.rot90([data_x,data_y]):
                        x = row[0]
                        y = row[1]
                        totals.append(myr2(x,y,dfs[index].to_numpy(), raw))
                
                
                    Area_poly = (np.sum(totals))
                    areas.append(Area_poly)
                local_mins[f'KNN {weights}'] = np.min(areas)
                local_mins_n[f'KNN {weights}']  = ks[np.argmin(areas)]
            #     plt.plot(ks,areas, label=f'KNN, {weights}')
            # plt.legend()
            # plt.title(target)
            # plt.grid()
            # plt.show()
            # global_mins[target] = min(local_mins, key=local_mins.get)

            loopmin = (min(local_mins, key=local_mins.get))
            
            loopminval = local_mins_n[loopmin]
            
            
            full_log.append([loopmin, loopminval, target])
            
            
            if loopmin == "KNN uniform":
                KNN_u += 1
            elif loopmin == "KNN distance":
                KNN_d += 1
            elif loopmin == "RNR uniform":
                RNR_u += 1
            elif loopmin == 'RNR distance':
                RNR_d += 1
            else:
                print("ERROR!!!")
            
            
        except:
            pass
    
    try:
        full_log = np.vstack([full_log, np.load('full_log.npy', allow_pickle=True)])

        np.save('full_log.npy', full_log, allow_pickle=True)
    except:
        np.save(f'full_log{filename}.npy', full_log, allow_pickle=True)
        
    return np.asarray([KNN_u, KNN_d, RNR_u, RNR_d])

totals = np.asarray([0,0,0,0])
print('start')
import glob
filelist = (glob.glob("*.csv"))

#for filename in filelist:
#   print(filename)
#   #data = pd.read_csv(filename)
#   
#   results = minfinder(filename)
#   totals = totals + results
#   np.save('resamplin_new_algo_basic.npy', totals, allow_pickle=True)
#   print(totals)



from multiprocessing import Pool as ThreadPool
pool = ThreadPool(20)
results = pool.map(minfinder, filelist)

np.save('resultfile_multi_algo_end.npy', totals, allow_pickle=True)
