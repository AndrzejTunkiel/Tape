# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:02:50 2021

@author: lloth
"""
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import pandas as pd

from scipy.optimize import least_squares

def fun_1(a, x, y): #new function that anchors gap ends at no error, just one parameter
    return a[0]*(x**2-1)-y

#forin is data WITHOUT the gaps, ie all gaps removed.
def pred3(forin, gapsize): #main predicting function

    steps = 2
    reg = LinearRegression()

    
    #steps = 5

    r2_array = []
    for f in range(1,steps+1): #iterate over steps (we are at f-step)
        local_score_array = []
        local_r2_array = []
        for s in range(0,f+1): #data multiplication by shifting (we are at s-shift)
            forin_loc = forin.copy()
            forin_loc = forin_loc[s:] # more results from one gap length
            for g in range(f):
                forin_loc[g+1::f+1] = np.nan
                
            while True:
                if np.isnan(forin_loc[-1]):
                    forin_loc = forin_loc[:-1]
                else:
                    break
    
            mask_nan = np.argwhere(np.isnan(forin_loc)).ravel()
            
            row_filled = pd.DataFrame(forin_loc).interpolate().to_numpy().ravel()
            
            local_score_array.append(r2_score(forin[s:][mask_nan], row_filled[mask_nan]))
            
            # in-gap correcitons
            
            y_test = row_filled[mask_nan]
            
            y = forin[s:][mask_nan]
            
            local_r2 = []
            for i in range(f):
                local_r2.append(r2_score(y[i::f], y_test[i::f]))
            
            local_r2_array.append(local_r2)

        if np.asarray(local_r2_array).ndim > 1:
            local_r2_array = np.average(local_r2_array, axis=0)

        ## calculate the curves
        a0 = np.ones(1) #initiating the parameters with ones

        x_fit = np.linspace(-1,1, len(local_r2_array) + 2)
        y_fit = 1- np.hstack([1, np.asarray(local_r2_array), 1])
        model = least_squares(fun_1, a0, args=(x_fit,y_fit),loss='soft_l1') #the soft_l1 is the more lenient loss function
        a = model.x[0]
        
        r2_array.append(1+a)
        
        if np.average(local_r2_array) < 0.99:
            pass
        
    #x = np.arange(1,steps+1,1).reshape(-1,1)
    x = np.arange(1,(1+steps)*2,2)[1:].reshape(-1,1)
    reg.fit(x,r2_array)
    
    predpoint = 1
    

    a0 = np.ones(1)
    x_fit = np.asarray([-1,0,1])
    y_fit = np.asarray([0, 1-(reg.predict([[predpoint]]))[0], 0])
    model = least_squares(fun_1, a0, args=(x_fit,y_fit),loss='soft_l1') #the soft_l1 is the more lenient loss function
    a = model.x[0]
    
    
    x_results = np.linspace(-1,1,gapsize+2)[1:-1]
    y_results = a*(x_results**2 - 1)
    
    return 1 - np.average(y_results)