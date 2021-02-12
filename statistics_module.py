#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:22:06 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no
"""
import numpy as np
import pandas as pd

# Function to provide basic statistics about your data in terms of gaps

def stats(data):
    if isinstance(data, pd.DataFrame):
        dataname = list(data)
        data = data.to_numpy()
    else:
        dataname = np.arange(0, data.shape[0],1)
    mask = np.zeros(data.shape)

    
    for j in range(data.shape[1]):
        nancount = 0
        for i in range(data.shape[0]):
            if pd.isnull(data[i,j]) == 1:
                nancount = nancount + 1
                mask[i,j] = nancount
                
            else:
                for k in range(nancount):
                    mask[i-1-k,j] = nancount
                nancount = 0
        # fix if last value is nan, to propagate the mask back        
        if pd.isnull(data[i,j]) == 1:
            for k in range(nancount):
                mask[i-1-k,j] = nancount
            nancount = 0
    
                
    
    distribution = np.unique(mask, return_counts=True)
    distribution = list(distribution)
    
    total = np.sum(distribution[1])
    percentages= []
    
    for i in distribution[1]:
        percentages.append(i/total*100)
        
    distribution.append(percentages)
    
    #gap of fours has four fours, hence you need to count them and divide by four
    for i in range(1,len(distribution[0])):
        distribution[1][i] = distribution[1][i] / distribution[0][i]

    statistics = {
        'gap_sizes' : distribution[0],
        'gap_counts': distribution[1],
        'percentage_cells_occupied' : np.asarray(percentages),
        'percentage_empty': (total - distribution[1][0])/total*100
        }
    
    
    # Per sample statistics
    
    persamples = {}
    
    for j in range(data.shape[1]):
        
        distribution = np.unique(mask[:,j], return_counts=True)
        distribution = list(distribution)
        
        total = np.sum(distribution[1])
        percentages= []
        
        for i in distribution[1]:
            percentages.append(i/total*100)
            
        distribution.append(percentages)
        
        #gap of fours has four fours, hence you need to count them and divide by four
        for i in range(1,len(distribution[0])):
            distribution[1][i] = distribution[1][i] / distribution[0][i]
            distribution[1][i] = int(distribution[1][i])
            distribution[0][i] = int(distribution[0][i])
        stat_i = {
            'column' : dataname[j],
            'gap_sizes' : [int(k) for k in distribution[0]],
            'gap_counts': distribution[1],
            'percentage_cells_occupied' : np.asarray(percentages),
            'percentage_empty': (total - distribution[1][0])/total*100
            }
        
        persamples[dataname[j]] = stat_i
    
    
    mask = pd.DataFrame(data=mask, columns=dataname)
    return statistics, mask, persamples