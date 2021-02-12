# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:38:34 2021

@author: lloth
"""

import matplotlib.pyplot as plt
import numpy as np

def overlap_plot(m, cutoff, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()
            
    m_plot = m.copy()
    m_plot_inv = m.copy()
    
    for i in range(m_plot.shape[1]):
        m_plot[:,i] = np.where(m_plot[:,i] <= cutoff, i, np.nan)
    
    for i in range(m_plot.shape[1]):
        m_plot_inv[:,i] = np.where(m_plot_inv[:,i] > cutoff, i, np.nan)
    
    m_plot = m_plot.T
    m_plot_inv = m_plot_inv.T
    
    x = np.arange(0,m_plot.shape[1],1)
    

    
    for i in range(m_plot.shape[0]):
        ax.scatter(x, m_plot[i], s=17, c='green', linewidth=1, marker=3)
    
        ax.scatter(x, m_plot_inv[i], s=17, c='red', linewidth=1, marker=3)
    return(ax)