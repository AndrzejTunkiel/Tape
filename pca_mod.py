# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:45:52 2021

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""

def shift_pca(MLP, scaler, pca, channel, shift):
    old_shape = MLP.shape
    
    MLP = MLP.reshape((MLP.shape[0]*MLP.shape[1], MLP.shape[2]))
    MLP = scaler.inverse_transform(MLP)
    MLP = pca.inverse_transform(MLP)
    MLP[:,channel] = MLP[:,channel] + shift
    
    MLP = pca.transform(MLP)
    MLP = scaler.transform(MLP)
    MLP = MLP.reshape(old_shape)
    
    return(MLP)

def shift_notpca(MLP, channel, shift):
    MLP[:,:,channel] = MLP[:,:,channel] + shift
    return MLP