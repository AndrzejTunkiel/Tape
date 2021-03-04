#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:02:13 2021

@author: llothar
"""
import functools
from sens_tape import tape

def twd(splits, *config):
    for split in splits:
        tape(config, split=split)
    
  
    
  
    
# def a(b=0,c=0):
#     print(f'{b=}')
#     print(f'{c=}')
    
    
# a_mod = functools.partial(a, b=2)
