#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:22:02 2017

@author: Jun
"""

import numpy as np

def AddLabel(String, Array):
    label = np.tile(np.array(String),[len(Array),1])
    Final = np.concatenate((label,Array),1)
    return Final

def Replicatecompany(Large):
    name, indices = np.unique(Large[:,2], return_index = True)
    return Large[indices]