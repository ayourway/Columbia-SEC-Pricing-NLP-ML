#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:36:27 2018

@author: Jun Guo

Modified from code originally written by 
Yue Chang (ulysses.cy@gmail.com)
Updated to comply with furthur deployments.

"""
import pandas as pd
import numpy as np

"""
Hand-selected word sentiment categories from Harvard Inquirer Dictionary
that are related to the financial report.
"""
Harvard_Selected = ['Positiv', 'Negativ', 'Pstv', 'Ngtv', 'Strong', 'Weak', 'Active',
                    'Passive', 'EMOT', 'Virtue', 'Vice', 'Ovrst', 'Undrst', 'Role',
                    'Need', 'Goal', 'Try', 'Means', 'Persist', 'Complet', 'Fail',
                    'Think', 'Know', 'Causal', 'Ought', 'Perceiv', 'Compare', 'EVAL',
                    'Solve', 'ABS']
"""
Categories by Loughran and McDonald
"""

LM_Selected = ['Negative', 'Positive', 'Uncertainty', 'Litigious',
               'Constraining', 'Superfluous', 'Interesting', 'Modal']
# Read LM Dictionary
LM_Dict = pd.read_excel('../data/dict/LoughranMcDonald_MasterDictionary_2014.xlsx')
# Read HID Dictionary
HV_Dict = pd.read_excel('../data/dict/inquireraugmented.xls')

"""
The word list contains the word "FALSE" which might be automatically
converted to boolean; forcing the original string format.
Construct the first index column
"""
"""
Read data; 
convert words in original dataset into boolean values for easier
calculation and manipulation
"""
HVred = HV_Dict.loc[2:, ['Entry']].astype(str)
HVred[Harvard_Selected] = pd.notnull(HV_Dict.loc[2:, Harvard_Selected])

LMred = LM_Dict.loc[2:, ['Word']].astype(str)
LMred[LM_Selected] = np.sign(LM_Dict.loc[2:, LM_Selected])

#For same word with multiple tenses, simply use the averaged value
HVred['Entry'] = HVred['Entry'].replace(r'#.*', '', regex=True)
HVred = HVred.groupby(['Entry'], as_index=False)[Harvard_Selected].mean()

HVred.to_csv('../data/dict/'+'HV.csv')
LMred.to_csv('../data/dict/'+'LM.csv')


