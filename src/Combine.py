#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:04:59 2017

@author: Jun
"""

import pandas as pd
import csv
import numpy as np
import re
import datetime

#Read DataFrame

compstat_temp = pd.read_csv("./data/Compstat_ThreeTwoOne_Month_Return.csv")
NLP_diff_temp = pd.read_csv("./data/NLP_cb_NormDoc_NormTic_Diff.csv")
Industry_temp = pd.read_csv("./data/Industry_3Months_Return.csv",header = None)

#Change date format; 
for i in range(len(NLP_diff_temp)):
    a = datetime.datetime.strptime(NLP_diff_temp['Date'][i],'%Y-%m-%d').strftime('%Y%m%d')
    NLP_diff_temp.loc[i,'Date'] = a

#Initialize by Adding Two Extra Columns
NLP_diff_temp["Industry_Benchmark"] = ""   
NLP_diff_temp["ThreeMonth_y"] = ""
NLP_diff_temp["TwoMonth_y"] = ""    
NLP_diff_temp["OneMonth_y"] = ""

#Find Unique Ticker in NLP result/ Company's 3month return/ Industry 3m return
Ticker_list = np.array(NLP_diff_temp.Tic.unique().astype(str))
NLP_diff_tic = np.array(NLP_diff_temp.Tic)
return_tic = np.array(compstat_temp.tic)
#return_tic = np.array(company_temp['2'])
industry_tic = np.array(Industry_temp.iloc[:,3])

for i in range(len(Ticker_list)):
    tic = Ticker_list[i]
    try:
        company_short = NLP_diff_temp.loc[np.where(NLP_diff_tic ==tic)]
        return_short = compstat_temp.loc[np.where(return_tic ==tic)]
        #return_short = company_temp.loc[np.where(return_tic ==tic)]
        industry_short = Industry_temp.loc[np.where(industry_tic == tic)]
        
        a = np.tile(np.array(company_short['Date']).T,(len(return_short),1)).astype(int)
        b = a - np.vstack(np.array(return_short['datadate']).astype(int))
        #b = a - np.vstack(np.array(return_short['1']).astype(int))

        corresponding_df = return_short.iloc[np.ma.array(b,mask=((b<0))).argmin(axis = 0)]
        NLP_diff_temp.loc[np.where(NLP_diff_tic ==tic)[0],"ThreeMonth_y"] = np.array(corresponding_df['rt3m'])
        
        NLP_diff_temp.loc[np.where(NLP_diff_tic ==tic)[0],"TwoMonth_y"] = np.array(corresponding_df['rt2m'])
        
        NLP_diff_temp.loc[np.where(NLP_diff_tic ==tic)[0],"OneMonth_y"] = np.array(corresponding_df['rt1m'])
        #NLP_diff_temp.loc[np.where(NLP_diff_tic ==tic)[0],"Dependent_Variable"] = np.array(corresponding_df.iloc[:,-3])
        
        c = np.tile(np.array(company_short['Date']).T,(len(industry_short),1)).astype(int)
        d = c - np.vstack(np.array(industry_short.iloc[:,1]).astype(int))        
        benchmark_df = industry_short.iloc[np.ma.array(d,mask=((d<0))).argmin(axis = 0)]    
        NLP_diff_temp.loc[np.where(NLP_diff_tic ==tic)[0],"Industry_Benchmark"] = np.array(benchmark_df.iloc[:,4])
        
    except ValueError:
#        print(tic)
        pass


NLP_diff_temp.to_csv("NLP_cb_NormDoc_NormTic_Diff_table.csv")