# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:28:48 2017

@author: Jun
"""

import csv 
import numpy as np
import Func as func

"""
Import data and convert into array for furthur cleaning and selection
""" 
Nasdaqraw = csv.reader(open("nasdaqcompanylist.csv"),delimiter = ",")
Amexraw = csv.reader(open("amexcompanylist.csv"),delimiter = ",")
NYSEraw = csv.reader(open("nysecompanylist.csv"),delimiter = ",")

aNasdaq = np.array(list(Nasdaqraw))
aAmex = np.array(list(Amexraw))
aNYSE = np.array(list(NYSEraw))

"""
Combine them into a huge file
"""
Nasdaq = func.AddLabel("Nasdaq",aNasdaq)
NYSE = func.AddLabel("NYSE",aNYSE)
Amex = func.AddLabel("Amex",aAmex)

Allmine = np.vstack([NYSE[1:len(NYSE)],Nasdaq[1:len(Nasdaq)],Amex[1:len(Amex)]])

"""
Select N random stocks and output two files: one with only the ticker, 
and one with detailed information
"""
Alltrim = func.Replicatecompany(Allmine)
Tic = np.vstack(Alltrim[:,1]).astype(str)
np.savetxt("Tic.csv",Tic,delimiter = ",",fmt = '%s')

N = 100
randomsample = np.random.choice(Alltrim[:,1],N,replace = False)
np.savetxt("Small sample.csv",randomsample,delimiter = ",",fmt = '%s')
