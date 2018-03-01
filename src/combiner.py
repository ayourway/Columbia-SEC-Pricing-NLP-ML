#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:31:08 2018

@author: Jun Guo
"""
import sys, os, time
import pandas as pd
import numpy as np

demo = 'VBXAY6WOO0VIFWXZ'

def merge_backup():
    batches = [file for file in os.listdir('../data/backup') if 'batch' in file]
    batches.sort()
    bch_dir = '../data/backup/'

    entries = [] 
    for file in batches: 
        entries.append(pd.read_csv(bch_dir+ file, index_col= 0))
    mastertb = pd.concat(entries,  ignore_index=True).drop_duplicates()
    mastertb.Date = pd.to_datetime(mastertb.Date, format= '%Y%m%d')

    print("[INFO] Backup Sentiment Tables Loaded")
    return mastertb

def diff_table(table):
    """
    Return the percentile difference between each quarters
    """
    sent = table.keys()[3:]
    nt = table.sort_values(by = ['Tic', 'Date'])

    test = nt.copy()
    test[sent] = nt.groupby('Tic')[sent].pct_change()

    # Remove first row (quarter) of each company
    tidy = test[test.groupby('Tic')['Tic'].transform(mask_first).astype(bool)]

    # Todo replace NaN by 0 and inf by a selected large number
    tidy = tidy.replace(np.nan, 0).replace(np.inf, 10000).replace(-np.inf, -10000)
    return tidy


def mask_first(x):
    """
    https://stackoverflow.com/a/31226290/7836408
    """
    result = np.ones_like(x)
    result[0] = 0
    return result

def load_price(tic):
    """
    Return quarterly stock return of the given company 
    using alpha vantage database
    """
    
    alpha_vantage = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol='\
                 + tic +'&apikey='+ demo+ '&datatype=csv'
    
    raw_price = pd.read_csv(alpha_vantage)
    raw_price['timestamps'] = pd.to_datetime(raw_price.iloc[:, 0], format = '%Y-%m-%d')
    qt_return = raw_price.sort_values(by = ['timestamps'])['adjusted close'].pct_change(periods = 2)
    
    return_tb = pd.DataFrame(data= qt_return.values, columns= ['Quarterly_return'], index= raw_price['timestamps'][::-1]).dropna()
    
    return return_tb

def attach_label(table):
    Tickers = table.Tic.unique()
    benchmark = load_price('SPX')
    #Get price
    matrix = []
    for i, tic in enumerate(Tickers):
        j = (i + 1)/ len(Tickers)
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %.2f%% Stock Price Downloaded" % ('='* int(j * 50), j*100))
        sys.stdout.flush()
        
        try:
            tb = load_price(tic)
            tb['result'] = np.where(benchmark.reindex(tb.index, method = 'nearest') < tb, 1, 0)
            temp = table[table.Tic == tic].set_index('Date')
            
            y = tb.drop('Quarterly_return', axis = 1)
            matrix.append(pd.concat([temp, y.reindex(temp.index, method= 'nearest')], axis = 1))
            time.sleep(2)
        except ValueError:
            print("Download Error")
    combined = pd.concat(matrix)
    return combined

if __name__ == '__main__':
    sent_table = merge_backup()
    
#    original = attach_label(sent_table)
#    original.to_csv('../data/original_matrix.csv')
    tidy = diff_table(sent_table)
    
    combined = attach_label(tidy)
    combined.to_csv('../data/combined_matrix.csv')