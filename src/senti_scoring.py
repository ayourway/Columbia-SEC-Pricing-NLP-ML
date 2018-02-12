# -*- coding: utf-8 -*-
"""
Created on Wed May  3 01:30:17 2017

@author: cydru
"""

import nltk
from nltk import sentiment
import os
import pandas as pd
import numpy as np
import re

# Only words of these kinds bear sentimental implications
Care_Tagset = ['JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS',
               'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
# Hand-selected word sentiment categories from Harvard Inquirer Dictionary
# that are related to the financial report.
Harvard_Selected = ['Positiv', 'Negativ', 'Pstv', 'Ngtv', 'Strong', 'Weak', 'Active',
                    'Passive', 'EMOT', 'Virtue', 'Vice', 'Ovrst', 'Undrst', 'Role',
                    'Need', 'Goal', 'Try', 'Means', 'Persist', 'Complet', 'Fail',
                    'Think', 'Know', 'Causal', 'Ought', 'Perceiv', 'Compare', 'EVAL',
                    'Solve', 'ABS']
# Categories by Loughran and McDonald
LM_Selected = ['Negative', 'Positive', 'Uncertainty', 'Litigious',
               'Constraining', 'Superfluous', 'Interesting', 'Modal']
# Read LM Dictionary
LM_MasterDict = pd.read_excel('LoughranMcDonald_MasterDictionary_2014.xlsx')
# Read HID Dictionary
HV4_INQ = pd.read_excel('inquireraugmented.xls')
# The word list contains the word "FALSE" which might be automatically
# converted to boolean; forcing the original string format.
# Construct the first index column
HVred = HV4_INQ.loc[2:, ['Entry']].astype(str)
# Read data; convert words in original dataset into boolean values for easier
# calculation and manipulation
HVred[Harvard_Selected] = pd.notnull(HV4_INQ.loc[2:, Harvard_Selected])
LMred = LM_MasterDict.loc[:, ['Word']].astype(str)
LMred[LM_Selected] = np.sign(LM_MasterDict.loc[2:, LM_Selected])
lLM = len(LM_Selected)
ltotal = len(Harvard_Selected + LM_Selected)

HVred['Entry'] = HVred['Entry'].replace(r'#.*', '', regex=True)
# For same word with multiple tenses, simply use the averaged value
HVred = HVred.groupby(['Entry'], as_index=False)[Harvard_Selected].mean()

# Word lists for faster lookup
HVdict = HVred['Entry'].tolist()
LMdict = LMred['Word'].tolist()


def read_bulk(filename):
    '''
    Get.All.Lines.
    '''
    lines = []
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            lines.append(line)
    blob = ' '.join(lines)
    return blob


def get_candidates(blob):
    '''
    Parse words and find those worth inspecting
    '''
    wnl = nltk.stem.wordnet.WordNetLemmatizer()
    wordlist = []
    # prev_not is the custom negation indicator
    # has effect only on the nearest verb/adjective
    # switches on by parsed "not", reset on effect or termination symbols
    prev_not = False
    tokens = nltk.tokenize.word_tokenize(blob)
    token_pospairs = nltk.pos_tag(tokens)
    # negated = sentiment.util.mark_negation(tokens) # not a good function
    for ind, pair in enumerate(token_pospairs):
        if pair[0] == 'not':
            prev_not = True
        elif (pair[1] == '.') or (pair[1] == ',') or (pair[1] == ':'):
            prev_not = False
        # Find the word stem using the Word Net Lemmatizer in NLTK package
        elif pair[1] in Care_Tagset:
            stemmed = wnl.lemmatize(pair[0])
            # Negation labeling
            if prev_not:
                stemmed = 'not_' + stemmed
                prev_not = False
            wordlist.append(stemmed.upper())
    return wordlist


def find_scores(docwords):
    '''
    Find a documents sentiment scores based on parsed words.
    return: 1-d numpy array
    '''
    wcount = float(len(docwords))
    cumscore = np.zeros((ltotal+1,), dtype=float)
    for word in docwords:
        # Basically a LUT process
        if 'NOT_' in word:
            word = word[4:]
            pola = -1
        else:
            pola = 1
            
        if word in LMdict:
            cumscore[0:lLM] += pola * \
                LMred.query('Word=="%s"' % word).as_matrix(LM_Selected)[0]
        if word in HVdict:
            cumscore[lLM:ltotal] += pola * HVred.query(
                'Entry=="%s"' % word).as_matrix(Harvard_Selected)[0]
    cumscore[ltotal] = wcount
    cumscore = np.array([cumscore])
    return cumscore


def extend_digits(num_digits):
    formstr = 'batch{0:0>' + str(num_digits) + '}.csv'
    for filename in os.listdir():
        if filename.startswith('batch'):
            numlist = re.findall(r'h(\d+)\.csv', filename)
            if numlist:
                newname = formstr.format(int(numlist[0]))
                os.rename(filename, newname)


def save_combiner():
    filelist = os.listdir()
    filelist.sort()
    tblist = []
    for filename in filelist:
        if 'batch' in filename:
            tblist.append(pd.read_csv(filename))
    mastertb = pd.concat(tblist, ignore_index=True)
    return mastertb


def find_invalid(tablein):
    LMarray = tablein[LM_Selected].as_matrix()
    
    # Criteria: Rows with 4 or fewer (out of 8) LM fields with scores >= 5
    threshing = np.sum(LMarray >= 5, axis=1)
    indinval = list(np.array(np.where(threshing <= 4)))
    tableout = tablein.drop(tablein.index[indinval])
    return tableout


def diff_ingroup(tablein, sorts, groups):
    tableout = tablein.sort_values(sorts, axis=0, inplace=False)
    tablegrp = tableout.groupby(groups)
    tableout[LM_Selected] = tablegrp[LM_Selected].diff()
    tableout[Harvard_Selected] = tablegrp[Harvard_Selected].diff()
    tableout.insert(2, 'Elapsed_Days', tablegrp['Date'].diff().dt.days)
    return tableout


def normalize_per_grpmedian(tablein, groups):
    tablegrp = tablein.groupby(groups)
    SelectList = LM_Selected + Harvard_Selected
    procdf = tablegrp[SelectList].transform(lambda x: x / x.median())
    tableout = tablein[['Tic','Date','Type']].join(procdf)
    tableout = tableout.replace(np.inf, np.float64(1))
    tableout = tableout.replace(np.nan, np.float64(0))
    return tableout








