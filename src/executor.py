#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:11:12 2018

@author: Jun Guo

Modified from code originally written by 
Qing Ma (qm2124@columbia.edu)
Yue Chang ()
Updated to comply with furthur deployments.

"""

"""
TO-DO: 
    Scrap from the SEC
    Clean Data
    Extract MD&A
    Output Word Embedded Matrix
"""

import pandas as pd
import os, time, sys
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import shutil
import pickle

HV_dict = pd.read_csv('../data/dict/HV.csv', index_col = 0)
LM_dict = pd.read_csv('../data/dict/LM.csv', index_col = 0)
HV_sent = HV_dict.keys().tolist()[1:]
LM_sent = LM_dict.keys().tolist()[1:]
    
ltotal = len(HV_sent) + len(LM_sent)
lLM = len(LM_sent)

# Word lists for faster lookup
HV_wl = HV_dict['Entry'].tolist()
LM_wl = LM_dict['Word'].tolist()

try: 
    edgar = pickle.load(open('../data/Edgar.p', 'rb'))
except FileNotFoundError:
    edgar = pd.read_csv('../data/Edgar.csv').drop_duplicates().values
    pickle.dump(edgar, open('../data/Edgar.p', 'wb'))
    
edgar_link = 'https://www.sec.gov/Archives/'
print("[INFO] Import Loaded")


def clean_html(saved_name, file_url):
    print("[INFO] Retrieving file: {}".format(saved_name))
    try:
        request = urllib.request.Request(file_url,\
                                         headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.167 Safari/537.36'})
    
        html_file = urllib.request.urlopen(request).read()
        soup = BeautifulSoup(html_file, 'lxml')
        file = soup.find_all("document")[0].get_text()
        return file
    except IOError:
        print("[WARNING] url error")
        return None

#    temp_file = codecs.open('../data/raw/' + saved_name + '.txt', 'w', 'utf-8')
#    temp_file.write(soup)
#    temp_file.close()


def tokenizer(string):
    patterns = '[A-Za-z]+'
    
    tokenizer = RegexpTokenizer(patterns)
    tokens = tokenizer.tokenize(string.lower())
    
    sw = stopwords.words('english')
    filtered_tokens = [w for w in tokens if w not in sw and len(w) > 1]
    
    return filtered_tokens

def wordnet_pos(penn_treebank_tag):
    """
    Source: https://stackoverflow.com/a/15590384/7836408
    """
    if penn_treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_treebank_tag.startswith('V'):
        return wordnet.VERB
    elif penn_treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif penn_treebank_tag.startswith('R'):
        return wordnet.ADV


def lemma(pos_pairs):
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(t[0], wordnet_pos(t[1])).upper() for t in pos_pairs]


def pos(string_list):
    """
    Penn Treebank Tag Set
    """
    Care_Tagset = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', \
                   'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    tks_pos = pos_tag(string_list)
    return [item for item in tks_pos if item[1] in Care_Tagset]


def word_occurance(string_list):
    wd = pd.DataFrame(string_list, columns = ['name'])
    return wd.name.value_counts()


def find_scores(string_list):
    '''
    Find a documents sentiment scores based on parsed words.
    return: 1-d numpy array
    '''
    wcount = float(len(string_list))
    cumscore = np.zeros((ltotal+1,), dtype=float)

    string_df = word_occurance(string_list)
    for word, freq in string_df.items():
        if word in LM_wl:
            cumscore[0:lLM] += freq * LM_dict.query('Word=="%s"' % word).as_matrix(LM_sent)[0]

        if word in HV_wl:
            cumscore[lLM:ltotal] += freq * HV_dict.query('Entry=="%s"' % word).as_matrix(HV_sent)[0]
        
        if word not in LM_wl and word not in HV_wl:
            wcount -= freq
        """
        To do:
            use external dictionary for else condition, average of meanings.
        """

    cumscore[ltotal] = wcount
    return cumscore

def word2vec(string):
    init = time.time()
    tks = tokenizer(string)
    tks_pos = pos(tks)
    tks_stm = lemma(tks_pos)
    vec = find_scores(tks_stm)
    vec[:-1] /= vec[-1]  # Normalized Sentiment Vector
    print("========== Convertion took {:.2f}s===========".format(time.time()-init))
    
    return vec

def main(beg, end):
    attrlist = ['Tic', 'Date','Type'] + LM_sent + HV_sent + ['WCount']
    entries = []

    writeind = 0
    fileind = 0
   
    for i in range(beg, end): 
        if edgar[i][2] in ['10-Q', '10-K']:
            print(i)
            idx = [edgar[i][-1], edgar[i][1], edgar[i][2]]
            text = clean_html(edgar[i][-1] + '-' + edgar[i][2] + '-' + str(edgar[i][1]),\
                       edgar_link + edgar[i][4])
            if text:
                doc_score = word2vec(text)
                entries.append(np.concatenate((idx, doc_score)))
            
                fileind += 1
                if fileind % 10 == 0:
                    try:
                        shutil.copy2('../data/backup/autosave.csv','../data/backup/tempread.csv')
                    except:
                        pass
                    tblsave = pd.DataFrame(entries, columns=attrlist)
                    tblsave.to_csv('../data/backup/autosave.csv',\
                               sep=',',encoding='utf-8', mode='w')
                    print('[INFO] Autosaved')
                if fileind % 50 == 0:
                    tblsave.to_csv('../data/backup/batch{0:0>8}.csv'.format(i),\
                                   sep=',', encoding='utf-8', mode='w')
                    print('[INFO] Saved 50 entries to file batch%s.csv' %i)
                    tblsave = []
                    writeind += 1
                    entries = []
                
    tblsave = pd.DataFrame(entries, columns=attrlist)
    tblsave.to_csv('../data/backup/batch{0:0>8}.csv'.format(i),\
                               sep=',', encoding='utf-8', mode='w')


           
if __name__=="__main__":
    if len(sys.argv) <= 1:
        beg = 0
        end = len(edgar)
    else:
        beg = int(sys.argv[1])
        end = int(sys.argv[2])
        if int(sys.argv[2]) > len(edgar) or int(sys.argv[1]) > len(edgar):
            print('[WARNING] Maximum Input Value is {}'.format(len(edgar)))
    main(beg, end)
    os.system('say Your Program is Done')


        
