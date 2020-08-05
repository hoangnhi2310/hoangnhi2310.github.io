# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:49:59 2020

@author: Hoangnhi
"""
#%% import packages
from pymongo import MongoClient
import pandas as pd
import numpy as np
import logging
from vncorenlp import VnCoreNLP
#%% post dataframe
client = MongoClient()
db_exp = client['test-api-experience']
data_post = pd.DataFrame(list(db_exp.post.find(projection={"_id": 1, 'when': 1, 'onid': 1, 'text':1})))
data_post['text'].isna().sum()
data_post.dropna(subset=['text'], inplace=True)
data_post = data_post.reset_index(drop=True)
data_post['text'] = data_post['text'].apply(lambda x: x[:70000])
# preprocessing
data_post['text'] = data_post['text'].str.replace(r'http?://\S+', '')    # xoa cac hyperlink
data_post['text'] = data_post['text'].str.replace(r'<.*?>', '')    # xoa cac the web
# xoa cac ky tu dac biet
data_post['text'] = data_post['text'].str.replace(r"[\^\^\[\]&*:;,?\\~\"\'\"{}\(\)\-$@#]+","")
data_post['text'] = data_post['text'].str.replace(r"[(\U0001F600-\U0001F92F|\U0001F300-\U0001F5FF|\U0001F680-\U0001F6FF|\U0001F190-\U0001F1FF|\U00002702-\U000027B0|\U0001F926-\U0001FA9F|\u200d|\u2640-\u2642|\u2600-\u2B55|\u23cf|\u23e9|\u231a|\ufe0f)]+","")
#%% search dataframe 
data_search = pd.read_csv(r'D:\Hahalolo\projects\sd_train\training\ai.hungvy\DataR8.csv')
data_search.drop(['Unnamed: 0'], axis = 1, inplace=True)
data_search['keys'] = data_search['keys'].str.replace(r'+', ' ')
#%% use vietnamese NLP library
vncorenlp_file = r'./VnCoreNLP-1.1.1.jar'
vncorenlp = VnCoreNLP(vncorenlp_file)
# tokenize 
data_post['nlp'] = data_post['text'].apply(lambda i:
                                            vncorenlp.tokenize(i))
data_search['vnnlp'] = data_search['keys'].apply(lambda x:
                                               vncorenlp.tokenize(x))
#%% convert to list of words
from nltk import flatten
data_search['vnnlp'] = data_search['vnnlp'].apply(lambda x: flatten(x))
data_post['nlp'] = data_post['nlp'].apply(lambda i: flatten(i))
#%% frequency 
list_count = []
for search in data_search['vnnlp']:
    count = 0
    for single in search:
        for lis in data_post['nlp']:
            for ele in lis:
                if single == ele:
                    count +=1
    list_count.append(count)
data_search['count'] = list_count