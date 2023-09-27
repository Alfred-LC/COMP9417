# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:04:20 2023

@author: hp
"""
import pandas as pd

# After dividing the data, we should think about how to process the data, such
# that it is more understandable for later on training.
# We adopt Chris Deotte's method to preprocess the data and 
# Reference: https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664/notebook
def feature_engineer(dataset, categorical, numerical):
    dfs = []
    for c in categorical:
        tmp = dataset.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in numerical:
        tmp = dataset.groupby(['session_id','level_group'])[c].agg('mean')
        dfs.append(tmp)
    for c in numerical:
        tmp = dataset.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)
    for c in numerical:
        tmp = dataset.groupby(['session_id','level_group'])[c].agg('skew')
        tmp.name = tmp.name + '_skew'
        dfs.append(tmp)
    dataset = pd.concat(dfs,axis=1)
    dataset = dataset.fillna(-1)
    dataset = dataset.reset_index()
    dataset = dataset.set_index('session_id')

# Data split, we divide 80% data to be train data and 20% to be the validation part in default
def dataSplit(dataset, ratio=0.20):
    session = dataset.index.unique()
    numSession = int(len(session))
    numTrain =int(numSession * (1 - ratio))
    return dataset.loc[session[:numTrain]], dataset.loc[session[numTrain:]]