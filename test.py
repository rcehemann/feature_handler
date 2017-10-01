# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:21:20 2017

test of feature handler

@author: blenderherad
"""
#import sys
#sys.path.append('G:\\Kaggle\\feature_handler\\')
from FeatureHandler import FeatureHandler
import pandas as pd
import numpy  as np

df = pd.DataFrame([['red',12.4,'good'],['blue',11,'bad'],['green',1.4,'ugly'],['blue',3.24,'ugly'],['red',10.90,'bad']], columns=['cat', 'num', 'rank'])
dt = pd.DataFrame([['red',6,'bad'],['green',2,'ugly']], columns=['cat', 'num', 'rank'])

class_dict = {'cat': 'categorical', 'num': 'numerical', 'rank': 'ranked'}
rank_dict  = {'rank':{'ugly': 0, 'bad': 1, 'good': 2}}

#print(df)
FH = FeatureHandler(df, class_dict, rank_dict=rank_dict)
df = FH.fit_transform()
print(df)

dt = FH.transform(dt)
print(dt)

