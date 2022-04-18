#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:50:21 2022

@author: yuezhao
"""

import os

import pandas as pd

DATA_PATH = '../data/'

df = pd.read_csv(DATA_PATH+'chemical_annotations.csv')
#%%
for mol in df['CPD_SMILES']:
    if type(mol) != str:
        print (mol)