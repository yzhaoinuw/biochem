#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 23:01:05 2022

@author: yuezhao
"""

import os


DATA_PATH = '../data/'
loc = DATA_PATH + 'profiles_mean_well/'
profile_path = DATA_PATH + 'profiles/'
profiles = os.listdir(profile_path)
for plate in profiles:
    if not plate.startswith('Plate'):
        continue
    plate_path = profile_path + plate + '/'
    #plate_folder = os.listdir(plate_path)
    csv_path = plate_path + 'profiles' + '/'
    csv_folder = os.listdir(csv_path)
    for file in csv_folder:
        if file.endswith('.csv'):
            os.rename(csv_path+file, loc+plate.lower()+'.csv')
    


