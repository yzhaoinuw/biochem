# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 11:47:27 2022

@author: Yue
"""

import random
import torch


def calculate_rank(y_sorted, y, y_pred):

    diff = abs(y_pred-y)
    lower = y - diff 
    higher = y + diff
    
    cover = y_sorted[y_sorted<=higher]
    cover = cover[cover>=lower]
    return len(cover)/len(y_sorted)
    
if __name__ == "__main__":
    x = torch.rand(100)
    x_sorted, indices = torch.sort(x)
    ind_y = random.randint(0, 99)
    ind_pred = random.randint(0, 99)
    y = x[ind_y]
    y_pred = x[ind_pred]
    print (calculate_rank(x_sorted, y, y_pred))