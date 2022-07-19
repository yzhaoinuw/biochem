# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:06:04 2022

@author: Yue
"""

from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size=1):
        super(MLP, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, output_size),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits