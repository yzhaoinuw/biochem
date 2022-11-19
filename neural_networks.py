# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:06:04 2022

@author: Yue
"""

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size=1):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, output_size),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class DMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layer1,
        hidden_layer2,
        output_size=1,
        dropout=0,
        batchnorm=False,
    ):
        super(DMLP, self).__init__()
        l1_layer = [
            nn.Linear(input_size, hidden_layer1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        ]
        l1_bn = [nn.BatchNorm1d(hidden_layer1)]
        l2_layer = [
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        ]
        l3_layer = [nn.Linear(hidden_layer2, output_size)]

        if batchnorm:
            layers = l1_layer + l1_bn + l2_layer + l3_layer
        else:
            layers = l1_layer + l2_layer + l3_layer

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
