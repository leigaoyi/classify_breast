# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:23:27 2020

@author: gaoyilei
"""


# -*- coding: utf-8 -*-
# Copyright (c) 2020, Tencent Inc. All rights reserved.
# Author: huye
# Date: 2020-03-04

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossV2(nn.CrossEntropyLoss):
    """Cross-entropy loss with label smoothing.
    label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
    meaning the confidence on label values are relaxed.
    e.g. label_smoothing=0.2 means that we will use a value of 0.1
    for label 0 and 0.9 for label 1"""

    def __init__(self, label_smoothing=0.1, weight=None, size_average=None,
            ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLossV2, self).__init__(weight, size_average,
                ignore_index, reduce, reduction)
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        num_classes = input.size(1)
        label_neg = self.label_smoothing / num_classes
        label_pos = 1. - label_neg * (num_classes - 1)
        with torch.no_grad():
            ignore = target == self.ignore_index
            n_valid = (ignore == 0).sum()
            onehot_label = torch.empty_like(input).fill_(label_neg).scatter_(
                    1, target.unsqueeze(1), label_pos)

        loss = F.log_softmax(input, dim=1)
        loss = -torch.sum(loss * onehot_label, dim=1) * (1. - ignore.float())
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CrossEntropyLossV3(nn.CrossEntropyLoss):
    """Cross-entropy loss with label smoothing.
    label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
    meaning the confidence on label values are relaxed.
    e.g. label_smoothing=0.2 means that we will use a value of 0.1
    for label 0 and 0.9 for label 1"""

    def __init__(self, label_smoothing=0.1, weight=None, size_average=None,
            ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLossV2, self).__init__(weight, size_average,
                ignore_index, reduce, reduction)
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        loss = F.log_softmax(input, dim=1)
        loss = -torch.sum(loss * target, dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

