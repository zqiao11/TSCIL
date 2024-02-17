# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import copy

import math
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class SingleHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(SingleHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features=in_features, out_features=self.out_features)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)

    def increase_neurons(self, n_new):
        n_old_classes = self.out_features
        self.out_features += n_new
        old_head = copy.deepcopy(self.fc)

        self.fc = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        with torch.no_grad():
            self.fc.weight[:n_old_classes] = old_head.weight
            self.fc.bias[:n_old_classes] = old_head.bias


class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def increase_neurons(self, n_new):
        n_old_classes = self.out_features
        self.out_features += n_new
        old_weight = copy.deepcopy(self.weight)

        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.reset_parameters()

        with torch.no_grad():
            self.weight.data[:n_old_classes] = old_weight.data

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(Module):
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1)
        if self.sigma is not None:
            out = self.sigma * out
        return out
