import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

"""
Adapted from https://github.com/thuml/StochNorm
"""

__all__ = ['StochNorm2d', 'StochNorm1d']


class _StochNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_StochNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.p = 0.5  # 0.0
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        if self.training:
            z_0 = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                False, self.momentum, self.eps)

            z_1 = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                True, self.momentum, self.eps)

            if input.dim() == 2:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1,
                                                                                      self.num_features)).float().cuda()
            elif input.dim() == 3:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1, self.num_features,
                                                                                      1)).float().cuda()
            elif input.dim() == 4:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1, self.num_features, 1,
                                                                                      1)).float().cuda()
            else:
                raise BaseException()

            z = (1 - s) * z_0 + s * z_1
        else:
            z = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                False, self.momentum, self.eps)

        return z


class StochNorm2d(_StochNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class StochNorm1d(_StochNorm):
    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))