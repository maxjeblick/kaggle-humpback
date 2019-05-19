from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def cross_entropy(**_):
    return torch.nn.CrossEntropyLoss()


def binary_cross_entropy(**_):
    return torch.nn.BCEWithLogitsLoss()


def mse_loss(**_):
    return torch.nn.MSELoss()


def l1_loss(**_):
    return torch.nn.L1Loss()


def smooth_l1_loss(**_):
    return torch.nn.SmoothL1Loss()


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)
