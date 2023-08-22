#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: yuchun   time: 2020/10/19
import torch
import torch.nn as nn
from .common import *
from .layers import *

def Conv1D(num_input_channels=1, num_output_channels=1, num_hidden=[16, 32, 64, 128, 256], kernel_size = [3, 3, 3, 3, 3], skipp = 4):


    model = nn.Sequential()
    model_tmp = model

    for i in range(len(num_hidden)):
        deeper = nn.Sequential()
        skip = nn.Sequential()

        model_tmp.add(Concat2D(1, skip, deeper))

        skip.add(nn.Conv1d(in_channels=num_input_channels, out_channels=skipp, kernel_size=kernel_size[i], padding=kernel_size[i]//2))
        skip.add(nn.ReLU())
        deeper.add(nn.Conv1d(in_channels=num_input_channels, out_channels=num_hidden[i], kernel_size=kernel_size[i], padding=kernel_size[i]//2))
        #deeper.add(nn.BatchNorm1d(num_hidden[i]))
        deeper.add(nn.ReLU())

        deeper_main = nn.Sequential()

        if i != len(num_hidden) - 1:
            deeper.add(deeper_main)


        model_tmp.add(nn.Conv1d(in_channels=num_hidden[i] + skipp, out_channels=num_output_channels, kernel_size=kernel_size[i], padding=kernel_size[i]//2))
        if i != len(num_hidden) - 1:
            model_tmp.add(nn.ReLU())

        model_tmp = deeper_main
        if i != len(num_hidden) - 1:
            num_input_channels = num_hidden[i]
            num_output_channels = num_hidden[i]
    # model.add(nn.BatchNorm1d(num_output_channels))
    return model


# def Conv1D(num_input_channels=1, num_output_channels=1, num_hidden=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16],
#            kernel_size=[3, 3, 5, 5, 7, 7, 5, 5, 3, 3]):
#     model = nn.Sequential()
#     # model_tmp = model
#     model.add(nn.Conv1d(in_channels=num_input_channels, out_channels=num_hidden[0], kernel_size=kernel_size[0],
#                         padding=kernel_size[0] // 2))
#     model.add(nn.ReLU())
#     for i in range(len(num_hidden) - 1):
#         model.add(nn.Conv1d(in_channels=num_hidden[i], out_channels=num_hidden[i + 1], kernel_size=kernel_size[i + 1],
#                             padding=kernel_size[i + 1] // 2))
#         model.add(nn.ReLU())
#     model.add(nn.Conv1d(in_channels=num_hidden[-1], out_channels=num_output_channels, kernel_size=kernel_size[-1],
#                         padding=kernel_size[-1] // 2))
#
#     return model