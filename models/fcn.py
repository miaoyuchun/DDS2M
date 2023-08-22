import torch
import torch.nn as nn
from .common import *

def fcn(num_input_channels=200, num_output_channels=1, num_hidden=[300, 500, 800, 1000, 800, 500, 300]):


    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden[0], bias=True))
    model.add(nn.ReLU6())

    for i in range(len(num_hidden)-1):
        model.add(nn.Linear(num_hidden[i], num_hidden[i+1], bias=True))
        model.add(nn.ReLU())

    model.add(nn.Linear(num_hidden[len(num_hidden)-1], num_output_channels))

    return model











