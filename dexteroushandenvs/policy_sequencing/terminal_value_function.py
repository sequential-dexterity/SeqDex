import numpy as np
import os
import torch
import random

from torch import nn

from einops import rearrange
import pickle
# # from utils.cnn_module import FeatureTunk

class RetriGraspTValue(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RetriGraspTValue, self).__init__()
      #  self.feature_tunk = FeatureTunk(pretrained=False, input_dim=input_dim, output_dim=output_dim)
    
    def forward(self, inputs):
        # 1 * 8 * 8 feat
        inputs = inputs / 255.0
        outputs = self.feature_tunk(inputs)

        return outputs
    
class GraspInsertTValue(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraspInsertTValue, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)

        self.activate_func = nn.ELU()
    
    def forward(self, inputs):
        x = self.activate_func(self.linear1(inputs))
        x = self.activate_func(self.linear2(x))
        x = self.activate_func(self.linear3(x))
        outputs = self.activate_func(self.output_layer(x))

        return outputs
    
