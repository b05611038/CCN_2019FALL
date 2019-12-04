import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *


__all__ = ['ANN']


class ANN(nn.Module):
    def __init__(self, img_size, num_actions, num_layers = 2, hidden_size = 256,
            dropout = 0.1, critic = False):
        super(ANN, self).__init__()

        self.img_size = img_size
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.critic = critic

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(np.prod(img_size), hidden_size))
        self.model.add_module('drop1', nn.Dropout(p = dropout))
        self.model.add_module('relu1', nn.ReLU(inplace = True))
        for i in range(2, num_layers):
            self.model.add_module('fc' + str(i), nn.Linear(hidden_size, hidden_size))
            self.model.add_module('drop' + str(i), nn.Dropout(p = dropout))
            self.model.add_module('relu' + str(i), nn.ReLU(inplace = True))

        self.actor_layer = nn.Linear(hidden_size, num_actions)
        if self.critic:
            self.critic_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        action = self.actor_layer(x)
        if self.critic:
            score = self.critic_layer(x)
            return action, score
        else:
            return action


