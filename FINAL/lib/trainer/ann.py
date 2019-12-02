import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import *


__all__ = ['ANNBaseTrainer', 'PolicyGradientTrainer', 'QLearningTrainer', 'ActorCriticTrainer']


class ANNBaseTrainer(object):
    def __init__(self):
        pass


class PolicyGradientTrainer(ANNBaseTrainer):
    def __init__(self):
        super(PolicyGradientTrainer, self).__init__()
        pass


class QLearningTrainer(ANNBaseTrainer):
    def __init__(self):
        super(QLearningTrainer, self).__init__()
        pass


class ActorCriticTrainer(ANNBaseTrainer):
    def __init__(self):
        super(ActorCriticTrainer, self).__init__()


