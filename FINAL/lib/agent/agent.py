import random
import numpy as np

import torch
import torch.cuda as cuda
from torch.distributions import Categorical

from lib.utils import *
from lib.agent.base import Agent
from lib.agent.preprocess import Transform
from lib.agent.ann import ANN
from lib.agent.snn import SNN


__all__ = ['PongAgent']


class PongAgent(Agent):
    def __init__(self):
        super(PongAgent, self).__init__()
        pass


