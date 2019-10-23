import os
import torch
import nengo
import numpy as np
import tensorflow as tf 

import lib.model

from lib.utils import *
from lib.model.cnn import CNN


class Container(object):
    # class to intergrate pytorch model and nengo model for classification problems
    def __init__(self, model_path, model_type):
        if not os.path.isfile(model_path):
            raise RuntimeError('Can not detect model object, please check your file path.')

        self.model_path = model_path
        if model_type not in ['torch', 'nengo']:
            raise ValueError('Container object only support for pytorch model or nengo model.')

        self.model_type = model_type
        self.model = self._load_model(model_path, model_type)

    def inference(self, image):
        if self.model_type == 'torch':
            pass
        elif self.model_type == 'nengo':
            pass

    def _load_model(self, path, model_type):
        state = load_pickle_obj(path)
        if model_type == 'torch':
            model = CNN(state['args']['num_classes'], state['args']['num_layers'],
                    state['args']['num_filters'], state['args']['kernel_sizes'])
            model.load_state_dict(state['weight'])
        elif model_type == 'nengo':
            pass

        return model


