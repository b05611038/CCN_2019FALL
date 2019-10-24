import os
import nengo
import numpy as np
import tensorflow as tf 

import torch
import torchvision
import torch.cuda as cuda
import torchvision.transforms as tfs 

from PIL import Image

import lib.model

from lib.utils import *
from lib.model.cnn import CNN
from lib.model.snn import SNN


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

    def inference(self, image, transforms = None):
        if self.model_type == 'torch':
            #if transforms is None:
            #    transform = tfs.Compose([tfs.ToTensor()])
            pass    
            
        elif self.model_type == 'nengo':
            pass

    def _load_model(self, save_dir, model_type):
        state = load_pickle_obj(os.path.join(save_dir, save_dir + '.pkl'))
        if model_type == 'torch':
            model = CNN(
                    state['args']['num_classes'],
                    state['args']['num_layers'],
                    state['args']['num_filters'],
                    state['args']['kernel_sizes']
                    )
            model.load_state_dict(state['weight'])
            self._init_torch_device()
            model = model.to(self.device)
        elif model_type == 'nengo':
            model = SNN(
                    state['num_classes'],
                    state['num_layers'],
                    state['num_filters'],
                    state['kernel_sizes'],
                    n_steps = state['n_steps'],
                    minibatch_size = state['minibatch_size']
                    )

            model.sim.load_params(os.path.join(save_dir, save_dir))

        return model

    def _init_torch_device(self):
        # select training environment and device
        print('Init training device and environment ...')
        if cuda.is_available():
            torch.backends.cudnn.benchmark = True
            cuda.set_device(0)
            training_device = torch.device('cuda:' + str(0))
            print('Envirnment setting done, using device: cuda:' + str(0))
        else:
            training_device = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')

        return training_device

