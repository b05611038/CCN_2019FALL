import math
import numpy as np

import torch
import torchvision
import torchvision.transforms as tfs

from PIL import Image


__all__ = ['Transform']


# transform dictionary format pass in the class
# {'implenmented string': True}
class Transform():

    implemented_list = ['slice_scoreboard', 'gray_scale', 'minus_observation']
    img_height = 84
    img_length = 84

    def __init__(self, preprocess_dict, device, slice_scoreboard=True, gray_scale=True, minus_observation=True):
        self._check_keys(preprocess_dict)
        self.preprocess_dict = preprocess_dict
        self.device = device
        self.transform = self._init_torchvision_method()
        self.slice_scoreboard = self._slice_scoreboard if preprocess_dict.get('slice_scoreboard') or slice_scoreboard else self._slice_scoreboard_noop
        if preprocess_dict.get('gray_scale') or gray_scale:
            self.gray_scale = self._gray_scale
            self.channel = 1
        else:
            self.gray_scale = self._gray_scale_noop
            self.channel = 3
        self.minus_observation = self._minus_observation if preprocess_dict.get(
            'minus_observation') or minus_observation else self._minus_observation_noop

    def __call__(self, observation, memory = None):
        observation = Image.fromarray(observation)
        observation = self.transform(observation)
        observation = self.slice_scoreboard(observation)
        observation = self.gray_scale(observation)
        observation = self.minus_observation(observation, memory)
        return observation

    def insert_init_memory(self, observation):
        observation = Image.fromarray(observation)
        observation = self.transform(observation)
        observation = self.slice_scoreboard(observation)
        observation = self.gray_scale(observation)
        return observation

    def _init_torchvision_method(self):
        method = [tfs.Resize((84, 84)), tfs.ToTensor()]

        return tfs.Compose(method)

    def _gray_scale(self, tensor, r = 0.2126, g = 0.7125, b = 0.0722):
        tensor = r * tensor[0, :, :] + g * tensor[1, :, :] + b * tensor[2, :, :]
        return tensor.unsqueeze(0)

    def _gray_scale_noop(self, tensor, r = 0.2126, g = 0.7125, b = 0.0722):
        return tensor

    def _minus_observation(self, observation, memory):
        if memory is None:
            raise RuntimeError("Please use agent.insert_memory() to insert initial data.")

        return observation.to(self.device) - memory

    def _minus_observation_noop(self, observation, memory):
        return observation

    def _slice_scoreboard(self, image):
        return image[24:, :, :]

    def _slice_scoreboard_noop(self, image):
        return image

    def image_size(self):
        return (self.img_height, self.img_length, self.channel)

    def _check_keys(self, preprocess_dict):
        keys = preprocess_dict.keys()
        for key in keys:
            if key not in self.implemented_list:
                raise KeyError(key, 'is not the implemented observation preprocess method.')
