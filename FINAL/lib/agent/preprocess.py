import math
import numpy as np

import torch
import torchvision
import torchvision.transforms as tfs

from PIL import Image


__all__ = ['Transform']


# transform dictionary format pass in the class
# {'implenmented string': True}
class Transform(object):
    def __init__(self, preprocess_dict, device):
        self.implemented_list = self.implenmented()
        self.preprocess_dict = preprocess_dict
        keys = preprocess_dict.keys()
        for key in keys:
            if key not in self.implemented_list:
                raise KeyError(key, 'is not the implemented observation preprocess method.')

        self.device = device
        self.transform = self._init_torchvision_method(preprocess_dict)

    def __call__(self, observation, memory = None):
        if self.preprocess_dict['slice_scoreboard'] == True:
            observation = self._slice_scoreboard(observation)

        observation = Image.fromarray(np.array(observation, dtype = np.uint8))
        observation = self.transform(observation)

        if self.preprocess_dict['gray_scale'] == True:
            observation = self._gray_scale(observation)

        if self.preprocess_dict['minus_observation'] == True:
            observation = self._minus_observation(observation, memory)

        if self.preprocess_dict['random_noise'] == True:
            observation = self._random_noise(observation)

        return observation

    def insert_init_memory(self, observation):
        if self.preprocess_dict['slice_scoreboard'] == True:
            observation = self._slice_scoreboard(observation)

        observation = Image.fromarray(observation)
        observation = self.transform(observation)
        if self.preprocess_dict['gray_scale'] == True:
            observation = self._gray_scale(observation)

        return observation

    def _init_torchvision_method(self, preprocess_dict):
        method = [tfs.Resize((84, 84)), tfs.ToTensor()]

        return tfs.Compose(method)

    def _random_noise(self, tensor, mean = 0., std = 0.01):
        noise = torch.normal(mean = mean, std = torch.full(size = tensor.size(),
                fill_value = std, device = tensor.device))

        tensor += noise
        return tensor

    def _gray_scale(self, tensor, r = 0.2126, g = 0.7125, b = 0.0722):
        tensor = r * tensor[0, :, :] + g * tensor[1, :, :] + b * tensor[2, :, :]
        return tensor.unsqueeze(0)

    def _minus_observation(self, observation, memory):
        if memory is None:
            raise RuntimeError("Please use agent.insert_memory() to insert initial data.")

        return observation.to(self.device) - memory

    def _slice_scoreboard(self, image):
        image = image[24:, :, :]
        return image

    def image_size(self):
        height = 84
        length = 84
        channel = 3
        if self.preprocess_dict['gray_scale'] == True:
            channel = 1
        return (height, length, channel)

    def implenmented(self):
        implemented_list = ['slice_scoreboard', 'gray_scale', 'minus_observation', 'random_noise']
        return implemented_list


