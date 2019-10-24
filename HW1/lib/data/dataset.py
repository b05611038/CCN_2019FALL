import torch
import torchvision
import numpy as np
import torchvision.transforms as tfs

from torch.utils.data import Dataset
from PIL import Image

from lib.utils import *
from lib.data.mnist import MNIST


__all__ = ['NengoDLDataset', 'TorchMNIST', 'TorchInference', 'NengoMNIST']


class NengoDLDataset(object):
    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def getitem(self):
        raise NotImplementedError()


class TorchMNIST(Dataset):
    # dataset object for pytorch training
    def __init__(self, data_path = './data', mode = 'train', transform = None):
        self.data_path = data_path
        if mode not in ['train', 'test']:
            raise ValueError('Argument: mode in TorchMNIST should be train or test.')

        self.mode = mode
        dataset = MNIST(data_path, save = False)
        self.images, self.labels = dataset.load(mode)

        if transform is None:
            self.transform = tfs.Compose([tfs.ToTensor()])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.images[index]))
        label = torch.tensor(self.labels[index]).long()
        return image, label


class TorchInference(Dataset):
    def __init__(self, data, transform = None):
        self.data = data

        if transform is None:
            self.transform = tfs.Compose([tfs.ToTensor()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.data[index]))
        return image


class NengoMNIST(NengoDLDataset):
    def __init__(self, data_path = './data', mode = 'train'):
        self.data_path = data_path
        if mode not in ['train', 'test']:
            raise ValueError('Argument: mode in NengoMNIST should be train or test.')

        self.mode = mode
        dataset = MNIST(data_path, save = False)
        self.images, self.labels = dataset.load(mode)
        self.num_classes = 10

    def __len__(self):
        return self.labels.shape[0]

    def getitem(self, index):
        image = np.expand_dims(self.images[index], axis = 2)
        label = self._one_hot(self.labels[index])
        return image, label

    def _one_hot(self, label):
        one_hot = np.zeros((self.num_classes))
        one_hot[label] = 1.
        return one_hot


