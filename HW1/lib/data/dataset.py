import torch
import torchvision
import torchvision.transforms as tfs

from torch.utils.data import Dataset
from PIL import Image

from lib.utils import *
from lib.data.mnist import MNIST


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


