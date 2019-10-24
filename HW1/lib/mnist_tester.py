import os
import torch
import nengo
import nengo_dl
import numpy as np
import torch.cuda as cuda

from torch.utils.data import DataLoader

from lib.utils import *

from lib.model.container import Container

from lib.data.mnist import MNIST
from lib.data.loader import NengoDLDataLoader 


class MNISTTester(object):
    def __init__(self, model_directory, model_type, mnist_path = './data'):
        self.model_directory = model_directory
        self.model_type = model_type
        self.mnist = MNIST(mnist_path)
        self.container = Container(model_directory, model_type)

    def test(self, batch_size = 2000):
        outcome, label = self._batch_inference(batch_size)
        acc = self._cal_acc(outcome, label)
        print('MNIST testing set accuracy: %.2f' % acc)
        return None

    def _cal_acc(self, output, target):
        assert output.shape[0] == target.shape[0]
        correct = (output == target).sum()
        acc = (correct / output.shape[0]) * 100
        return acc

    def _batch_inference(self, batch_size):
        data, label = self.mnist.load('test')

        if label.shape[0] % batch_size == 0:
            step = label.shape[0] // batch_size
        else:
            step = (label.shape[0] // batch_size) + 1

        outcome = []
        for i in range(step):
            batch_data = data[i * batch_size: (i + 1) * batch_size]

            if self.model_type == 'nengo':
                batch_data = np.expand_dims(batch_data, axis = 3)

            output = self.container.inference(batch_data)
            output = np.argmax(output, axis = 1)
            outcome += output.tolist()

        return np.array(outcome), label


