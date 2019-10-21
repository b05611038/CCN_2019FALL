import os
import csv
import numpy as np

from lib.utils import *

class MNIST(object):
    # mnist dataset downloaded from Kaggle
    # ref: https://www.kaggle.com/oddrationale/mnist-in-csv
    def __init__(self, data_path = './data', save = True):
        self.data_path = data_path
        if self._detect_his(data_path):
            self.train_images = np.load(os.path.join(data_path, 'train_images.npy'))
            self.train_labels = np.load(os.path.join(data_path, 'train_labels.npy'))
            self.test_images = np.load(os.path.join(data_path, 'test_images.npy'))
            self.test_labels = np.load(os.path.join(data_path, 'test_labels.npy'))
        else:
            self.train_images, self.train_labels = self._from_csv('mnist_train.csv')
            self.test_images, self.test_labels = self._from_csv('mnist_test.csv')

        if save:
            np.save(os.path.join(data_path, 'train_images.npy'), self.train_images)
            np.save(os.path.join(data_path, 'train_labels.npy'), self.train_labels)
            np.save(os.path.join(data_path, 'test_images.npy'), self.test_images)
            np.save(os.path.join(data_path, 'test_labels.npy'), self.test_labels)

        print('Load dataset: MNIST done !!')

    def load(self, mode = 'train'):
        if mode == 'train':
            return self.train_images, self.train_labels
        elif mode == 'test':
            return self.test_images, self.test_labels
        else:
            raise ValueError("Argument: mode in Mnist.load only support ['train',' test'].")

    def _detect_his(self, path):
        result = True
        for f in ['train_images.npy', 'train_labels.npy', 'test_images.npy', 'test_labels.npy']:
            if not os.path.isfile(os.path.join(path, f)):
                result = False
                break

        return result

    def _from_csv(self, file_name, img_size = (28, 28)):
        path = os.path.join(self.data_path, file_name)

        file = open(path, 'r')
        content = file.readlines()
        file.close()

        images, labels = [], []
        for i in range(1, len(content)):
            data = content[i].split(',')
            labels.append(int(data[0]))
            temp = list(map(int, data[1: ]))
            temp = np.array(temp, dtype = np.uint8).reshape(img_size[0], img_size[1])
            images.append(temp)

        images = np.stack(images, axis = 0)
        labels = np.array(list(map(int, labels)))

        return images, labels


