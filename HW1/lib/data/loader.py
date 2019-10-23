import random
import numpy as np

from lib.utils import *
from lib.data.dataset import NengoDLDataset

__all__ = ['NengoDLDataLoader']


class NengoDLDataLoader(object):
    def __init__(self, dataset, batch_size, shuffle = True, collect_fn = None):
        if not isinstance(dataset, NengoDLDataset):
            raise TypeError('Dataset pass in NengoDLDataLoader should be a NengoDLDataset.')

        self.dataset = dataset
        self.batch_size = positive_int_check(batch_size, 'batch_size')
        self.shuffle = shuffle

        if collect_fn is None:
            self.collect_fn = self._default_collect_fn

    def load(self):
       if self.iter_index > len(self.dataset):
           raise RuntimeError('Index overflow !! Check NengoDLDataLoader.help() may be helpful.')
       elif (self.iter_index + self.batch_size) > len(self.dataset) and self.iter_index < len(self.dataset):
           indices = self.shuffled_index[self.iter_index:]
           self.iter_index += self.batch_size
       else:
           indices = self.shuffled_index[self.iter_index: (self.iter_index + self.batch_size)]
           self.iter_index += self.batch_size

       data = self._batch_list_data(indices)
       data = self.collect_fn(data)
       return data

    def batch(self):
       self.iter_index = 0
       self.shuffled_index = self._shuffle_indexing(self.shuffle)

       return self.__len__()

    def help(self):
        print('Usage:')
        print('for i in range(loader.batch()):')
        print('    data = loader.load()')
        return None

    def __len__(self):
        return (len(self.dataset) // self.batch_size) + 1

    def _shuffle_indexing(self, shuffle):
        index = [i for i in range(len(self.dataset))]
        if shuffle:
            random.shuffle(index)

        return index

    def _batch_list_data(self, indices):
        data = []
        for index in indices:
            data.append(self.dataset.getitem(index))
        
        return data

    # only support numpy now
    def _default_collect_fn(self, data_list):
        assert len(data_list[0]) > 0
        batch_data = [[] for i in range(len(data_list[0]))]
        for i in range(len(data_list)):
            for j in range(len(batch_data)):
                batch_data[j].append(data_list[i][j])

        for i in range(len(batch_data)):
            batch_data[i] = np.stack(batch_data[i], axis = 0)

        return tuple(batch_data)


