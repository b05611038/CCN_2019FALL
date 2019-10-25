import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.utils import *


__all__ = ['TrainingPlot', 'TorchTrainingPlot', 'NengoTrainingPlot']


class TrainingPlot(object):
    def __init__(self, file_name, save, fig_size):
        self.file_name = file_name
        self.save = save
        self.fig_size = fig_size

    def plot(self):
        raise NotImplementedError()

    def _plot(self, x, his, key, legend, save_name, title, axis_name, save = True):
        plt.figure(figsize = self.fig_size)
        for i in range(len(his)):
            plt.plot(x, np.array(list(his[i][key])))

        plt.title(title)
        plt.xlabel(axis_name[0])
        plt.ylabel(axis_name[1])
        plt.legend(legend, loc = 'upper right')
        if save:
            plt.savefig(save_name + '.png')
            print('Picture: ' + save_name + '.png done.')
        else:
            plt.show()

        return None

    def _read_history(self, file_list):
        history = []
        longest = 0
        for i in range(len(file_list)):
            df = pd.read_csv(file_list[i])
            if len(df) > longest:
                longest = len(df)

            history.append(df)

        return history, longest

    def _get_filenames(self, name_list, file_name):
        files = []
        if file_name is None:
            for name in name_list:
                files.append(os.path.join(name, name + '.csv'))
        else:
            for name in name_list:
                files.append(os.path.join(name, self.file_name))

        return files


class TorchTrainingPlot(TrainingPlot):
    def __init__(self, names, file_name = None, save = True,
            fig_size = (10, 4)):
        super(TorchTrainingPlot, self).__init__(file_name, save, fig_size)

        self.names = names

        self.file_names = self._get_filenames(self.names, file_name)
        self.history, self.epochs = self._read_history(self.file_names)

    def plot(self, select):
        if select == 'epoch-train_loss':
            self._plot(np.arange(self.epochs + 1)[1: ], self.history, 'train_loss',
                    self.names, 'epoch-train_loss', 'epoch-train_loss',
                    ('epoch', 'train_loss'), save = self.save)
        elif select == 'epoch-test_loss':
            self._plot(np.arange(self.epochs + 1)[1: ], self.history, 'test_loss',
                    self.names, 'epoch-test_loss', 'epoch-test_loss',
                    ('epoch', 'test_loss'), save = self.save)
        elif select == 'epoch-test_acc':
            self._plot(np.arange(self.epochs + 1)[1: ], self.history, 'test_accuracy',
                    self.names, 'epoch-test_acc', 'epoch-test_acc',
                    ('epoch', 'test_acc'), save = self.save)
        elif select == 'all':
            self.plot('epoch-train_loss')
            self.plot('epoch-test_loss')
            self.plot('epoch-test_acc')
        else:
            print(select, 'is not in plotting selections.')
            print('[epoch-train_loss, epoch-test_loss, epoch-test_acc, all] is avaliable.')


class NengoTrainingPlot(TrainingPlot):
    def __init__(self, names, batch_size = 2000, file_name = None, save = True,
            fig_size = (10, 4)):
        super(NengoTrainingPlot, self).__init__(file_name, save, fig_size)

        self.names = names
        self.batch_size = batch_size

        self.file_names = self._get_filenames(self.names, file_name)
        self.history, self.epochs = self._read_history(self.file_names)
        self.history = self._devide_loss(self.history, batch_size)

    def plot(self, select):
        if select == 'step-train_loss':
            self._plot(np.arange(self.epochs + 1)[1: ], self.history, 'train_loss',
                    self.names, 'step-train_loss', 'step-train_loss',
                    ('step', 'train_loss'), save = self.save)
        elif select == 'all':
            self.plot('step-train_loss')
        else:
            print(select, 'is not in plotting selections.')
            print('[step-train_loss, all] is avaliable.')

        return None

    def _devide_loss(self, history, batch_size):
        new = []
        for obj in history:
            new.append(obj / batch_size)

        return new
