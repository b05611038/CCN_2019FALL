import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.utils import *


class TorchTrainingPlot(object):
    def __init__(self, names, file_name = None, save = True,
            fig_size = (10, 4)):
        self.names = names
        self.save = save
        self.fig_size = fig_size

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

        return None

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
                files.append(os.path.join(name, name))

        return files


