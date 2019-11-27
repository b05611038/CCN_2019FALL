import os
import csv
import numpy as np
import pandas as pd

import lib
from lib.utils import *

import matplotlib.pyplot as plt

__all__ = ['TrainingPlot']


class TrainingPlot(object):
    def __init__(self, names, file_name = None, save = True, fig_size = (10, 4)):
        self.names = names
        self.file_name = file_name
        self.save = save
        self.fig_size = fig_size

        self.file_names = self._get_filenames(self.names, file_name)
        self.history, self.epochs = self._read_history(self.file_names)

    def plot(self, select):
        if select == 'iteration-acc_last':
            self._plot(self.history, 'iters', 'acc_last',
                    self.names, 'iteration-acc_last', 'iteration-acc_last',
                    ('iteration', 'acc_last'), save = self.save) 
        elif select == 'iteration-acc_mean':
            self._plot(self.history, 'iters', 'acc_mean',
                    self.names, 'iteration-acc_mean', 'iteration-acc_mean',
                    ('iteration', 'acc_mean'), save = self.save) 
        elif select == 'iteration-acc_best':
            self._plot(self.history, 'iters', 'acc_best',
                    self.names, 'iteration-acc_best', 'iteration-acc_best',
                    ('iteration', 'acc_best'), save = self.save)
        elif select == 'iteration-weighted_acc_last':
            self._plot(self.history, 'iters', 'weighted_acc_last',
                    self.names, 'iteration-weighted_acc_last', 'iteration-weighted_acc_last',
                    ('iteration', 'weighted_acc_last'), save = self.save)   
        elif select == 'iteration-weighted_acc_mean':
            self._plot(self.history, 'iters', 'weighted_acc_mean',
                    self.names, 'iteration-weighted_acc_mean', 'iteration-weighted_acc_mean',
                    ('iteration', 'weighted_acc_mean'), save = self.save) 
        elif select == 'iteration-weighted_acc_best':
            self._plot(self.history, 'iters', 'weighted_acc_best',
                    self.names, 'iteration-weighted_acc_best', 'iteration-weighted_acc_best',
                    ('iteration', 'weighted_acc_best'), save = self.save)
        elif select == 'all':
            self.plot('iteration-acc_last')
            self.plot('iteration-acc_mean')
            self.plot('iteration-acc_best')
            self.plot('iteration-weighted_acc_last')
            self.plot('iteration-weighted_acc_mean')
            self.plot('iteration-weighted_acc_best')
        else:
            print(select, 'is not in plotting selections.')
            print('Only [iteration-acc_last,')
            print('      iteration-acc_mean,')
            print('      iteration-acc_best,')
            print('      iteration-weighted_acc_last,')
            print('      iteration-weighted_acc_mean,')
            print('      iteration-weighted_acc_best,')
            print('      all] is avaliable.')

    def _plot(self, his, axis_key, his_key, legend, save_name, title, axis_name, save = True):
        plt.figure(figsize = self.fig_size)
        for i in range(len(his)):
            plt.plot(np.array(list(his[i][axis_key])), np.array(list(his[i][his_key])))

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


