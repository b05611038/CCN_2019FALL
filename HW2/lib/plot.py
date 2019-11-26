import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lib
from lib.utils import *


__all__ = ['TrainingPlot']


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

