import os
import sys

from lib.utils import *
from lib.plot import TrainingPlot


def sys_check(length):
    if length < 3:
        print('Usage:')
        print('python3 plot_cnn_history.py [model1] [model2] ... [plot selection]')
        exit()

    return None

if __name__ == '__main__':
    sys_check(len(sys.argv))
    plotter = TrainingPlot(sys.argv[1: -1])
    plotter.plot(sys.argv[-1])


