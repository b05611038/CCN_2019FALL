import os
import sys

from lib.utils import *
from lib.visualize import NengoTrainingPlot

def sys_check(length):
    if length < 3:
        print('Usage:')
        print('python3 plot_snn_history_manual.py [model1] [model2] ... [plot selection]')
        exit()

    return None

if __name__ == '__main__':
    sys_check(len(sys.argv))
    plotter = NengoTrainingPlot(sys.argv[1: -1], file_name = 'screen.csv')
    plotter.plot(sys.argv[-1])


