import os
import sys
import argparse

from lib.utils import *
from lib.mnist_tester import MNISTTester

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_folder_name', type = str, help = 'Folder name that you named tour model')
    parser.add_argument('model_type', type = str, help = 'SNN type nengo, CNN type torch')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = init_parser()
    tester = MNISTTester(opt.model_folder_name, opt.model_type)
    tester.test()


