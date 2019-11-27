import os
import sys
import argparse

from lib.utils import *
from lib.eval import SNNEvaluator

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_folder_name', type = str, help = 'Folder name that you named tour model')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = init_parser()
    evaluator = SNNEvaluator(opt.model_folder_name)
    evaluator.eval()


