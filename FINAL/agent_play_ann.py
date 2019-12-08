import os
import sys
import argparse

import lib
from lib.utils import *
from lib.trainer import ANNTrainer

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type = str, help = 'Training config you want to use for training.')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = init_parser()
    trainer = ANNTrainer(opt.config)
    trainer.play()

