import os
import sys
import argparse

from lib.utils import *
from lib.trainer import TorchTrainer

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type = str, help = 'Training config you want to use fro training.')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = init_parser()
    trainer = TorchTrainer(opt)
    trainer.train()


