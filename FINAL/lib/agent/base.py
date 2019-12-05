import lib
from lib.utils import *


__all__ = ['Agent']


class Agent(object):
    def __init__(self, name, env_name):
        self.name = name
        self.env_name = env_name

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def make_action(self):
        raise NotImplementedError()


