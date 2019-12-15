import lib
from lib.utils import *
from abc import ABC, abstractmethod


__all__ = ['Agent']


class Agent(ABC):
    def __init__(self, name, env_name):
        self.name = name
        self.env_name = env_name

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def make_action(self):
        pass


