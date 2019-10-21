import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *


def _padding_cal(kernel_size):
    return kernel_size // 2


class CNN(nn.Module):
    def __init__(self, num_classes, num_layers, num_filters, kernel_sizes):
        self.num_classes = positive_int_check(num_classes, 'num_classes')
        self.num_layers = positive_int_check(num_layers, 'num_layers')

        assert len(num_filters) == num_layers
        assert len(kernel_sizes) == num_layers
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        self.conv_net = nn.Sequential()
        self.conv_net.add_modules('conv1', nn.Conv2d(1, num_filters[0], kernel_sizes[0],
                padding = _padding_cal(kernel_sizes[0])))
        self.conv_net.add_modules('relu1', nn.ReLU())
        for i in range(1, num_layers):
            self.conv_net.add_modules('conv' + str(i + 1), nn.Conv2d(num_filters[i] - 1, num_filters[i],
                    kernel_sizes[i], padding = _padding_cal(kernel_sizes[i])))
            self.conv_net.add_modules('relu' + str(i + 1), nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d(num_filters[-1])
        self.fc = nn.Linear(num_filters[-1], num_classes)

    def forward(self, x):
        x = self.conv_net(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


