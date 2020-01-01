from typing import Optional, Dict, Union, Tuple, List, Sequence, Iterable

import numpy as np

import torch
import torch.nn as nn

import bindsnet
from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes, SRM0Nodes

from .learning import WrapNoOp, WrapMSTDPET, MSTDP, Rmax


__all__ = ['SNN']


class SNN(Network):
    def __init__(self,
            img_size: tuple,
            num_actions: int,
            num_layers: int = 2,
            n_neurons: int = 256,
            learning_rule: str = 'NoOp',
            nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
            dt: float = 0.1,
            wmin: float = 0.0,
            wmax: float = 1.0,
            inpt_shape: Optional[Iterable[int]] = None,
            ) -> None:

        super(SNN, self).__init__(dt = dt)

        self.img_size = img_size
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.n_neurons = n_neurons
        self.learning_rule = learning_rule

        rule = self._lr_selection(learning_rule)
        self.traces_additive = True if self.learning_rule == 'Rmax' else False
        self.node_func = self._node(learning_rule)

        self.nu = nu
        self.dt = dt
        self.wmin = wmin
        self.wmax = wmax
        if inpt_shape is None:
            self.inpt_shape = img_size

        # layers of neural
        inpt = Input(n = np.prod(img_size), shape = self.inpt_shape, traces = True,
                traces_additive = self.traces_additive)

        middles = []
        for i in range(1, num_layers):
            middles.append(self.node_func(n = n_neurons, traces = True,
                    traces_additive = self.traces_additive))

        out = self.node_func(n = num_actions, refrac = 0, traces = True,
                traces_additive = self.traces_additive)

        # connections of layers
        connections = []
        if len(middles) > 0:
             connections.append(Connection(source = inpt, target = middles[0],
                     nu = nu, wmin = wmin, wmax = wmax, update_rule = WrapNoOp))
             for i in range(len(middles) - 1):
                 connections.append(Connection(source = middles[i], target = middles[i + 1],
                         nu = nu, wmin = wmin, wmax = wmax, update_rule = rule))

             connections.append(Connection(source = middles[-1], target = out,
                     nu = nu, wmin = wmin, wmax = wmax, update_rule = rule))
        else:
             connections.append(Connection(source = inpt, target = out,
                     nu = nu, wmin = wmin, wmax = wmax, update_rule = rule))

        # Add to network
        names = []
        self.add_layer(inpt, name = 'Input Layer')
        names.append('Input Layer')
        for i in range(len(middles)):
            self.add_layer(middles[i], name = ('Middle ' +  str(i)))
            names.append('Middle ' +  str(i))

        self.add_layer(out, name = 'Output Layer')
        names.append('Output Layer')
        for i in range(len(connections)):
            self.add_connection(connections[i], source = names[i], target = names[i + 1])

    def to(self, device):
        if self.learning_rule == 'Rmax' or self.learning_rule == 'MSTDP':
            for key in self.connections:
                self.connections[key].update_rule.set_device(device)

        self = super(SNN, self).to(device)
        return self

    def _node(self, learning_rule):
        if learning_rule == 'Rmax':
            return SRM0Nodes
        else:
            return LIFNodes

    def _lr_selection(self, rule):
        if rule not in ['NoOp', 'MSTDP', 'MSTDPET', 'Rmax', 'help']:
             raise ValueError(rule, 'is not a valid learning rule. If need some selection please type help in rule selection.')

        if rule == 'help':
            print('[NoOp, MSTDP, MSTDPET, Rmax] is valid selections.')
            exit(0)

        if rule == 'NoOp':
            return WrapNoOp
        elif rule == 'MSTDP':
            return MSTDP
        elif rule == ' MSTDPET':
            return WrapMSTDPET
        elif rule == 'Rmax':
            return Rmax


