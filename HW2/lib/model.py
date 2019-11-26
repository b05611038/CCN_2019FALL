from typing import Optional, Union, Tuple, List, Sequence, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

import torchvision
from torchvision import models

import bindsnet
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
from bindsnet.network.topology import Connection, LocalConnection

from bindsnet.learning import NoOp, PostPre, WeightDependentPostPre, Hebbian, MSTDP, MSTDPET, Rmax

import lib
from lib.utils import *


__all__ = ['DiehlAndCook2015']


# reference: https://github.com/BindsNET/bindsnet/blob/master/bindsnet/models/models.py
class DiehlAndCook2015(Network):
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        learning_rule: str, 
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        """
        Constructor for class ``DiehlAndCook2015``.
        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.learning_rule = learning_rule
        rule = self._lr_selection(learning_rule)
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=rule,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")

    def _lr_selection(self, rule):
        if rule not in ['NoOp', 'PostPre', 'WeightDependentPostPre', 'Hebbian', 
                'MSTDP', 'MSTDPET', 'Rmax', 'help']:
             raise ValueError(rule, 'is not a valid learning rule. If need some selection please type help in rule selection.')

        if rule == 'help':
            print('[NoOp, PostPre, WeightDependentPostPre, Hebbian, MSTDP, MSTDPET, Rmax] is valid selections.')
            exit(0)

        if rule == 'NoOp':
            return NoOp
        elif rule == 'PostPre':
            return PostPre
        elif rule == 'WeightDependentPostPre':
            return WeightDependentPostPre
        elif rule == 'Hebbian':
            return Hebbian
        elif rule == 'MSTDP':
            return MSTDP
        elif rule == ' MSTDPET':
            return MSTDPET
        elif rule == 'Rmax':
            return Rmax


