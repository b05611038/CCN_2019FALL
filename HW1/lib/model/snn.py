import nengo
import nengo_dl

import numpy as np
import tensorflow as tf

import lib.model

from lib.utils import *


# ref: https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html
class SNN(object):
    # for classifying mnist image, the shape must be (28, 28)
    def __init__(self, num_classes, num_layers, num_filters, kernel_sizes,
                input_size = (28, 28, 1),
                minibatch_size = 1,
                n_steps = 30,
                padding_value = 0.0,
                synapse = 0.1):

        self.num_classes = positive_int_check(num_classes, 'num_classes')
        self.num_layers = positive_int_check(num_layers, 'num_layers')

        assert len(num_filters) == num_layers
        assert len(kernel_sizes) == num_layers
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        assert len(input_size) == 3
        self.input_size = input_size

        self.minibatch_size = positive_int_check(minibatch_size, 'minibatch_size')
        self.n_steps =  positive_int_check(n_steps, 'n_steps')
        self.padding_value = padding_value
        self.synapse = synapse

        self.inp, self.out_p, self.out_p_filt, net = self._build(num_classes, num_layers, num_filters, kernel_sizes)
        self.sim = nengo_dl.Simulator(net, minibatch_size = minibatch_size)

    def train_interface(self):
        return {'input': self.inp,
                'output': self.out_p,
                'output_filter': self.out_p_filt,
                'simulator': self.sim
                 }

    def load_trained(self, obj_dict):
        self.inp = obj_dict['input']
        self.out_p = obj_dict['output']
        self.out_p_filt = obj_dict['output_filter']
        self.sim = obj_dict['simulator']
        return None

    # inference part are designed for nengo_dl.Simulator, but argument minibatch_size
    # influence tf.Graph object. therefore, it is hard to do flexible inference if the
    # simulator is trained by large minibatch_size 
    def __call__(self, x):
        if x.shape[1: ] != self.input_size:
            raise RuntimeError('Size mismatch !!! Input shape must be (b, h, w, c).')

        pulse, origin_size = self._image_to_pulse(x)
        self.sim.run_steps(self.n_steps, data = pulse)
        output = self.sim.data[self.out_p_filt]

        if origin_size is not None:
            output = output[: origin_size]

        return output

    def _image_to_pulse(self, x):
        x, origin_size = self._padding_images(x, padding_value = self.padding_value)
        x = x.reshape(x.shape[0], -1)
        x = np.tile(x[:, None, :], (1, self.n_steps, 1))
        return {self.inp: x}, origin_size

    def _padding_images(self, x, padding_value = 0.0):
        if x.shape[0] == self.minibatch_size:
            return x, None
        else:
            size = tuple([self.minibatch_size] + list(self.input_size))
            new = np.full(size, padding_value)
            new[:x.shape[0]] = x
            return new, x.shape[0]

    def _build(self, num_classes, num_layers, num_filters, kernel_sizes):
        channel_each_layer = ([1] + num_filters)
        with nengo.Network() as net:
            # set some default parameters for the neurons that will make
            # the training progress more smoothly
            net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            neuron_type = nengo.LIF(amplitude = 0.01)

            # init state, make neural trainable
            nengo_dl.configure_settings(trainable = True)

            inp = nengo.Node([0] * self.input_size[0] * self.input_size[1])

            x = nengo_dl.tensor_layer(inp, tf.layers.conv2d,
                    shape_in = (self.input_size[0], self.input_size[1], channel_each_layer[0]),
                    filters = num_filters[0], kernel_size = kernel_sizes[0], padding = 'same')
            x = nengo_dl.tensor_layer(x, neuron_type)

            for i in range(1, num_layers):
                x = nengo_dl.tensor_layer(x, tf.layers.conv2d,
                        shape_in = (self.input_size[0], self.input_size[1], channel_each_layer[i]),
                        filters = num_filters[i], kernel_size = kernel_sizes[i], padding = 'same')
                x = nengo_dl.tensor_layer(x, neuron_type)

            x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d,
                    shape_in = (self.input_size[0], self.input_size[1], channel_each_layer[-1]),
                    pool_size = (self.input_size[0], self.input_size[1]),
                    strides = (self.input_size[0], self.input_size[1]))

            x = nengo_dl.tensor_layer(x, tf.layers.dense, units = num_classes)
            out_p = nengo.Probe(x)
            out_p_filt = nengo.Probe(x, synapse = self.synapse)
            return inp, out_p, out_p_filt, net


