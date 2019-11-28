import os
import sys
import numpy as np
        
import torch
import torch.cuda as cuda
import torchvision
import torchvision.transforms as tfs 
from torch.utils.data import DataLoader

    
from tqdm import tqdm
        
import bindsnet 
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels

import lib
from lib.utils import *


__all__ = ['SNNEvaluator']


# reference: https://github.com/BindsNET/bindsnet/blob/master/examples/mnist/eth_mnist.py
class SNNEvaluator(object):
    def __init__(self, directory, device_index = None):
        self.directory = directory
        self.cfg = load_pickle_obj(os.path.join(directory, 'configs.pkl'))
        self._init_seed(self.cfg['seed'], self.cfg['device'])
        self.device = self._init_device(device_index)
        self.network = self._init_network(directory)

        self.record_title = ['acc_last', 'acc_mean', 'acc_best',
                'weighted_acc_last', 'weighted_acc_mean', 'weighted_acc_best']
        self.recorder = Recorder(self.record_title)

    def eval(self, config = None, save = True):
        if config is None:
            cfg = self.cfg

        time = cfg['time']
        n_neurons = cfg['network']['n_neurons']
        dataset, n_classes, update_interval = self._init_dataset(cfg)
        print(update_interval)

        # Record spikes during the simulation
        spike_record = torch.zeros(update_interval, time, n_neurons)

        # Neuron assignments and spike proportions
        assignments = -torch.ones(n_neurons)
        proportions = torch.zeros(n_neurons, n_classes)
        rates = torch.zeros(n_neurons, n_classes)

        # Sequence of accuracy estimates
        accuracy = {"all": [], "proportion": []}

        # Set up monitors for spikes and voltages
        exc_voltage_monitor, inh_voltage_monitor, spikes, voltages = self._init_network_monitor(self.network, cfg)

        inpt_ims, inpt_axes = None, None
        spike_ims, spike_axes = None, None
        weights_im = None
        assigns_im = None
        perf_ax = None
        voltage_axes, voltage_ims = None, None

        print("\nBegin evaluation.\n")
        labels = []

        # run testing simulation
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = cfg['n_workers'])
        for step, batch in enumerate(tqdm(dataloader)):
            inputs = {'X': batch["encoded_image"].view(time, 1, 1, 28, 28)}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            labels.append(batch["label"])

            # Run the network on the input.
            self.network.run(inputs = inputs, time = time, input_time_dim = 1)
   
            # Get voltage recording.
            exc_voltages = exc_voltage_monitor.get("v")
            inh_voltages = inh_voltage_monitor.get("v")
   
            # Add to spikes recording.
            spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()

            # Reset state variables
            self.network.reset_state_variables()

        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(labels)

        all_activity_pred = all_activity(
            spikes=spike_record, assignments=assignments, n_labels=n_classes
        )
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )

        # Compute network accuracy according to available classification strategies.
        accuracy["all"].append(
            100
            * torch.sum(label_tensor.long() == all_activity_pred).item()
            / len(label_tensor)
        )
        accuracy["proportion"].append(
            100
            * torch.sum(label_tensor.long() == proportion_pred).item()
            / len(label_tensor)
        )

        print(
            "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
            % (
                accuracy["all"][-1],
                np.mean(accuracy["all"]),
                np.max(accuracy["all"]),
            )
        )
        print(
            "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
            % (
                accuracy["proportion"][-1],
                np.mean(accuracy["proportion"]),
                np.max(accuracy["proportion"]),
            )
        )
   
        self.recorder.insert((accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]),
                accuracy["proportion"][-1], np.mean(accuracy["proportion"]), np.max(accuracy["proportion"])))
   
        assignments, proportions, rates = assign_labels(
            spikes = spike_record,
            labels = label_tensor,
            n_labels = n_classes,
            rates = rates,
        )

        if save:
            self.recorder.write(self.directory, 'results')

        print("Evaluaiton complete.\n")
        return None

    def _init_network_monitor(self, network, cfg):
        exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time = cfg['time'])
        inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time = cfg['time'])
        network.add_monitor(exc_voltage_monitor, name = "exc_voltage")
        network.add_monitor(inh_voltage_monitor, name = "inh_voltage")

        spikes = {}
        for layer in set(network.layers):
            spikes[layer] = Monitor(network.layers[layer], state_vars = ["s"], time = cfg['time'])
            network.add_monitor(spikes[layer], name = "%s_spikes" % layer)

        voltages = {}
        for layer in set(network.layers) - {"X"}:
            voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time = cfg['time'])
            network.add_monitor(voltages[layer], name= "%s_voltages" % layer)

        return exc_voltage_monitor, inh_voltage_monitor, spikes, voltages

    def _init_dataset(self, cfg):
        if cfg['dataset'] not in ['MNIST']:
            raise ValueError(name, 'is not a valid dataset in SNN training.')

        if cfg['dataset'] == 'MNIST':
            dataset = MNIST(
                PoissonEncoder(time = cfg['time'], dt = cfg['dt']),
                None,
                root = os.path.join("data", "MNIST"),
                train = False,
                download = True,
                transform = tfs.Compose(
                    [tfs.ToTensor(), tfs.Lambda(lambda x: x * cfg['intensity'])]
                ),
            )

            n_classes = 10
            # total sample numbers in testing set
            update_interval = len(dataset)

        return dataset, n_classes, update_interval

    def _init_seed(self, seed, device):
        if device < 0:
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)

        return None

    def _init_network(self, directory, name = 'network.pkl'):
        network = load_pickle_obj(os.path.join(directory, name))
        return network.to(self.device)

    def _init_device(self, index):
        # select training environment and device
        print('Init training device and environment ...')
        if index is None:
            if cuda.is_available():
                torch.backends.cudnn.benchmark = True
                cuda.set_device(0)
                training_device = torch.device('cuda:' + str(0))
                print('Envirnment setting done, using device: cuda:' + str(0))
            else:
                training_device = torch.device('cpu')
                print('Envirnment setting done, using device: cpu')
        else:
            if index < 0:
                training_device = torch.device('cpu')
                print('Envirnment setting done, using device: cpu')
            else:
                torch.backends.cudnn.benchmark = True
                cuda.set_device(device)
                training_device = torch.device('cuda:' + str(device))
                print('Envirnment setting done, using device: cuda:' + str(device))

        return training_device


