import os
import sys
import time as T
import numpy as np

import torch
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
from lib.model import DiehlAndCook2015


__all__ = ['SNNTrainer']


# reference: https://github.com/BindsNET/bindsnet/blob/master/examples/mnist/eth_mnist.py
class SNNTrainer(object):
    def __init__(self, config):
        self.cfg = load_config(config)

        self.name = self.cfg['name']
        self.save_dir = self._init_directory(self.name)
        self.device = self._init_device(self.cfg['device'])
        self._init_seed(self.cfg['seed'], self.cfg['device'])
        self.network = self._init_network(self.cfg['network'])

        self.record_title = ['iters', 'acc_last', 'acc_mean', 'acc_best',
                'weighted_acc_last', 'weighted_acc_mean', 'weighted_acc_best']
        self.recorder = Recorder(self.record_title)

    def save(self, cfg = None):
        self.network = self.network.cpu()
        if cfg is None:
            cfg = self.cfg

        save_object(os.path.join(self.save_dir, 'network.pkl'), self.network)
        save_object(os.path.join(self.save_dir, 'configs.pkl'), cfg)
        self.network = self.network.to(self.device)
        return None

    def train(self, config = None):
        if config is None:
            cfg = self.cfg

        update_interval = cfg['update_interval']
        time = cfg['time']
        n_neurons = cfg['network']['n_neurons']
        dataset, n_classes = self._init_dataset(cfg)

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

        print("\nBegin training.\n")
        iteration = 0
        for epoch in range(cfg['epochs']):
            print("Progress: %d / %d" % (epoch, cfg['epochs']))
            labels = []
            start_time = T.time()

            dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = cfg['n_workers'])

            for step, batch in enumerate(tqdm(dataloader)):
                # Get next input sample.
                inputs = {'X': batch["encoded_image"].view(time, 1, 1, 28, 28)}
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                if step % update_interval == 0 and step > 0:
                    # Convert the array of labels into a tensor
                    label_tensor = torch.tensor(labels)

                    # Get network predictions.
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
    
                    iteration += len(label_tensor)
    
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
    
                    self.recorder.insert((iteration, accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]),
                            accuracy["proportion"][-1], np.mean(accuracy["proportion"]), np.max(accuracy["proportion"])))
    
                    assignments, proportions, rates = assign_labels(
                        spikes = spike_record,
                        labels = label_tensor,
                        n_labels = n_classes,
                        rates = rates,
                    )
    
                    labels = []

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

                if step % 1000 == 0:
                    self.save(cfg = cfg)

            print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, cfg['epochs'], T.time() - start_time))

        self.recorder.write(self.save_dir, cfg['name'])
        print("Training complete.\n")
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
                root=os.path.join("..", "..", "data", "MNIST"),
                download=True,
                transform = tfs.Compose(
                    [tfs.ToTensor(), tfs.Lambda(lambda x: x * cfg['intensity'])]
                ),
            )

            n_classes = 10

        return dataset, n_classes

    def _init_network(self, network_config):
        network = DiehlAndCook2015(**network_config)
        return network.to(self.device)

    def _init_seed(self, seed, device):
        if device < 0:
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)

        return None

    def _init_device(self, device):
        # select training environment and device
        print('Init training device and environment ...')
        if device < 0:
            training_device = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')
        else:
            torch.backends.cudnn.benchmark = True
            cuda.set_device(device)
            training_device = torch.device('cuda:' + str(device))
            print('Envirnment setting done, using device: cuda:' + str(device))

        return training_device

    def _init_directory(self, model_name):
        # information saving directory
        # save model checkpoint and episode history
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        save_dir = model_name
        print('All object (model checkpoint, trainning history, ...) would save in', save_dir)
        return save_dir


