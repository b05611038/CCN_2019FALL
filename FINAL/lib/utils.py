import os
import sys
import json
import yaml
import pickle
import logging
import pandas as pd

import torch
import torch.cuda as cuda

import lib


__all__ = ['save_object', 'load_pickle_obj', 'load_json_obj', 'load_config',
        'init_torch_device', 'Recorder', 'default_to']


def default_to(x, d):
    return d if x is None else x


def save_pickle(fname: str, obj) -> None:
    if not fname.endswith('.pkl'):
            fname += '.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def save_json(fname: str, obj) -> None:
    if not fname.endswith('.json'):
        fname += '.json'
    with open(fname, 'w') as f:
        f.write(json.dumps(obj))


def save_object(fname: str, obj, mode = 'pickle') -> None:
    if not isinstance(fname, str):
        raise TypeError('Filename should be a string')
    if mode == 'pickle':
        save_pickle(fname, obj)
    elif mode == 'json':
        save_json(fname, obj)
    else:
        logging.warning("Neither in pickle nor json mode, ignoring request.")


def load_pickle_obj(fname: str):
    """The function is used to read the data in .pkl file"""
    if not isinstance(fname, str):
        raise TypeError('File name should be a string')
    if not fname.endswith('.pkl'):
        raise RuntimeError(fname, 'is not a pickle file.')
    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)


def load_json_obj(fname: str):
    """the function is used to read the data in .json file"""
    if not isinstance(fname, str):
        raise TypeError('File name should be a string')
    if not fname.endswith('.json'):
        raise RuntimeError(fname, 'is not a json file.')
    with open(fname, 'r') as in_file:
        return json.loads(in_file.read())

def load_obj(fname: str):
    if fname.endswith('.pkl'):
        return load_pickle_obj(fname)
    elif fname.endswith('.json'):
        return load_json_obj(fname)
    else:
        raise RuntimeError(fname, 'is not either a json or pickle file.')
    return None


def load_config(fname: str):
    with open(fname) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    return content


def init_torch_device(select = None):
    """
    Selects PyTorch device. Accepts either a device object, a string ('cpu' or 'gpu'),
    or a number (ID of GPU, negative number indicating using CPU)
    """
    # Passthrough if it's already the torch device
    if isinstance(select, torch.device):
        return select

    if select is None:
       if cuda.is_available():
           deviceNo = 0
       else:
           deviceNo = -1
    else:
        if select.lower() == 'cpu':
            deviceNo = -1
        elif select.lower() == 'gpu':
            deviceNo = 0
        else:
            deviceNo = select

    if deviceNo >= 0:
        torch.backends.cudnn.benchmark = True
        cuda.set_device(deviceNo)
        deviceStr = f'cuda:{deviceNo}'
    else:
        deviceStr = 'cpu'
    
    device = torch.device(deviceStr)
    logging.info(f'Hardware setting done, using device: {deviceStr}')
    return device


class Recorder:
    """
    Recording the whole history in training
    """

    def __init__(self, columns):
        self._columns = columns
        self._data = []

    @property
    def record_columns(self):
        return self._columns

    @property
    def ncols(self):
        return len(self.record_columns)

    def from_old_file(self, file_name):
        df = pd.read_csv(file_name)
        self._columns = df.columns
        self._data = df.values.tolist()

    def insert(self, new_data) -> None:
        if len(new_data) != self.ncols:
            raise IndexError('Input data length is not equal to init record length.')
        self._data.append([str(obj) for obj in new_data])

    def write(self, path: str, file_name: str, file_type = '.csv') -> None:
        path = os.path.join(path, file_name) + file_type
        pd.DataFrame(self._data, columns=self.record_columns).to_csv(path)
