import os
import sys
import json
import yaml
import pickle
import logging

import torch
import torch.cuda as cuda

import lib


__all__ = ['save_object', 'load_pickle_obj', 'load_json_obj', 'load_config',
        'init_torch_device', 'Recorder']


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
        raise TypeError('File name should be a string')
    if mode == 'pickle':
        save_pickle(fname, obj)
    elif mode == 'json':
        save_json(fname, obj)
    else:
        logging.warning("Neither in pickle nor json mode, ignoring request.")


def load_pickle_obj(fname: str):
    # the function is used to read the data in .pkl file
    # if not fname.endswith('.pkl'):
        # raise RuntimeError(fname, 'is not a pickle file.')
    if not isinstance(fname, str):
        raise TypeError('File name should be a string')
    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)


def load_json_obj(fname: str):
    # the function is used to read the data in .json file
    # if not fname.endswith('.json'):
        # raise RuntimeError(fname, 'is not a json file.')
    if not isinstance(fname, str):
        raise TypeError('File name should be a string')
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
    or a number (ID of GPU, negative number indicating use cpu)
    """
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
    logging.info('Hardware setting done, using device:', deviceStr)
    return device


class Recorder():
    """
    Recording the whole history in training
    """

    def __init__(self, record_column):
        self.record_column = record_column
        self.length_check = len(record_column)
        self.data = []

    def from_old_file(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            elements = line.replace('\n', '').split(',')
            temp_list = []
            assert len(elements) == self.length_check
            for obj in elements:
                temp_list.append(obj)

            self.data.append(obj)

        return None

    def insert(self, new_data):
        if len(new_data) != self.length_check:
            raise IndexError('Input data length is not equal to init record length.')

        insertion = [str(obj) for obj in new_data]
        self.data.append(insertion)
        return None

    def write(self, path: str, file_name: str, file_type = '.csv'):
        logging.info('Start writing recording file ...')
        with open(os.path.join(path, file_name) + file_type, 'w') as f:
            lines = self._build_file()
            f.writelines(lines)
        logging.info('Recoder writing done.')
        return None

    def _build_file(self):
        lines = ['']
        for (i, rc) in enumerate(self.record_column):
            if i == len(self.record_column) - 1:
                lines[0] = lines[0] + rc + '\n'
            else:
                lines[0] = lines[0] + rc + ','

        for (i, da) in enumerate(self.data):
            new_lines = ''
            for (j, bits) in enumerate(da):
                if j == len(da) - 1:
                    new_lines = new_lines + bits + '\n'
                else:
                    new_lines = new_lines + bits + ','

            lines.append(new_lines)

        return lines
