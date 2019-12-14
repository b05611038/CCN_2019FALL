import os
import sys
import json
import yaml
import pickle

import torch
import torch.cuda as cuda

import lib


__all__ = ['save_object', 'load_pickle_obj', 'load_json_obj', 'load_config',
        'init_torch_device', 'Recorder']


def save_object(fname, obj, mode = 'pickle'):
    # the function is used to save some data in class or object in .pkl file
    # or json file
    if not isinstance(fname, str):
        raise TypeError(fname, "is not string thus cannot be the file name of save.")

    if mode == 'pickle':
        if not fname.endswith('.pkl'):
            fname += '.pkl'

        with open(fname, 'wb') as out_file:
            pickle.dump(obj, out_file)

    elif mode == 'json':
        if not fname.endswith('.json'):
            fname += '.json'

        obj = json.dumps(obj)
        with open(fname, 'w') as out_file:
            out_file.write(obj)
    else:
        print("WARNING: not either pickle or json mode, ignoring request.")

    return None

def load_pickle_obj(fname):
    # the function is used to read the data in .pkl file
    # if not fname.endswith('.pkl'):
        # raise RuntimeError(fname, 'is not a pickle file.')

    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)

def load_json_obj(fname):
    # the function is used to read the data in .json file
    # if not fname.endswith('.json'):
        # raise RuntimeError(fname, 'is not a json file.')

    with open(fname, 'r') as in_file:
        return json.loads(in_file.read())

def load_obj(fname):
    if fname.endswith('.pkl'):
        return load_pickle_obj(fname)
    elif fname.endswith('.json'):
        return load_json_obj(fname)
    else:
        raise RuntimeError(fname, 'is not either a json or pickle file.')
    return None

def load_config(fname):
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
        deviceStr = 'cuda:' + str(deviceNo)
    else:
        deviceStr = 'cpu'
    device = torch.device(deviceStr)
    print('Hardware setting done, using device:', deviceStr)
    return device


class Recorder():
    # Recoder of recording whole history in traninig.
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

        insertion = []
        for obj in new_data:
            insertion.append(str(obj))

        self.data.append(insertion)
        return None

    def write(self, path, file_name, file_type = '.csv'):
        print('Start writing recording file ...')
        with open(os.path.join(path, file_name) + file_type, 'w') as f:
            lines = self._build_file()
            f.writelines(lines)
        print('Recoder writing done.')
        return None

    def _build_file(self):
        lines = ['']
        for i in range(len(self.record_column)):
            if i == len(self.record_column) - 1:
                lines[0] = lines[0] + self.record_column[i] + '\n'
            else:
                lines[0] = lines[0] + self.record_column[i] + ','

        for i in range(len(self.data)):
            new_lines = ''
            for j in range(len(self.data[i])):
                if j == len(self.data[i]) - 1:
                    new_lines = new_lines + self.data[i][j] + '\n'
                else:
                    new_lines = new_lines + self.data[i][j] + ','

            lines.append(new_lines)

        return lines
