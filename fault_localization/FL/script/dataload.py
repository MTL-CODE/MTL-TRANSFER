import pickle
import os

def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_data(opt):
    data_path = opt['data']['path']
    
    dataType = ['train', 'valid', 'test']

    data = {}
    for type in dataType:
        data[type] = read_pkl(data_path + '/' + type + '.pkl')
    return data
