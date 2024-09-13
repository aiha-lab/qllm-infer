import numpy as np
import torch
import argparse

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() in ('None'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    if v=='none':
        return []
    vv = v.split(',')
    ret = []
    for vvv in vv:
        ret.append(vvv)
    return ret

def str2intlist(v):
    vv = v.split(',')
    ret = []
    for vvv in vv:
        ret.append(int(vvv))
    return ret

def str2int(v):
    if v.lower() in ('none'):
        return None
    else:
        return int(v)
