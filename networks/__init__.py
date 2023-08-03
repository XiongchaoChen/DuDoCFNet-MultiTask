import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import *
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])  # Default to the 1st GPU
    network = nn.DataParallel(network, device_ids=gpu_ids)  # Parallel computing on multiple GPU

    return network


def get_generator(name, opts, ic):
    if name == "TSFuseSE_TL0":
        network = TSFuseSE_TL0(n_channels=ic, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm)

    if name == "TSFuseSE_TL":
        network = TSFuseSE_TL(n_channels=ic, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm)

    if name == "TSFuseSE":
        network  = TSFuseSE(n_channels=ic, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm)

    # (1) DuRDN / default_depth = 4
    if name == 'DuRDN4':
        network = scSERDUNet(n_channels=1, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm)

    # (2) DuRDN
    elif name == 'DuRDN3':
        network = scSERDUNet3(n_channels=1, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm)



    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters of Generator: {}'.format(num_param))

    return set_gpu(network, opts.gpu_ids)
