from torch_geometric.data import Data
from dataclasses import dataclass

import os.path as osp
import os

import torch
import yaml

torch.manual_seed(0)


DATASET = 'dataset'
TRAIN_LOADER = 'train_loader'
SUBGRAPH_LOADER = 'subgraph_loader'


@dataclass
class Config:
    name: str
    device: object
    batch_size: int = 1000
    aug_dim: int = 256
    model_dim: int = 128
    dropout: float = 0.5
    epochs: int = 1
    gamma: float = 0.1
    lr: float = 0.01
    loader: str = "full"
    layers: int = 2
    norm: bool = True
    pre_aug: bool = True
    root: str = osp.expanduser("~/workspace/data/surgeon/")
    task: str = "mcc"
    workers: int = 32
    verbose: bool = True
        

def parse_args(use_best=True):
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    
    if config['active'] != "all":
        name = config['active']
        config = config["datasets"][name]
        config['root'] = osp.expanduser(config['root'])
        config['device'] = torch.device(f"cuda:{get_device_id()}")
        config['name'] = name
        if use_best:
            with open(f"./params/{name}.yml") as f:
                best_conf = yaml.safe_load(f)
                config['lr'] = best_conf['lr']
                config['gamma'] = best_conf['gamma']
                config['dropout'] = best_conf['dropout']
                config['aug_dim'] = best_conf['aug_dim']
    else:
        return config

    return Config(**config)


def log(msg, stream=print, verbose=False):
    if verbose:
        stream(msg)


def index_mask(train_mask, val_mask=None, test_mask=None, index=0):
    train_mask = train_mask if len(train_mask.shape) == 1 else train_mask[:, index]
    val_mask = val_mask if val_mask is None or len(val_mask.shape) == 1 else val_mask[:, index]
    test_mask = test_mask if test_mask is None or len(test_mask.shape) == 1 else test_mask[:, index]
    return train_mask, val_mask, test_mask


def create_mask(data, train_rate=0.05, val_rate=0.15):
    perm = torch.randperm(data.num_nodes)
    train_size = int(data.num_nodes * train_rate)
    val_size = int(data.num_nodes * val_rate)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask = ~(train_mask + val_mask)
    return train_mask, val_mask, test_mask


def create_dirs(root, name_list):
    for name in name_list:
        os.makedirs(osp.join(root, name), exist_ok=True)


def to_surgeon_input(batch, full_data=None):
    if isinstance(batch, Data):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
    else:
        batch_size, node_ids, adjs = batch
        x = full_data.x[node_ids]
        edge_index = [elem for elem in adjs]  # edge_index = (edge_index, e_id, size)
        edge_attr = None
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}


def get_device_id():
    gpu_0_free_space, _ = get_gpu_memory_from_nvidia_smi(device=0)
    gpu_1_free_space, _ = get_gpu_memory_from_nvidia_smi(device=1)
    if gpu_0_free_space < 2000 and gpu_1_free_space < 2000:
        device_id = -1
    else:
        device_id = 0 if gpu_0_free_space > gpu_1_free_space else 1
    print(f"Automatically selected device: gpu {device_id}")
    return device_id