from torch_geometric.data import Data
from dataclasses import dataclass

import os.path as osp
import os

import torch
import yaml

torch.manual_seed(0)


DATASET = 'dataset'
TRAIN_LOADER = 'train_loader'
VAL_LOADER = 'val_loader'
TEST_LOADER = 'test_loader'
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
    with open("../config.yml") as f:
        config = yaml.safe_load(f)
    
    if config['active'] != "all":
        name = config['active']
        config = config["datasets"][name]
        config['root'] = osp.expanduser(config['root'])
        config['device'] = torch.device("cuda:0")
        config['name'] = name
        if use_best:
            with open(f"../params/{name}.yml") as f:
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


def split_dataset(dataset):
    data = dataset.data
    train_indices = data.train_mask.nonzero(as_tuple=True)[0]
    val_indices = data.val_mask.nonzero(as_tuple=True)[0]
    test_indices = data.test_mask.nonzero(as_tuple=True)[0]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    return train_dataset, val_dataset, test_dataset


def to_gnn_input(batch, full_data=None):
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