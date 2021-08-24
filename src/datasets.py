from torch_geometric.data import Data, Batch, ClusterData, ClusterLoader, InMemoryDataset
from torch_geometric.data import DataLoader, GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.datasets import CitationFull, Coauthor, Amazon, WikiCS, Flickr
from torch_geometric.datasets import Actor, PPI, Reddit, Yelp
from pyg_datasets import AmazonProducts, FacebookPagePage, GitHub
from torch_geometric.transforms import GDC
import torch_geometric.transforms as T

import os.path as osp

import torch

import utils


class CompiledDataset:
    
    def __init__(self, args):
        self._args = args
        
    def compile(self):
        args = self._args
        compiled_data = {}
        if args.loader.lower() == "full":
            dataset = Dataset(args.root)
            compiled_data[utils.DATASET] = dataset
            compiled_data[utils.TRAIN_LOADER] = [dataset.data]
        elif args.loader.lower() == "saint":
            dataset = Dataset(args.root)
            loader = GraphSAINTRandomWalkSampler(
                dataset[0], batch_size=args.batch_size, walk_length=2,
                num_steps=5, sample_coverage=100,
                save_dir=dataset.processed_dir,
                num_workers=32)
            compiled_data[utils.DATASET] = dataset
            compiled_data[utils.TRAIN_LOADER] = loader
            if args.name.lower() in {'reddit', 'ogbn_products'}:
                subgraph_loader = NeighborSampler(
                    dataset.data.edge_index, sizes=[-1], batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers)
                compiled_data[utils.SUBGRAPH_LOADER] = subgraph_loader
        elif args.loader.lower() == "cluster":
            dataset = Dataset(args.root)
            data = dataset[0]
            cluster_data = ClusterData(
                data, num_parts=args.parts, recursive=False,
                save_dir=dataset.processed_dir)
            train_loader = ClusterLoader(
                cluster_data, batch_size=20, shuffle=True, num_workers=args.workers)
            subgraph_loader = NeighborSampler(
                data.edge_index, sizes=[-1], batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers)
            compiled_data[utils.DATASET] = dataset
            compiled_data[utils.TRAIN_LOADER] = train_loader
            compiled_data[utils.SUBGRAPH_LOADER] = subgraph_loader
        elif args.loader.lower() == "neighborhood":
            dataset = Dataset(args.root)
            data = dataset.data
            train_idx = data.train_mask.nonzero(as_tuple=True)[0]
            train_loader = NeighborSampler(
                data.edge_index, node_idx=train_idx, sizes=[32] * args.layers, 
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            subgraph_loader = NeighborSampler(
                data.edge_index, node_idx=None, sizes=[-1], batch_size=args.batch_size, 
                shuffle=False, num_workers=args.workers)
            compiled_data[utils.DATASET] = dataset
            compiled_data[utils.TRAIN_LOADER] = train_loader
            compiled_data[utils.SUBGRAPH_LOADER] = subgraph_loader
        else:
            raise ValueError(
                """Unknown value for the argument 'loader'. Valid options are 'full', 'saint', 
                'neighborhood' and 'cluster'""")
        return compiled_data
            


class Dataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(Dataset, self).__init__(root, transform, pre_transform)
        print("Loading data ...")
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(self.data)

    def download(self):
        if not osp.exists(self.processed_paths[0]):
            if self.root.startswith("~"):
                self.root = osp.expanduser(self.root)
            root, name = osp.split(self.root)
            fetch_data(root=root, name=name)

    def process(self):
        pass

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]


def fetch_data(root, name):
    if name.lower() in {'cora', "pubmed", "dblp"}:
        dataset = CitationFull(root=root, name=name)
    elif name.lower() in {'computers', "photo"}:
        dataset = Amazon(root=root, name=name)
    elif name.lower() in {'cs',  'physics'}:
        dataset = Coauthor(root=root, name=name)
    elif name.lower() in {"wiki", "wikics"}:
        dataset = WikiCS(osp.join(root, name)) 
    elif name.lower() == "flickr":
        dataset = Flickr(osp.join(root, name))
    elif name.lower() == "actor":
        dataset = Actor(osp.join(root, name))
    elif name.lower() == "yelp":
        dataset = Yelp(osp.join(root, 'yelp'))
    elif name.lower() == "reddit":
        dataset = Reddit(osp.join(root, name))
    elif name.lower() == "facebook":
        dataset = FacebookPagePage(osp.join(root, name))
    elif name.lower() == "git":
        dataset = GitHub(osp.join(root, name))
    return update_if(dataset)


def update_if(dataset):
    data = dataset[0]
    updated = False
    if not hasattr(data, "train_mask"):
        print("Creating masks")
        updated = True
        train_mask, val_mask, test_mask = utils.create_mask(data)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    if isinstance(data.y, torch.LongTensor):
        if len(data.y.shape) > 1 and data.y.sum(dim=-1) > 1: # is multi-label
            print("Casting class labels to float tensor ...")
            data.y = data.y.float()
            updated = True
            
    if updated:
        data, slices = dataset.collate([data])
        torch.save((data, slices), dataset.processed_paths[0])
