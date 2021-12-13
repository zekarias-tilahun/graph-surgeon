from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch

import copy


torch.manual_seed(1)


def sparse_identity(size):
    i = torch.arange(size)
    diag = torch.stack([i, i])
    return torch.sparse_coo_tensor(
        diag, torch.ones(size, dtype=torch.float32), (size, size))


class SurgeonModule(nn.Module):
    
    @property
    def device(self):
        return next(self.parameters()).device


class LogisticRegression(SurgeonModule):
    """
    A simple logistic regression classifier for evaluating SelfGNN 
    """
    def __init__(self, num_dim, num_class, task='mcc'):
        super().__init__()
        assert task in {"bc", "mcc", "mlc"}
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        if task in {'bc', 'mcc'}:
            self.loss_fn = nn.CrossEntropyLoss()
        elif task == "mlc":
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        prd = self.linear(x.to(self.device))
        loss = self.loss_fn(prd, y.to(self.device))
        return prd, loss
    
    
class AugmentationLayer(SurgeonModule):
    
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.aug1 = nn.Linear(in_dim, out_dim)
        self.aug2 = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        x1 = self.aug1(x.to(self.device))
        x1 = F.dropout(x1, self.dropout, self.training)
        x2 = self.aug2(x.to(self.device))
        x2 = F.dropout(x2, self.dropout, self.training)
        return x1, x2
    
    @torch.no_grad()
    def inference(self, x_all, batch_size=10000):
        """
        Subgraph inference code adapted from PyTorch Geometric:
        https://github.com/rusty1s/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py#L36
        
        Compute representations of nodes layer by layer, using *all*
        available edges. This leads to faster computation in contrast to
        immediately computing the final representations of each batch.
        
        """
        pbar = tqdm(total=x_all.size(0))
        pbar.set_description('Inferring views')
        
        x1s, x2s = [], []
        for i in range(0, x_all.shape[0], batch_size):
            end = i + batch_size if x_all.shape[0] - batch_size > i else x_all.shape[0]
            x = x_all[i:end]
            x1, x2 = self.forward(x)
            x1s.append(x1.cpu())
            x2s.append(x2.cpu())
            
            pbar.update(end)
            
        pbar.close()
        
        x1, x2 = torch.cat(x1s), torch.cat(x2s)
        return x1, x2


class Encoder(SurgeonModule):

    def __init__(self, 
                 in_dim, out_dim, dropout, encoder='gcn', 
                 use_norm=False, layers=2, skip=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.encoder = encoder
        self.use_norm = use_norm
        self.layers = layers
        self.skip = skip
        self._init_modules()
        
    def _init_modules(self):
        if self.encoder == "gcn":
            GNNLayer = GCNConv
        elif self.encoder == "sage":
            GNNLayer = SAGEConv
            
        self.stacked_gnn = nn.ModuleList()
        
        self.stacked_gnn.append(GNNLayer(self.in_dim, self.out_dim))
        for _ in range(1, self.layers):
            self.stacked_gnn.append(GNNLayer(self.out_dim, self.out_dim))
            
        if self.skip:
            self.skips = nn.ModuleList()
            self.skips.append(nn.Linear(self.in_dim, self.out_dim))
            for _ in range(self.layers - 1):
                self.skips.append(nn.Linear(self.out_dim, self.out_dim))
            
        self.norm = BatchNorm(self.out_dim) if self.use_norm else lambda x: x

    def __full_batch_forward(self, x, edge_index, edge_attr):
        edge_attr = edge_attr if edge_attr is None else edge_attr.to(self.device)
        outputs = []
        for conv in self.stacked_gnn[:-1]:
            x = conv(x.to(self.device), edge_index=edge_index.to(self.device))
            x = F.relu(self.norm(x))
            x = F.dropout(input=x, p=self.dropout, training=self.training)
            outputs.append(x)
        
        x = self.stacked_gnn[-1](x.to(self.device), edge_index=edge_index.to(self.device))
        outputs.append(x)
        if self.skip:
            x = F.dropout(input=x, p=self.dropout, training=self.training)
            outputs[-1] = x
            return torch.stack(outputs, dim=0).sum(dim=0)
        return x
    
    def __mini_batch_forward(self, x, adj):
        """
        Subgraph inference code adapted from PyTorch Geometric:
        https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_gat.py#L58
        
        `train_loader` computes the k-hop neighborhood of a batch of nodes,
        and returns, for each layer, a bipartite graph object, holding the
        bipartite edges `edge_index`, the index `e_id` of the original edges,
        and the size/shape `size` of the bipartite graph.
        Target nodes are also included in the source nodes so that one can
        easily apply skip-connections or add self-loops.
        
        """
        outputs = []
        for i, (edge_index, _, size) in enumerate(adj):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.stacked_gnn[i]((x.to(self.device), x_target.to(self.device)),
                                    edge_index.to(self.device))
            if self.skip:
                x = x + self.skips[i](x_target.to(self.device))
            if i != self.layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def forward(self, x, edge_index, edge_attr):
        if isinstance(edge_index, torch.Tensor):
            return self.__full_batch_forward(x, edge_index, edge_attr)
        else:
            return self.__mini_batch_forward(x, edge_index)
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        """
        Subgraph inference code adapted from PyTorch Geometric:
        https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py#L47
        
        Compute representations of nodes layer by layer, using *all*
        available edges. This leads to faster computation in contrast to
        immediately computing the final representations of each batch.
        
        """
        pbar = tqdm(total=x_all.size(0) * len(self.stacked_gnn))
        pbar.set_description('Inferring embeddings of a view')
        
        for i, conv in enumerate(self.stacked_gnn):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj
                x = x_all[n_id]
                x_target = x[:size[1]]
                x = conv((x.to(self.device), x_target.to(self.device)), 
                         edge_index.to(self.device)).cpu()
                if i != len(self.stacked_gnn) - 1:
                    x = F.relu(self.norm(x.to(self.device)))
                xs.append(x.cpu())

                pbar.update(batch_size)
                torch.cuda.empty_cache()

            x_all = torch.cat(xs, dim=0)
        torch.cuda.empty_cache()
        pbar.close()

        return x_all


class Surgeon(SurgeonModule):

    def __init__(
        self, net, aug_layer, gamma=0.01, agg_method="concat", 
        pre_augment=True, use_improved_loss=True
    ):
        super().__init__()
        self.aug_layer = aug_layer
        self.net = net
        self.gamma = gamma
        self.agg_method = agg_method
        self.pre_augment = pre_augment
        self.use_improved_loss = use_improved_loss

    def lap_eimap_loss(self, x1, x2):
        x1 = F.normalize(x1, dim=1, p=2)
        x2 = F.normalize(x2, dim=1, p=2)
        mse = 2 - 2 * (x1 * x2).sum(dim=-1).mean()
        
        if self.use_improved_loss:
            I = sparse_identity(size=x1.shape[1]).to(self.device)
            constraint = self.gamma * (
                (x1.t().matmul(x1) - I).norm() + 
                (x2.t().matmul(x2) - I).norm()
            )
        else:
            I = sparse_identity(size=x1.shape[0]).to(self.device)
            constraint = self.gamma * (
                (x1.matmul(x1.t()) - I).norm() + 
                (x2.matmul(x2.t()) - I).norm()
            )
        return 1000 * mse + constraint

    def aggregate_views(self, x1, x2):
        if self.agg_method == "mean":
            return (x1 + x2) / 2.
        elif self.agg_method == "sum":
            return x1 + x2
        return torch.cat([x1, x2], dim=-1)
    
    def _forward(self, x, edge_index, edge_attr=None):
        if self.pre_augment:
            view1, view2 = self.aug_layer(x)
            z1 = self.net(x=view1, edge_index=edge_index, edge_attr=edge_attr)
            z2 = self.net(x=view2, edge_index=edge_index, edge_attr=edge_attr)
            return z1, z2
        else:
            x = self.net(x=x, edge_index=edge_index, edge_attr=edge_attr)
            return self.aug_layer(x)

    def forward(self, x, edge_index, edge_attr=None):
        z1, z2 = self._forward(x, edge_index, edge_attr)
        return self.lap_eimap_loss(z1, z2)
    
    @torch.no_grad()
    def infer(self, x, edge_index, edge_attr, loader=None):
        if loader is None:
            z1, z2 = self._forward(x, edge_index, edge_attr)
        else:
            if self.pre_augment:
                x1, x2 = self.aug_layer.inference(x)
                torch.cuda.empty_cache()
                z1 = self.net.inference(x1, loader).cpu()
                torch.cuda.empty_cache()
                z2 = self.net.inference(x2, loader).cpu()
            else:
                x = self.net.inference(x, loader).cpu()
                torch.cuda.empty_cache()
                z1, z2 = self.aug_layer.inference(x)
                
        torch.cuda.empty_cache()
        return self.aggregate_views(z1, z2)
