from model import Encoder, Surgeon, LogisticRegression, AugmentationLayer
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from collections import namedtuple

import numpy as np

import datasets
import utils

import torch

torch.manual_seed(1)


class SelfExec:

    def __init__(self, args):
        self._args = args
        self.__init_data()
        self.__init_state()

    def __init_data(self):
        cd = datasets.CompiledDataset(self._args)
        compiled_dataset = cd.compile()
        self.dataset = compiled_dataset[utils.DATASET]
        self._loader = compiled_dataset[utils.TRAIN_LOADER]
        self._subgraph_loader = None
        if utils.SUBGRAPH_LOADER in compiled_dataset:
            self._subgraph_loader = compiled_dataset[utils.SUBGRAPH_LOADER]

        utils.create_dirs(root=self._args.root,
                          name_list=['results', "model"])

    def __init_state(self):
        learner, optimizer = self.__reset_model()
        self.state = {"learner": learner, "optimizer": optimizer}
        self.state = (
            namedtuple("StateDict", self.state.keys())
            (*self.state.values())
        )

    def __reset_model(self):
        print("Initializing model ...")
        args = self._args
        hidden_dim = args.aug_dim
        out_dim = args.model_dim
        if args.pre_aug:
            aug_in_dim = self.dataset.data.x.shape[1]
            aug_out_dim = hidden_dim
            enc_in_dim = aug_out_dim
            enc_out_dim = args.model_dim
        else:
            aug_in_dim = hidden_dim
            aug_out_dim = args.model_dim
            enc_in_dim = self.dataset.data.x.shape[1]
            enc_out_dim = hidden_dim

        aug_layer = AugmentationLayer(
            in_dim=aug_in_dim, out_dim=aug_out_dim, dropout=args.dropout)

        encoder_type = "sage" if args.loader in {"cluster", "neighborhood"} else "gcn"
        net = Encoder(
            in_dim=enc_in_dim, out_dim=enc_out_dim,
            dropout=args.dropout, layers=args.layers,
            encoder=encoder_type, use_norm=args.norm,
            skip=True)

        learner = Surgeon(
            net=net, aug_layer=aug_layer, gamma=args.gamma,
            pre_augment=args.pre_aug).to(args.device)

        utils.log(learner, verbose=args.verbose)
        optimizer = torch.optim.Adam(
            learner.parameters(), lr=args.lr)
        return learner, optimizer

    def __train_epoch(self, epoch, epochs):
        learner, optimizer = self.state.learner, self.state.optimizer
        losses = []
        torch.autograd.set_detect_anomaly(True)
        for data in self._loader:
            surgeon_input = utils.to_surgeon_input(batch=data, full_data=self.dataset.data)
            loss = learner(**surgeon_input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            msg = f"Epoch: {epoch + 1:03d}/{epochs:03d} training loss: {loss:.4f}"
            utils.log(msg, verbose=self._args.verbose)
            losses.append(float(loss.detach()))
        return np.mean(losses)

    def execute(self):
        print("Training ...")
        args = self._args
        learner, optimizer = self.state.learner, self.state.optimizer
        learner.train()
        for epoch in range(args.epochs):
            self.__train_epoch(epoch, args.epochs)

    def resume(self, epochs=1):
        for epoch in range(epochs):
            self.__train_epoch(epoch, epochs)

    def pause_training_mode(self):
        self.state.learner.eval()

    def get_inference_args(self):
        args = self._args
        inference_args = {
            "full": {
                "data": self.dataset.data
            },
            "saint": {
                "data": self.dataset.data
            },
            "cluster": {
                "data": self.dataset.data,
                "loader": self._subgraph_loader
            },
            "neighborhood": {
                "data": self.dataset.data,
                "loader": self._subgraph_loader
            }
        }
        return inference_args[args.loader.lower()]

    def infer_embedding(
            self, data=None, loader=None, aggregate="concat", return_loader=False):
        error_message = (
            "Both the 'data' or 'loader' argument can not be none. "
            "Please specify on or both of them"
        )
        assert data is not None or loader is not None, message
        args = self._args
        utils.log("Inferring embeddings", verbose=args.verbose)
        learner = self.state.learner
        learner.eval()
        if data is not None and loader is not None:
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
            return learner.infer(
                x=data.x, edge_index=data.edge_index,
                edge_attr=edge_attr, loader=loader)
        elif data is None:
            z = y = None
            for data in loader:
                gnn_input = utils.to_surgeon_input(batch=data, full_data=self.dataset.data)
                z_ = learner.infer(**gnn_input)
                if z is None:
                    z, y = z_, data.y
                else:
                    z, y = torch.cat([z, z_]), torch.cat([y, data.y])
            if return_loader:
                return DataLoader(
                    list(zip(z.detach().cpu(), y.detach().cpu())),
                    batch_size=args.batch_size,
                    num_workers=args.workers)
            return z, y
        else:
            return learner.infer(
                x=data.x, edge_index=data.edge_index,
                edge_attr=data.edge_attr)
            # return torch.cat([z1, z2], dim=-1).detach().cpu()

    def reset_model(self):
        self.__init_state()


class LinearEvalExec:

    def __init__(self, in_dim, out_dim, device, task, verbose=True):
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._device = device
        self._task = task
        self._metric = "accuracy" if task in {"bc", "mcc"} else "roc_auc"
        self._verbose = verbose

    def __feed(self, x, y, mask=None):
        if mask is None:
            return self._classifier(x, y)
        mask.to(self._device)
        return self._classifier(
            x.to(self._device)[mask],
            y.to(self._device)[mask]
        )

    def score(self, scores, truth):

        if self._metric == "accuracy":
            prediction = torch.argmax(scores, dim=1)
            return (
                    (prediction.to(self._device) == truth.to(self._device)).sum() /
                    (truth.shape[0] + 0.)
            )
        elif self._metric == "roc_auc":
            # scores = (scores > 0).float()
            return (
                roc_auc_score(truth.detach().cpu().numpy(), scores.detach().cpu().numpy())
            )

    def execute(self, x, y, train_mask, val_mask=None, test_mask=None):
        iters = 10 if len(train_mask.shape) == 1 else train_mask.shape[1]
        print("Training a linear classifier")
        seeds = range(iters)
        val_accs, test_accs = [], []
        val_msg = test_msg = ""
        val_best = test_best = 0.
        for run in range(iters):
            torch.manual_seed(seeds[run])
            self._classifier = LogisticRegression(
                self._in_dim, self._out_dim, task=self._task).to(self._device)
            opt = torch.optim.Adam(
                self._classifier.parameters(), lr=0.01, weight_decay=0.0)
            mask_index = None if len(train_mask.shape) == 1 else run
            train_mask, val_mask, test_mask = utils.index_mask(
                train_mask, val_mask=val_mask,
                test_mask=test_mask, index=mask_index)

            for i in range(500):
                self._classifier.train()
                logits, loss = self.__feed(x=x, y=y, mask=train_mask)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if (i + 1) % 5 == 0:
                    self._classifier.eval()
                    train_acc = self.score(scores=logits, truth=y[train_mask].squeeze())
                    val_logits, _ = self.__feed(x=x, y=y, mask=val_mask)
                    val_acc = self.score(scores=val_logits, truth=y[val_mask].squeeze())
                    test_logits, _ = self.__feed(x=x, y=y, mask=test_mask)
                    test_acc = self.score(scores=test_logits, truth=y[test_mask].squeeze())
                    print(f"""
                    Run {run + 1} Epoch {i + 1} train : {train_acc} 
                    validation : {val_acc} test: {test_acc}""")

                    if val_acc > val_best:
                        val_best = val_acc
                        test_best = test_acc

            val_acc, test_acc = val_best, test_best
            val_msg = f"validation {self._metric}: {val_acc:.4f}"
            test_msg = f"test {self._metric}: {test_acc:.4f}"

            val_accs.append(float(val_acc * 100))
            test_accs.append(float(test_acc * 100))

            msg = (
                f"Finished experiment {i + 1:03d}/{iters:03d} of the logistic regression "
                f"classifier. training {self._metric}: "
                f"{train_acc:.2f} {val_msg} {test_msg}".strip()
            )

            utils.log(msg, verbose=self._verbose)

        if len(val_accs) > 0:
            print("Validation", np.mean(val_accs), np.std(val_accs))
        if len(test_accs) > 0:
            print("Test", np.mean(test_accs), np.std(test_accs))

        return np.mean(val_accs)
