# A PyTorch implementation of the GraphSurgeon [paper](https://arxiv.org/abs/2108.10420)

## Requirements!

- Python 3.7+
- PyTorch 1.9.0
- PyTorch Geometric 1.7.2
- Numpy 1.17.2+
- Networkx 2.3+
- SciPy 1.5.4+
- (OPTINAL) OPTUNA 2.8.0+ ```If you wish to tune the hyper-parameters of GraphSrugeon```

## Example usage

### Training

```sh
$ python src/main.py
```

### Tuning

```sh
$ python src/tune.py
```

## Possible options for training GraphSurgeon

The ```config.yml``` file can be used to configure the hyperparameters of GraphSurgeon, and this can be done by using
the following ```key```:```value``` pairs

`active`: `<name>` - the `active` key is used to specify the `name` of the dataset that we wish to run <br>
`datasets`: `dataset_config_list` - The `datasets` key identifies a list of dataset config. Each dataset config in
`dataset_config_list` is further identified by a key (the `name` of the dataset) that contains the following nested
`key`:`value` pairs. <br>

```

batch_size: <int_value> - batch size for sampling based GNNs, for full batch GNN this will be ignored <br>
aug_dim: <int_value> - the dimension of the augmentation head <br>
model_dim: <int_value> - the dimension used in the output representation <br>
dropout: <float_value> - the dropout rate between 0 (inclusive) and 1 (exclusive) <br>
epochs: <int_value> - the number of self-supervised training epochs <br>
loader: <string_value> from {`full`, `neighborhood`, `cluster`, `saint`} - Here we specify the type of GNN, full-batch or sampling based (`neighborhood` - for GraphSAGE neighborhood sampling, `cluster` - for ClusterGCN, `saint` - for GraphSaint subgraph sampling) <br>
lr: <float_value> - learning rate <br>
layers: <int_value> - The number of layers of the GNN encoder <br>
pre_aug: <bool_value> - True for pre-augmenation and False for post-augmentation <br>
root: <string_value> - A path to the directory to store the dataset <br>
task: <string_value> from {`bc`, `mcc`, `mlc`} - The desired down-stream task, `bc` - binary classification, `mcc` - multi-class classification and `mlc` - multi-label classification <br>
workers: <int_value> - The number of cpu workers<br>
```

Citing
------

If you find this research helpful, please cite it as

```
@misc{kefato2021jointly,
      title={Jointly Learnable Data Augmentations for Self-Supervised GNNs}, 
      author={Zekarias T. Kefato and Sarunas Girdzijauskas and Hannes Stärk},
      year={2021},
      eprint={2108.10420},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

or 

```
@misc{kefatosslneurips2021,
      title={Self-Supervised GNN that Jointly Learns to Augment}, 
      author={Zekarias T. Kefato and Sarunas Girdzijauskas and Hannes Stärk},
      year={2021},
      booktitle={NeurIPS 2021 Workshop: Self-Supervised Learning - Theory and Practice}
}
```
