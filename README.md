# A PyTorch implementation of the GraphSurgeon [paper](https://arxiv.org/abs/2108.10420)

## Requirements!

- Python 3.6+
- PyTorch 1.6+
- PyTorch Geometric 1.6+
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

The following options can be configured in the ```config.yml``` file, which contains the following ```key```:```value```
pairs

`active`: `<name>` - the `active` key is used to specify the `name` of the dataset that we wish to run <br>

`datasets`: `dataset_config_list` - The `datasets` key identifies a list of dataset config. Each dataset config in
`dataset_config_list` is further identified by a key (the `name` of the dataset) that contains the following nested
`key`:`value` pairs. <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`batch_size`: `<int_value>` - batch size for sampling based GNNs, for full batch GNN
this will be ignored <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`aug_dim`: `<int_value>` - the dimension of the augmentation head <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`model_dim`: `<int_value>` - the dimension used in the output representation <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`dropout`: `<float_value>` - the dropout rate between 0 (inclusive) and 1 (
exclusive) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`epochs`: `<int_value>` - the number of self-supervised training epochs <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`loader`: `<string_value>` from {`full`, `neighborhood`, `cluster`, `saint`} - Here
we specify the type of GNN, full-batch or sampling based (`neighborhood` - for GraphSAGE neighborhood
sampling, `cluster` - for ClusterGCN, `saint` - for GraphSaint subgraph sampling) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`lr`: `<float_value>` - learning rate <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`layers`: `<int_value>` - The number of layers of the GNN encoder <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`pre_aug`: `<bool_value>` - True for pre-augmenation and False for post-augmentation <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`root`: `<string_value>` - A path to the directory to store the dataset <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`task`: `<string_value>` from {`bc`, `mcc`, `mlc`} - The desired down-stream task,
`bc` - binary classification, `mcc` - multi-class classification and
`mlc` - multi-label classification <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`workers`: `<int_value>` - The number of cpu workers<br>


