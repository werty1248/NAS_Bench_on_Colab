import os
import random
import torch
import torch.nn as nn
import numpy as np
from nasbench import api
from ConfigSpace import ConfigurationSpace
from nasbench_pytorch.model import Network
from nasbench_pytorch.model import ModelSpec
from torch_geometric.data import Data

from .base import NASBenchConfig, NASBenchAPIBase


# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

OP_LIST = [INPUT,CONV3X3, CONV1X1, MAXPOOL3X3, OUTPUT]

class NASBench101API(NASBenchAPIBase):
  def __init__(self, bench_config):
    super(NASBench101API, self).__init__()

    self.config = bench_config
    self.filepath = os.path.join(self.config.filepath, 'nasbench_only108.tfrecord')
    self.api = api.NASBench(self.filepath)

  def query_by_config(self, config):
    cell = api.ModelSpec(**config['normal_cell'])
    metrics = self.api.query(cell)
    nparam = metrics['trainable_parameters']
    val_acc = metrics['validation_accuracy']
    test_acc = metrics['test_accuracy']
    train_time = metrics['training_time']
    return {'nparam':nparam, 'val_acc':val_acc, 'test_acc':test_acc, 'training_time':train_time}

  def config2graph(self, config):
    node_feature = []
    for op in config['normal']['ops']:
      op_index = OP_LIST.index(op)
      node_feature.append(torch.eye(len(OP_LIST))[op_index])
    node_feature = torch.stack(node_feature)

    graph = Data(x=node_feature, edge_index=torch.tensor(config['normal']['matrix']).nonzero().t().contiguous())
    return {"normal":graph}

  def graph2config(self, graph):
    matrix = torch.sparse.FloatTensor(graph.edge_index, torch.ones(NUM_VERTICES).type(torch.int8), torch.Size([NUM_VERTICES,NUM_VERTICES])).to_dense().numpy()
    op_index_list = graph.x.nonzero().t().contiguous()[1,:]
    ops = []
    for op_index in op_index_list:
      ops.append(OP_LIST[op_index])
    return {"normal":{'matrix':matrix, 'ops':ops}}

  def _get_children(self, model):
      children = list(model.children())
      all_children = []
      if children == []:
          return model
      else:
          for child in children:
              try:
                  all_children.extend(self._get_children(child))
              except TypeError:
                  all_children.append(self._get_children(child))

      return all_children

  def get_model(self, config):
    spec = ModelSpec(config['normal']['matrix'], config['normal']['ops'])
    model = Network(spec,
                    num_labels=10,
                    in_channels=3,
                    stem_out_channels=128,
                    num_stacks=3,
                    num_modules_per_stack=3)

    all_leaf_modules = self._get_children(model)
    inplace_relus = [module for module in all_leaf_modules if (isinstance(module, nn.ReLU) and module.inplace == True)]

    for relu in inplace_relus:
        relu.inplace = False

    return model

  def sample_configuration(self, num_sample):
    arch_hash_list = np.random.choice(list(self.api.hash_iterator()), num_sample)
    configs = []
    for arch_hash in arch_hash_list:
      metrics = self.api.get_metrics_from_hash(arch_hash)[0]
      spec_dict = {"normal":{'matrix':metrics['module_adjacency'], 'ops':metrics['module_operations']}}
      configs.append(spec_dict)
    return configs