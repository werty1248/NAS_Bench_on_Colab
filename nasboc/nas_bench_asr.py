import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import nasbench_asr as nbasr
from nasbench_asr.search_space import get_random_architectures
from nasbench_asr.graph_utils import get_model_graph
from nasbench_asr.model import get_model, set_default_backend

from .base import NASBenchConfig, NASBenchAPIBase


# Useful constants
INPUT = 'input'
OUTPUT = 'output'
LINEAR = 'linear'
CONV5X5 = 'conv5'
CONV7X7 = 'conv7'
DILCONV5X5 = 'conv5d2'
DILCONV7X7 = 'conv7d2'
ZERO = 'zero'

NUM_VERTICES = 5

OP_LIST = [INPUT, LINEAR, CONV5X5, DILCONV5X5, CONV7X7, DILCONV7X7, ZERO, OUTPUT]

DEFAULT_RNN = True
DEFAULT_DROPOUT = 0.2

def config_to_archvec(config):
  matrix = config['matrix']
  ops = config['ops']
  archvec = []
  for i in range(1,4):
    op_index = OP_LIST.index(ops[i]) - 1 #INPUT is external op
    outer_edge = matrix[:i,i+1].tolist()
    archvec.append([op_index,] + outer_edge)
  return archvec

class NASBenchASRAPI(NASBenchAPIBase):
  def __init__(self, bench_config):
    super(NASBenchASRAPI, self).__init__()

    self.config = bench_config
    self.filepath = self.config.filepath
    
    self.api = nbasr.from_folder(self.filepath, include_static_info=True)
    set_default_backend('torch')

  def query_by_config(self, config):
    archvec = config_to_archvec(config)
    full_info = self.api.full_info(archvec)

    model = self.get_model(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nparam = sum([np.prod(p.size()) for p in model_parameters])
    del model

    val_acc = full_info['val_per'][-1]
    test_acc = full_info['test_per']
    train_time = 0.
    return {'nparam':nparam, 'val_acc':val_acc, 'test_acc':test_acc, 'training_time':train_time}

  def config2graph(self, config):
    node_feature = []
    for op in config['normal']['ops']:
      op_index = OP_LIST.index(op)
      node_feature.append(torch.eye(len(OP_LIST))[op_index])
    node_feature = torch.stack(node_feature)

    graph = Data(x=node_feature, edge_index=torch.tensor(config['normal']['matrix']).nonzero().t().contiguous())
    return graph

  def graph2config(self, graph):
    matrix = torch.sparse.FloatTensor(graph.edge_index, torch.ones(NUM_VERTICES).type(torch.int8), torch.Size([NUM_VERTICES,NUM_VERTICES])).to_dense().numpy()
    op_index_list = graph.x.nonzero().t().contiguous()[1,:]
    ops = []
    for op_index in op_index_list:
      ops.append(OP_LIST[op_index])
    return {'normal':{'matrix':matrix, 'ops':ops}}

  def get_model(self, config):
    archvec = config_to_archvec(config)
    model = get_model(archvec, backend = 'torch', use_rnn = DEFAULT_RNN, dropout_rate = DEFAULT_DROPOUT)

    return model

  def sample_configuration(self, num_sample):
    configs = []
    archs = get_random_architectures(num_sample)

    for arch in archs:
      matrix, ops = get_model_graph(arch)[1]
      config = {'normal':{'matrix':matrix.astype(np.int64), 'ops':ops}}
      configs.append(config)
    return configs