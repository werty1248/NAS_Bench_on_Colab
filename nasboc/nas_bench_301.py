import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.anynet import AnyNet
from pycls.models.nas.genotypes import GENOTYPES, Genotype
from ConfigSpace.read_and_write import json as cs_json

import nasbench301 as nb

from .base import NASBenchAPIBase
from .utils import config_to_genotype, genotype_to_config

INPUT1 = 'input_1'
INPUT2 = 'input_2'
OUTPUT = 'output'
AVGPOOL3X3 = 'avg_pool_3x3'
MAXPOOL3X3 = 'max_pool_3x3'
SEPCONV3X3 = 'sep_conv_3x3'
SEPCONV5X5 = 'sep_conv_5x5'
DILCONV3X3 = 'dil_conv_3x3'
DILCONV5X5 = 'dil_conv_5x5'
SKIP_CONNECT = 'skip_connect'

OP_LIST   =[INPUT1, INPUT2, AVGPOOL3X3, MAXPOOL3X3, SKIP_CONNECT, SEPCONV3X3, SEPCONV5X5, DILCONV3X3, DILCONV5X5, OUTPUT]

NUM_VERTICES = 11
NUM_CLASSES = 10
DEFAULT_WIDTH = 32
DEFAULT_DEPTH = 8
DEFAULT_AUX = True

NUM_BLOCK = 4
CONCAT_TYPE = 'all'

def random_cell(num_block = 4, concat_type = 'all'):
  edges = []
  edge_concat = None

  if concat_type == 'all':
    edge_concat = list(range(2,num_block+2))
  else:
    edge_concat = (np.where(np.random.rand(4) > 0.5)[0] + 2).tolist()
  
  for i in range(num_block):
    for op, in_index in zip(np.random.choice(OP_LIST[2:-1], 2), np.random.choice(range(i+2), 2, replace=False)):
      edges.append((op, in_index))
  
  return edges, edge_concat

def random_genotype(num_block = 4, concat_type = 'all', same_reduce = False):
  normal, normal_concat = random_cell(num_block, concat_type)

  if same_reduce:
    reduce, reduce_concat = normal, normal_concat
  else:
    reduce, reduce_concat = random_cell(num_block, concat_type)
  
  return Genotype(normal = normal, normal_concat = normal_concat,
                  reduce = reduce, reduce_concat = reduce_concat)

class NASBench301API(NASBenchAPIBase):
  OP_LIST   =[INPUT1, INPUT2, AVGPOOL3X3, MAXPOOL3X3, SKIP_CONNECT, SEPCONV3X3, SEPCONV5X5, DILCONV3X3, DILCONV5X5, OUTPUT]
  NUM_VERTICES = 11

  def __init__(self, bench_config, version = '1.0'):
    super(NASBench301API, self).__init__()

    self.bench_config = bench_config

    self.version = version

    models_0_9_dir = os.path.join(self.bench_config.filepath, 'nb_models_0.9')
    model_paths_0_9 = {
        model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    models_1_0_dir = os.path.join(self.bench_config.filepath, 'nb_models_1.0')
    model_paths_1_0 = {
        model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    model_paths = model_paths_0_9 if self.version == '0.9' else model_paths_1_0

    # If the models are not available at the paths, automatically download
    # the models
    # Note: If you would like to provide your own model locations, comment this out
    if not all(os.path.exists(model) for model in model_paths.values()):
        nb.download_models(version=self.version, delete_zip=True,
                          download_dir=self.bench_config.filepath)

    # Load the performance surrogate model
    #NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
    #NOTE: Defaults to using the default model download path
    print("==> Loading performance surrogate model...")
    ensemble_dir_performance = model_paths['xgb']
    print(ensemble_dir_performance)
    self.performance_model = nb.load_ensemble(ensemble_dir_performance)

    # Load the runtime surrogate model
    #NOTE: Defaults to using the default model download path
    print("==> Loading runtime surrogate model...")
    ensemble_dir_runtime = model_paths['lgb_runtime']
    self.runtime_model = nb.load_ensemble(ensemble_dir_runtime)

  def query_by_config(self, config, noise = True):
    model = self.get_model(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nparam = sum([np.prod(p.size()) for p in model_parameters])
    del model

    genotype = config_to_genotype(config, self.NUM_VERTICES, self.OP_LIST)

    val_acc = self.performance_model.predict(config=genotype, representation="genotype", with_noise=noise)
    test_acc = val_acc

    train_time = self.runtime_model.predict(config=genotype, representation="genotype")

    return {'nparam':nparam, 'val_acc':val_acc, 'test_acc':test_acc, 'training_time':train_time}

  def config2genotype(self, config):
    return config_to_genotype(config, self.NUM_VERTICES, self.OP_LIST)
    
  def genotype2config(self, genotype):
    return genotype_to_config(genotype, self.NUM_VERTICES, self.OP_LIST)

  def config2graph(self, config):
    graphs = {}
    for cell_name in ['normal','reduce']:
      node_feature = []
      for op in config[cell_name]['ops']:
        op_index = self.OP_LIST.index(op)
        node_feature.append(torch.eye(len(self.OP_LIST)).type(torch.LongTensor)[op_index])
      node_feature = torch.stack(node_feature)

      graph = Data(x=node_feature, edge_index=torch.tensor(config[cell_name]['matrix']).nonzero().t().contiguous())
      graphs[cell_name] = graph
    return graphs

  def graph2config(self, graphs):
    config = {}
    for cell_name, graph in graphs.item():
      matrix = torch.sparse.FloatTensor(graph.edge_index, torch.ones(self.NUM_VERTICES).type(torch.int8), torch.Size([self.NUM_VERTICES,self.NUM_VERTICES])).to_dense().numpy()
      op_index_list = graph.x.nonzero().t().contiguous()[1,:]
      ops = []
      for op_index in op_index_list:
        ops.append(self.OP_LIST[op_index])
      config[cell_name] = {'matrix':matrix, 'ops':ops}

    return config

  def get_model(self, config):
    genotype = self.config2genotype(config)
    network = NetworkCIFAR(DEFAULT_WIDTH, NUM_CLASSES, DEFAULT_DEPTH, DEFAULT_AUX,  genotype)
    network.drop_path_prob = 0.

    return network

  def sample_configuration(self, num_sample):
    configs = []
    
    for i in range(num_sample):
      genotype = random_genotype(NUM_BLOCK, CONCAT_TYPE)
      config = genotype_to_config(genotype, self.NUM_VERTICES, self.OP_LIST)
      configs.append(config)

    return configs