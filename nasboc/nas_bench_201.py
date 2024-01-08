#메인 아이디어: NB201은 인덱스를 통한 관리를 한다
#ModelSpec <-> str <-> index
import numpy as np

OP_LIST = ['input', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'none', 'output']
ADJ_MATRIX = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 1 ,0 ,0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0]])
import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from nas_201_api import NASBench201API as api
from xautodl.models import get_cell_based_tiny_net, get_search_spaces

from .base import NASBenchAPIBase


NUM_VERTICES = 8
MAX_EDGES = 10
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
NUM_CLASSES = {'cifar10':10, 'cifar100':100, 'ImageNet16-120':120}

class NASBench201API(NASBenchAPIBase):
  def __init__(self, bench_config):
    super(NASBench201API, self).__init__()

    self.bench_config = bench_config
    self.filepath = os.path.join(self.bench_config.filepath, 'NAS-Bench-201-v1_0-e61699.pth')
    self.api = api(self.filepath, verbose=False)
    
    self.num_classes = NUM_CLASSES[self.bench_config.dataset]
    if self.bench_config.dataset == 'cifar10':
      self.dataset = 'cifar10-valid'
      self.val_name = 'x-valid'
      self.test_name = 'ori-test'
    else:
      self.dataset = self.bench_config.dataset
      self.val_name = 'x-valid'
      self.test_name = 'x-test'
  
  @staticmethod
  def str2config(arch_str):
    arch_str = arch_str[1:-1]
    arch_str = arch_str.replace("+|","")
    arch_str = arch_str.split("|")
    ops = ['input']
    matrix = ADJ_MATRIX
    for op_str in arch_str:
      ops.append(op_str[:-2])
    ops += ['output']
    return {"normal":{'matrix':matrix, 'ops':ops}}

  @staticmethod
  def config2str(config):
    arch_str = ""
    ops = config['normal']['ops'][1:-1]
    i = 0
    for out_block in range(3):
      for in_block in range(out_block+1):
        op_str = f"{ops[i]}~{in_block}"
        arch_str += f"|{op_str}"
        i+= 1
      arch_str += "|+"
    return arch_str[:-1]

  def query_by_config(self, config):
    arch_str = self.config2str(config)
    arch_index = self.api.archstr2index[arch_str]

    metadata = self.api.query_meta_info_by_index(arch_index, hp='200')
    nparam = metadata.get_compute_costs('cifar10-valid')['params']
    nparam = nparam * 1e+6

    val_acc = metadata.get_metrics(self.dataset, self.val_name)['accuracy']
    test_acc = metadata.get_metrics(self.dataset, self.test_name)['accuracy']

    train_time = self.api.get_more_info(arch_index, 'cifar10-valid', None)['train-per-time']

    return {'nparam':nparam, 'val_acc':val_acc, 'test_acc':test_acc, 'training_time':train_time}

  def config2graph(self, config):
    node_feature = []
    for op in config['normal']['ops']:
      op_index = OP_LIST.index(op)
      node_feature.append(torch.eye(len(OP_LIST)).type(torch.LongTensor)[op_index])
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
    #config to nb201_config
    arch_str = self.config2str(config)
    arch_index = self.api.archstr2index[arch_str]
    np201_config = self.api.get_net_config(arch_index, 'cifar10-valid')

    np201_config['num_classes'] = self.num_classes
    model = get_cell_based_tiny_net(np201_config)

    return model

  def sample_configuration(self, num_sample):
    arch_index_list = np.random.choice(list(range(15625)), num_sample)
    configs = []
    for arch_index in arch_index_list:
      arch_str = self.api.arch(arch_index)
      spec_dict = self.str2config(arch_str)
      configs.append(spec_dict)
    return configs