import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from collections import namedtuple
from models.cell_searchs.genotypes import Structure
from copy import deepcopy
from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.anynet import AnyNet
from pycls.models.nas.genotypes import GENOTYPES, Genotype

from .base import NASBenchAPIBase

INPUT1 = 'input_1'
INPUT2 = 'input_2'
OUTPUT = 'output'

Amoeba_OP_LIST  =[INPUT1, INPUT2, 'avg_pool_3x3', 'conv_7x1_1x7', 'dil_sep_conv_3x3', 'max_pool_3x3', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 'skip_connect', OUTPUT]
DARTS_OP_LIST   =[INPUT1, INPUT2, 'avg_pool_3x3', 'dil_conv_3x3', 'dil_conv_5x5', 'max_pool_3x3', 'none', 'sep_conv_3x3', 'sep_conv_5x5', 'skip_connect', OUTPUT]
ENAS_OP_LIST    =[INPUT1, INPUT2, 'avg_pool_3x3', 'max_pool_3x3', 'sep_conv_3x3', 'sep_conv_5x5', 'skip_connect', OUTPUT]
NASNet_OP_LIST  =[INPUT1, INPUT2, 'avg_pool_3x3', 'conv_1x1', 'conv_3x1_1x3', 'conv_3x3', 'conv_7x1_1x7', 'dil_conv_3x3', 'max_pool_3x3', 'max_pool_5x5', 'max_pool_7x7', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 'skip_connect', OUTPUT]
PNAS_OP_LIST    =[INPUT1, INPUT2, 'avg_pool_3x3', 'conv_7x1_1x7', 'dil_conv_3x3', 'max_pool_3x3', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 'skip_connect', OUTPUT]

OP_LIST = {'Amoeba':Amoeba_OP_LIST,
           'DARTS':DARTS_OP_LIST,
           'ENAS':ENAS_OP_LIST,
           'NASNet':NASNet_OP_LIST,
           'PNAS':PNAS_OP_LIST,
           }
           
NUM_VERTICES = {'Amoeba':13,
                'DARTS':11,
                'ENAS':13,
                'NASNet':13,
                'PNAS':13,
                }

genotype = {'normal': [['avg_pool_3x3', 0], ['conv_7x1_1x7', 1], ['sep_conv_3x3', 2], ['sep_conv_5x5', 0], ['dil_sep_conv_3x3', 2], ['dil_sep_conv_3x3', 2], ['skip_connect', 4], ['dil_sep_conv_3x3', 4], ['conv_7x1_1x7', 2], ['sep_conv_3x3', 4]], 'normal_concat': [3, 5, 6], 'reduce': [['avg_pool_3x3', 0], ['dil_sep_conv_3x3', 1], ['sep_conv_3x3', 0], ['sep_conv_3x3', 0], ['skip_connect', 2], ['sep_conv_7x7', 0], ['conv_7x1_1x7', 4], ['skip_connect', 4], ['conv_7x1_1x7', 0], ['conv_7x1_1x7', 5]], 'reduce_concat': [3, 6]}

search_space_name = 'Amoeba'

op_list = OP_LIST[search_space_name]

def _cell_to_config(edge_list, concat, search_space_name):
  num_vertices = NUM_VERTICES[search_space_name]

  ops = [INPUT1, INPUT2]
  adj_matrix = np.zeros((num_vertices, num_vertices), dtype=np.int64)

  #Edges to output
  for c_index in concat:
    adj_matrix[c_index*2-2, -1] = 1
    adj_matrix[c_index*2-1, -1] = 1

  #Inner edges
  for out_index, (op, in_index) in enumerate(edge_list):
    ops.append(op)
    if in_index <= 1:
      adj_matrix[in_index, out_index + 2] = 1
    else:
      adj_matrix[in_index*2-2, out_index + 2] = 1
      adj_matrix[in_index*2-1, out_index + 2] = 1

  ops.append(OUTPUT)
  return {'matrix':adj_matrix, 'ops':ops}

def _config_to_cell(config, search_space_name):
  num_vertices = NUM_VERTICES[search_space_name]

  edge_list = []
  concat = []

  ops = config['ops']
  adj_matrix = config['matrix']

  for idx in range(2, num_vertices - 1):
    in_index = np.where(adj_matrix[:,idx] == 1)[0][0]
    if in_index >= 2:
      in_index = in_index // 2 + 1
    edge_list.append([ops[idx], in_index])
  
  output_edge = np.where(adj_matrix[:,-1] == 1)[0]
  for edge in output_edge:
    if edge >= 2:
      edge = edge // 2 + 1
    concat.append(edge)
  
  concat = sorted(set(concat))
  
  return edge_list, concat

def genotype_to_config(genotype, search_space_name):  
  if isinstance(genotype, dict):
    genotype = Genotype(**genotype)
  elif not isinstance(genotype, Genotype):
    raise NotImplementedError(type(genotype))

  normal_cell, normal_concat = genotype.normal, genotype.normal_concat
  normal_config = _cell_to_config(normal_cell, normal_concat, search_space_name)
  
  reduce_cell, reduce_concat = genotype.reduce, genotype.reduce_concat
  reduce_config = _cell_to_config(reduce_cell, reduce_concat, search_space_name)

  return {"normal":normal_config, "reduce":reduce_config}

def config_to_genotype(config, search_space_name):
  normal_cell, normal_concat = _config_to_cell(config['normal'], search_space_name)
  reduce_cell, reduce_concat = _config_to_cell(config['reduce'], search_space_name)

  return Genotype(normal = normal_cell, normal_concat = normal_concat,
                  reduce = reduce_cell, reduce_concat = reduce_concat)


class NDSAPI(NASBenchAPIBase):
  def __init__(self, bench_config):
    super(NDSAPI, self).__init__()

    self.bench_config = bench_config
    self.search_space = bench_config.benchmark.split("-")[-1]

    if bench_config.dataset == 'ImageNet':
      self.filepath = os.path.join(self.bench_config.filepath, 'nds_data', self.search_space + "_in.json")
    elif bench_config.dataset == 'cifar10':
      self.filepath = os.path.join(self.bench_config.filepath, 'nds_data', self.search_space + ".json")
    else:
      raise KeyError(bench_config.dataset)

    data = json.load(open(self.filepath, 'r'))

    try:
      data = data['top'] + data['mid']
    except Exception as e:
      pass

    self.data = data
    
    self._genotype2uid_map = {}
    for uid, net in enumerate(self.data):
      key = str(Genotype(**net['net']['genotype']))
      self._genotype2uid_map[key] = uid
  
  def _genotype2uid(self, genotype):
    if isinstance(genotype, dict):
      genotype = Genotype(**genotype)
    elif not isinstance(genotype, Genotype):
      raise NotImplementedError(type(genotype))

    return self._genotype2uid_map[str(genotype)]

  def query_by_config(self, config):
    genotype = config_to_genotype(config, self.search_space)
    uid = self._genotype2uid(genotype)
    
    nparam = self.data[uid]['params']

    #Approximated training time = fwbw * num_mini_batch * epoch
    train_time = self.data[uid]['prec_time']['train_fw_bw'] * 100 * 60000 / 128

    val_acc = 100. - self.data[uid]['test_ep_top1'][-1]
    test_acc = 100. - self.data[uid]['test_ep_top1'][-1]

    return {'nparam':nparam, 'val_acc':val_acc, 'test_acc':test_acc, 'training_time':train_time}

  def config2graph(self, config):
    graphs = {}
    op_list = OP_LIST[self.search_space]
    for cell_name in ['normal','reduce']:
      node_feature = []
      for op in config[cell_name]['ops']:
        op_index = op_list.index(op)
        node_feature.append(torch.eye(len(op_list)).type(torch.LongTensor)[op_index])
      node_feature = torch.stack(node_feature)

      graph = Data(x=node_feature, edge_index=torch.tensor(config[cell_name]['matrix']).nonzero().t().contiguous())
      graphs[cell_name] = graph
    return graphs

  def graph2config(self, graphs):
    config = {}
    num_vertices = NUM_VERTICES[self.search_space]
    for cell_name, graph in graphs.item():
      matrix = torch.sparse.FloatTensor(graph.edge_index, torch.ones(num_vertices).type(torch.int8), torch.Size([num_vertices,num_vertices])).to_dense().numpy()
      op_index_list = graph.x.nonzero().t().contiguous()[1,:]
      ops = []
      for op_index in op_index_list:
        ops.append(OP_LIST[op_index])
      config[cell_name] = {'matrix':matrix, 'ops':ops}

    return config

  def get_model(self, config):
    genotype = config_to_genotype(config, self.search_space)
    uid = self._genotype2uid(genotype)
    
    netinfo = self.data[uid]
    config = netinfo['net']
    #print(config)
    if 'genotype' in config:
      #print('geno')
      gen = config['genotype']
      genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
      if '_in' in self.search_space:
        network = NetworkImageNet(config['width'], 120, config['depth'], config['aux'],  genotype)
      else:
        network = NetworkCIFAR(config['width'], 10, config['depth'], config['aux'],  genotype)
      network.drop_path_prob = 0.
      #print(config)
      #print('genotype')
      L = config['depth']
    else:
      if 'bot_muls' in config and 'bms' not in config:
        config['bms'] = config['bot_muls']
        del config['bot_muls']
      if 'num_gs' in config and 'gws' not in config:
        config['gws'] = config['num_gs']
        del config['num_gs']
      config['nc'] = 1
      config['se_r'] = None
      config['stem_w'] = 12
      L = sum(config['ds'])
      if 'ResN' in self.searchspace:
        config['stem_type'] = 'res_stem_in'
      else:
        config['stem_type'] = 'simple_stem_in'
      #"res_stem_cifar": ResStemCifar,
      #"res_stem_in": ResStemIN,
      #"simple_stem_in": SimpleStemIN,
      if config['block_type'] == 'double_plain_block':
        config['block_type'] = 'vanilla_block'
      network = AnyNet(**config)
    #return_feature_layer(network)
    return network

  def sample_configuration(self, num_sample):
    arch_list = np.random.choice(self.data, num_sample)
    configs = []
    for arch_info in arch_list:
      arch_config = genotype_to_config(arch_info['net']['genotype'], self.search_space)
      configs.append(arch_config)
    return configs