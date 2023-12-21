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

from .base import NASBenchAPIBase
from .utils import config_to_genotype, genotype_to_config

INPUT1 = 'input_1'
INPUT2 = 'input_2'
OUTPUT = 'output'
AVGPOOL3X3 = 'avg_pool_3x3'
MAXPOOL3X3 = 'max_pool_3x3'
MAXPOOL5X5 = 'max_pool_5x5'
MAXPOOL7X7 = 'max_pool_7x7'
CONV1X1 = 'conv_1x1'
CONV3X3 = 'conv_3x3'
CONV3X1_1X3 = 'conv_3x1_1x3'
CONV7X1_1X7 = 'conv_7x1_1x7'
SEPCONV3X3 = 'sep_conv_3x3'
SEPCONV5X5 = 'sep_conv_5x5'
SEPCONV7X7 = 'sep_conv_7x7'
DILCONV3X3 = 'dil_conv_3x3'
DILCONV5X5 = 'dil_conv_5x5'
DILSEPCONV5X5 = 'dil_sep_conv_3x3'
SKIP_CONNECT = 'skip_connect'
NONE = 'none'

Amoeba_OP_LIST  =[INPUT1, INPUT2, AVGPOOL3X3, MAXPOOL3X3, SKIP_CONNECT, SEPCONV3X3, SEPCONV5X5, SEPCONV7X7, CONV7X1_1X7, DILSEPCONV5X5, OUTPUT]
DARTS_OP_LIST   =[INPUT1, INPUT2, AVGPOOL3X3, MAXPOOL3X3, SKIP_CONNECT, SEPCONV3X3, SEPCONV5X5, DILCONV3X3, DILCONV5X5, NONE, OUTPUT]
ENAS_OP_LIST    =[INPUT1, INPUT2, AVGPOOL3X3, MAXPOOL3X3, SKIP_CONNECT, SEPCONV3X3, SEPCONV5X5, OUTPUT]
NASNet_OP_LIST  =[INPUT1, INPUT2, AVGPOOL3X3, MAXPOOL3X3, MAXPOOL5X5, MAXPOOL7X7, SKIP_CONNECT, SEPCONV3X3, SEPCONV5X5, SEPCONV7X7, CONV1X1, CONV3X3, CONV3X1_1X3, CONV7X1_1X7, DILCONV3X3, OUTPUT]
PNAS_OP_LIST    =[INPUT1, INPUT2, AVGPOOL3X3, MAXPOOL3X3, SKIP_CONNECT, SEPCONV3X3, SEPCONV5X5, SEPCONV7X7, CONV7X1_1X7, DILCONV3X3, OUTPUT]

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

NUM_CLASSES = 10
NUM_CLASSES_IN = 120

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
    genotype = config_to_genotype(config, NUM_VERTICES[self.search_space], OP_LIST[self.search_space])
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
    genotype = config_to_genotype(config, NUM_VERTICES[self.search_space], OP_LIST[self.search_space])
    uid = self._genotype2uid(genotype)
    
    netinfo = self.data[uid]
    config = netinfo['net']
    #print(config)
    if 'genotype' in config:
      #print('geno')
      gen = config['genotype']
      genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
      if '_in' in self.search_space:
        network = NetworkImageNet(config['width'], NUM_CLASSES_IN, config['depth'], config['aux'],  genotype)
      else:
        network = NetworkCIFAR(config['width'], NUM_CLASSES, config['depth'], config['aux'],  genotype)
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
      arch_config = genotype_to_config(arch_info['net']['genotype'], NUM_VERTICES[self.search_space], OP_LIST[self.search_space])
      configs.append(arch_config)
    return configs