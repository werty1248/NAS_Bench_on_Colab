#메인 아이디어: NB201은 인덱스를 통한 관리를 한다
#ModelSpec <-> str <-> index
import argparse
import numpy as np

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
from torch_geometric.data import Data
from TransNASBench.api import TransNASBenchAPI as API
import TransNASBench
from TransNASBench.tools.utils import setup_model, setup_config

from .base import NASBenchAPIBase

class TransNASBenchAPI(NASBenchAPIBase):
  NUM_VERTICES = 8
  MAX_EDGES = 10
  TASK_LIST = ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']
  OP_LIST = ['input', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'output']
  METRIC_DICT = {'class_scene':'top1', 'class_object':'top1', 'room_layout':'loss',
                'jigsaw':'top1', 'segmentsemantic':'mIoU', 'normal':'ssim', 'autoencoder':'ssim'}

  def __init__(self, bench_config):
    super(TransNASBenchAPI, self).__init__()

    self.bench_config = bench_config
    
    self.task_name = bench_config.dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_dir', type=str, help='directory containing config.py file')
    parser.add_argument('--encoder_str', type=str, default='resnet50')
    parser.add_argument('--nopause', dest='nopause', action='store_true')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--ddp', action='store_true')
    parser.set_defaults(nopause=True)
    self.args = parser.parse_args(args=[f'TransNASBench/configs/train_from_scratch/{self.task_name}'])

    self.filepath = os.path.join(self.bench_config.filepath, 'transnas-bench_v10141024.pth')
    self.api = API(self.filepath)

  @staticmethod
  def str2config(arch_str):
    arch_str = arch_str.split("-")[-1]
    arch_str = arch_str.replace("_","")
    ops = ['input']
    matrix = ADJ_MATRIX
    for op_str in arch_str:
      ops.append(TransNASBenchAPI.OP_LIST[int(op_str) + 1])
    ops += ['output']
    return {"normal":{'matrix':matrix, 'ops':ops}}

  @staticmethod
  def config2str(config):
    arch_str = "64-41414-"
    ops = config['normal']['ops'][1:-1]
    i = 0
    for out_block in range(3):
      for in_block in range(out_block+1):
        op_str = TransNASBenchAPI.OP_LIST.index(ops[i]) - 1
        arch_str += str(op_str)
        i+= 1
      arch_str += "_"
    return arch_str[:-1]

  def query_by_config(self, config):
    arch_str = self.config2str(config)
    task_metric = TransNASBenchAPI.METRIC_DICT[self.task_name]
    valid_name = f'valid_{task_metric}'

    nparam = self.api.get_model_info(arch_str, self.task_name, 'model_params')

    val_acc = self.api.get_best_epoch_status(arch_str, self.task_name, metric=valid_name)[valid_name]
    test_acc = val_acc

    train_time = self.api.get_epoch_status(arch_str, self.task_name, epoch=-1)['time_elapsed']

    return {'nparam':nparam, 'val_acc':val_acc, 'test_acc':test_acc, 'training_time':train_time}

  def config2graph(self, config):
    node_feature = []
    for op in config['normal']['ops']:
      op_index = self.OP_LIST.index(op)
      node_feature.append(torch.eye(len(self.OP_LIST)).type(torch.LongTensor)[op_index])
    node_feature = torch.stack(node_feature)

    graph = Data(x=node_feature, edge_index=torch.tensor(config['normal']['matrix']).nonzero().t().contiguous())
    return {'normal':graph}

  def graph2config(self, graph):
    matrix = torch.sparse.FloatTensor(graph.edge_index, torch.ones(self.NUM_VERTICES).type(torch.int8), torch.Size([self.NUM_VERTICES,self.NUM_VERTICES])).to_dense().numpy()
    op_index_list = graph.x.nonzero().t().contiguous()[1,:]
    ops = []
    for op_index in op_index_list:
      ops.append(self.OP_LIST[op_index])
    return {'normal':{'matrix':matrix, 'ops':ops}}

  def get_model(self, config):
    cfg = setup_config(self.args, 0)
    arch_str = self.config2str(config)
    cfg['warmup_epochs'] = None
    cfg['encoder_str'] = arch_str
    model = setup_model(cfg, ['cpu',], 1, ddp=self.args.ddp)
    return model


  def sample_configuration(self, num_sample):
    arch_index_list = np.random.choice(list(range(3256, 3256 + 4096)), num_sample)
    configs = []
    for arch_index in arch_index_list:
      arch_str = self.api.index2arch(arch_index)
      spec_dict = self.str2config(arch_str)
      configs.append(spec_dict)
    return configs
