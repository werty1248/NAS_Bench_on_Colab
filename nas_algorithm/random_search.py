from graph_hash import config_hash
from utils import set_seed
from tqdm.auto import tqdm
import numpy as np
import random

class RandomSearch():
  def __init__(self, search_space, seed = 0, use_memory = False, use_isomorphism = False):
    self.search_space = search_space
    self.use_isomorphism = use_isomorphism
    self.use_memory = use_memory
    if use_isomorphism:
      self.hash_function = lambda x : config_hash(x, self.search_space)
    else:
      self.hash_function = str
    self.initialize(seed)

  def initialize(self, seed = 0):
    self.total_time = 0
    self.history = []
    self.search_hash = {}
    self.total_search_count = 0

    self.base_acc = 0
    self.seed = seed
    
  def search(self, max_time, num_initial = 10):
    set_seed(self.seed)
    iter = tqdm(range(max_time))
    while self.total_time < max_time:
      new_arch = self.search_space.sample_configuration(1)[0]
      g_hash = self.hash_function(new_arch)
      if g_hash not in self.search_hash or not self.use_memory:
        res = self.search_space.query_by_config(new_arch)
        self.total_search_count += 1
        acc = res['val_acc']
        self.total_time += res['training_time']
        self.search_hash[self.hash_function(new_arch)] = acc
        if acc > self.base_acc:
          self.base_acc = acc
          start = new_arch
    
      if self.total_time <= max_time:
        self.history.append([self.total_time, self.base_acc])
    return self.history
