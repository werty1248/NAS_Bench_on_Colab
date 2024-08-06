from graph_hash import config_hash
from utils import set_seed
from tqdm.auto import tqdm
import numpy as np
import random

class LocalSearch():
  def __init__(self, search_space, seed = 0, use_memory = False, use_isomorphism = False):
    self.search_space = search_space
    self.use_isomorphism = use_isomorphism
    self.use_memory = use_memory
    if use_isomorphism:
      self.hash_function = lambda x : config_hash(x, self.search_space)
      self.build_oracle_hash()
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

  def build_oracle_hash(self):
    self.hash_table = {}
    iter = tqdm(self.search_space.sample_all())
    for config in iter:
      query = self.search_space.query_by_config(config)
      hash = config_hash(config, self.search_space)
      if hash not in self.hash_table:
        self.hash_table[hash] = []
      self.hash_table[hash].append(config)
      iter.set_description(str(len(self.hash_table)))
      
    self.inv_hash_table = {}
    for key, value in self.hash_table.items():
      for config in value:
        self.inv_hash_table[str(config)] = key

  def get_all_neighbour(self, config):    
    hash = self.inv_hash_table[str(config)]
    isomorphic_configs = self.hash_table[hash]
    all_neighbour_hashs = []
    for config in isomorphic_configs:
      neighbour_configs = self.search_space.get_neighbours(config)
      for neighbour_config in neighbour_configs:
        all_neighbour_hashs.append(self.inv_hash_table[str(neighbour_config)])
    all_neighbour_hashs = set(all_neighbour_hashs)
  
    unique_configs = []
    for neighbour_hash in all_neighbour_hashs:
      unique_configs.append(random.choice(self.hash_table[neighbour_hash]))
  
    return unique_configs

    
  def search(self, max_time, num_initial = 10):
    set_seed(self.seed)
    iter = tqdm(range(max_time))
    for _ in range(num_initial):
      new_arch = self.search_space.sample_configuration(1)[0]
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
    
    while self.total_time < max_time:
        if self.use_isomorphism:
          neighs = self.get_all_neighbour(start)
        else:
          neighs = self.search_space.get_neighbours(start)
      
        new_start = start
        secondary_config = '@@@'
        secondary = 0
        for neigh in (np.random.permutation(neighs)):
          g_hash = self.hash_function(neigh)
          if g_hash not in self.search_hash or not self.use_memory:
            res = self.search_space.query_by_config(neigh)
            self.total_search_count += 1
            new_acc = res['val_acc']
            iter.set_description("{}-{:.3f}".format(self.total_search_count, new_acc))
            self.search_hash[g_hash] = new_acc
            self.total_time += res['training_time']
          else:
            new_acc = self.search_hash[g_hash]
  
          if new_acc > self.base_acc:
            self.base_acc = new_acc
            new_start = neigh
            break
  
          if new_acc > secondary and g_hash not in self.search_hash:
            secondary = new_acc
            secondary_config = neigh
  
          if self.total_time > max_time:
            break
  
        if self.total_time <= max_time:
          self.history.append([self.total_time, self.base_acc])
  
        if str(start) == str(new_start):
          start = secondary_config
          self.base_acc = secondary
  
        start = new_start
    return self.history
