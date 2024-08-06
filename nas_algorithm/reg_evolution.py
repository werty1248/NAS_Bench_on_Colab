from graph_hash import config_hash
from utils import set_seed
from tqdm.auto import tqdm
import numpy as np
import random

class RegEvolution():
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

    
  def search(self, max_time, num_initial = 10, num_population = 30, num_sample = 10):
    set_seed(self.seed)
    iter = tqdm(range(max_time))
    population = []
    
    for _ in range(min(num_initial, num_population)):
      new_arch = self.search_space.sample_configuration(1)[0]
      res = self.search_space.query_by_config(new_arch)
      self.total_search_count += 1
      acc = res['val_acc']
      self.total_time += res['training_time']
      self.search_hash[self.hash_function(new_arch)] = acc
      population.append([acc, new_arch])
    
      if self.total_time <= max_time:
        self.history.append([self.total_time, acc])
    
    while self.total_time < max_time:
      target_archs = np.array(population[-num_population:])
      target_archs = np.random.permutation(target_archs)[:num_sample]
      start = 0
      best_acc = 0
      for target_arch in target_archs:
        if target_arch[0] > best_acc:
          start = target_arch[1]
          best_acc = target_arch[0]
          
      if self.use_isomorphism:
        neighs = self.get_all_neighbour(start)
      else:
        neighs = self.search_space.get_neighbours(start)
      neigh = np.random.choice(neighs)

      new_acc = None
      g_hash = self.hash_function(neigh)
      if g_hash not in self.search_hash or not self.use_memory:
        res = self.search_space.query_by_config(neigh)
        self.total_search_count += 1
        new_acc = res['val_acc']
        iter.set_description("{}-{:.3f}".format(self.total_search_count, new_acc))
        self.search_hash[g_hash] = new_acc
        self.total_time += res['training_time']
        population.append([new_acc, neigh])

      if not new_acc:
          continue
      if self.total_time <= max_time:
        self.history.append([self.total_time, new_acc])
  
    return self.history
