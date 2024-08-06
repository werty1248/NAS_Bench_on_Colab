import hashlib
import numpy as np

# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def hash_module(matrix, labeling):
  """Computes a graph-invariance MD5 hash of the matrix and label pair.

  Args:
    matrix: np.ndarray square upper-triangular adjacency matrix.
    labeling: list of int labels of length equal to both dimensions of
      matrix.

  Returns:
    MD5 hash of the matrix and labeling.
  """
  vertices = np.shape(matrix)[0]
  in_edges = np.sum(matrix, axis=0).tolist()
  out_edges = np.sum(matrix, axis=1).tolist()

  assert len(in_edges) == len(out_edges) == len(labeling)
  hashes = list(zip(out_edges, in_edges, labeling))
  hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
  # Computing this up to the diameter is probably sufficient but since the
  # operation is fast, it is okay to repeat more times.
  for _ in range(vertices // 2):
    new_hashes = []
    for v in range(vertices):
      in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
      out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
      new_hashes.append(hashlib.md5(
          (''.join(sorted(in_neighbors)) + '|' +
           ''.join(sorted(out_neighbors)) + '|' +
           hashes[v]).encode('utf-8')).hexdigest())
    hashes = new_hashes
  fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

  return fingerprint

#############################################################################

import torch
import torch_geometric

import networkx as nx
import matplotlib.pyplot as plt

ZEROIZE = 'none'
SKIP_CONN = 'skip_connect'

def hash_by_graph(graph):
  full_hash = ""
  for key, g in graph.items():
    g = torch_geometric.utils.from_networkx(g)
    matrix = torch.sparse.FloatTensor(g.edge_index, torch.ones(len(graph[key].edges)).type(torch.int8), torch.Size([g.num_nodes,g.num_nodes])).to_dense()
    op_index_list = g['name']
    hash = hash_module(matrix.numpy(), op_index_list)
    full_hash += hash
  return full_hash

def remove_zero_and_skip(g, zeroize = ZEROIZE, skip_conn = SKIP_CONN):
  mul_g = nx.MultiDiGraph(g)
  for node_idx in list(mul_g.nodes):
    node = mul_g.nodes[node_idx]
    if node['name'] == zeroize:
      mul_g.remove_node(node_idx)
    elif node['name'] == skip_conn:
      in_nodes = [x[0] for x in mul_g.in_edges(node_idx)]
      out_nodes = [x[1] for x in mul_g.out_edges(node_idx)]

      mul_g.remove_node(node_idx)

      for in_idx in in_nodes:
        for out_idx in out_nodes:
          mul_g.add_edge(in_idx, out_idx)

  new_node_idx = len(g.nodes)
  for st, ed, cnt in list(mul_g.edges):
    if cnt > 0:
      mul_g.remove_edge(st, ed, cnt)
      mul_g.add_node(new_node_idx, name = skip_conn)
      mul_g.add_edge(st, new_node_idx)
      mul_g.add_edge(new_node_idx, ed)
      new_node_idx += 1

  return nx.DiGraph(mul_g)

def remove_leaves(g):
  remove_nodes = []
  for node_idx in list(g.nodes):
    node_name = g.nodes[node_idx]['name']
    if len(g.in_edges(node_idx)) == 0 and 'input' not in node_name and 'output' not in node_name:
      remove_nodes.append((0, node_idx))
    if len(g.out_edges(node_idx)) == 0 and 'input' not in node_name and 'output' not in node_name:
      remove_nodes.append((1, node_idx))

  while(len(remove_nodes) > 0):
    rm_type, node_idx = remove_nodes.pop()
    if node_idx not in g.nodes:
      continue
    if rm_type == 0: # Add all out nodes
      out_nodes = [x[1] for x in g.out_edges(node_idx)]
      g.remove_node(node_idx)
      for node_idx2 in out_nodes:
        node_name = g.nodes[node_idx2]['name']
        if g.in_edges(node_idx2) == 0 and 'input' not in node_name and 'output' not in node_name:
          remove_nodes.append((0, node_idx))
    elif rm_type == 1:
      in_nodes = [x[0] for x in g.in_edges(node_idx)]
      g.remove_node(node_idx)
      for node_idx2 in in_nodes:
        if g.out_edges(node_idx2) == 0 and 'input' not in node_name and 'output' not in node_name:
          remove_nodes.append((0, node_idx))
    remove_nodes = []
    for node_idx in list(g.nodes):
      node_name = g.nodes[node_idx]['name']
      if len(g.in_edges(node_idx)) == 0 and 'input' not in node_name and 'output' not in node_name:
        remove_nodes.append((0, node_idx))
      if len(g.out_edges(node_idx)) == 0 and 'input' not in node_name and 'output' not in node_name:
        remove_nodes.append((1, node_idx))

  return g

def config_hash(config, search_space, zeroize = ZEROIZE, skip_conn = SKIP_CONN):
  g = search_space.config2graph(config)
  for key in g.keys():
    g[key] = torch_geometric.utils.to_networkx(g[key], to_undirected=False)
    for i in range(len(g[key].nodes)):
      g[key].nodes[i]['name'] = config[key]['ops'][i]
    g[key] = remove_zero_and_skip(g[key], zeroize, skip_conn)
    g[key] = remove_leaves(g[key])
  hash = hash_by_graph(g)
  return hash
