import numpy as np
from pycls.models.nas.genotypes import GENOTYPES, Genotype

def _cell_to_config(edge_list, concat, num_vertices, op_list):

  ops = [op_list[0], op_list[1]] #INPUT1, INPUT2
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

  ops.append(op_list[-1]) #OUTPUT
  return {'matrix':adj_matrix, 'ops':ops}

def _config_to_cell(config, num_vertices):

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

def genotype_to_config(genotype, num_vertices, op_list):  
  if isinstance(genotype, dict):
    genotype = Genotype(**genotype)
  elif not isinstance(genotype, Genotype):
    raise NotImplementedError(type(genotype))

  normal_cell, normal_concat = genotype.normal, genotype.normal_concat
  normal_config = _cell_to_config(normal_cell, normal_concat, num_vertices, op_list)
  
  reduce_cell, reduce_concat = genotype.reduce, genotype.reduce_concat
  reduce_config = _cell_to_config(reduce_cell, reduce_concat, num_vertices, op_list)

  return {"normal":normal_config, "reduce":reduce_config}

def config_to_genotype(config, num_vertices, op_list):
  normal_cell, normal_concat = _config_to_cell(config['normal'], num_vertices)
  reduce_cell, reduce_concat = _config_to_cell(config['reduce'], num_vertices)

  return Genotype(normal = normal_cell, normal_concat = normal_concat,
                  reduce = reduce_cell, reduce_concat = reduce_concat)

