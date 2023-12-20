import json

class NASBenchConfig():
  @staticmethod
  def from_dict(arg_dict):
    config = NASBenchConfig()
    config.update(arg_dict)
    return config

  @staticmethod
  def from_file(file_path):
    with open(file_path, 'r') as f:
      arg_dict = json.load(f)
    NASBenchConfig.from_dict(arg_dict)

  def update(self, arg_dict):
    self.benchmark = arg_dict['benchmark']
    self.dataset = arg_dict['dataset']
    self.filepath = arg_dict['filepath']

class NASBenchAPIBase():
  space = None
  def query_by_config(self, config):
    raise NotImplementedError

  def config2graph(self, config):
    raise NotImplementedError

  def graph2config(self, graph):
    raise NotImplementedError

  def get_model(self, config):
    raise NotImplementedError