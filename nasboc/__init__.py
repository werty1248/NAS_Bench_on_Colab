from .base import NASBenchConfig

def get_nas_bench(bench_config: NASBenchConfig):
  name = bench_config.benchmark
  if name == 'NAS-Bench-101':
    from .nas_bench_101 import NASBench101API
    return NASBench101API(bench_config)
  elif name == 'NAS-Bench-201':
    from .nas_bench_201 import NASBench201API
    return NASBench201API(bench_config)
  elif name == 'NAS-Bench-301':
    from .nas_bench_301 import NASBench301API
    return NASBench301API(bench_config)
  elif name in ['NDS-DARTS','NDS-Amoeba','NDS-ENAS','NDS-NASNet','NDS-PNAS']:
    from .nds import NDSAPI
    return NDSAPI(bench_config)
  else:
    raise KeyError(name)