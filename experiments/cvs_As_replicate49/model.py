"""Unchanged baseline model; actual F factory seed must follow the row."""
from contextlib import contextmanager
import importlib.util
from .runner import REFERENCE
from . import design as d
s=importlib.util.spec_from_file_location(__package__+'._model_impl',REFERENCE/'experiments/cvs_noise_shape_factorial43/model.py');impl=importlib.util.module_from_spec(s);s.loader.exec_module(impl)
classifier,source_domain_count=impl.classifier,impl.source_domain_count
@contextmanager
def factory_seed(c):
 d.validate(c)
 from experiments.cvs_fullsource_concat20 import design as full
 prior=full.SEED;full.SEED=c['model_seed']
 try:
  cfg=full.base_model_config()
  if cfg['model_seed']!=c['model_seed'] or cfg['arm']!='F':raise ValueError('Actual F factory seed differs')
  yield
 finally:full.SEED=prior

def install(m,c):
 with factory_seed(c):m=impl.install(m,c)
 m._replication_factory_seed=c['model_seed'];return m

def build(c,device='cpu'):
 with factory_seed(c):m=impl.build(c,device)
 m._replication_factory_seed=c['model_seed'];return m

def own_model(c,device='cpu'):
 with factory_seed(c):return impl.own_model(c,device)
