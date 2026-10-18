"""Reuse fixed E50 complete telemetry and verify actual row native/F factory seeds."""
import importlib.util
import torch
from . import design as d,model
s=importlib.util.spec_from_file_location(__package__+'._observer_impl',d.ROOT/'experiments/cvs_practical_budget50/observer.py');impl=importlib.util.module_from_spec(s);s.loader.exec_module(impl)
class Observer(impl.Observer):
 def initialize(self,m,c,out):
  super().initialize(m,c,out)
  if m._replication_factory_seed!=c['model_seed']:raise ValueError('F factory did not follow row seed')
  d.write(out/'seed_binding.json',dict(model_seed=c['model_seed'],actual_F_factory_seed=m._replication_factory_seed,physical_roles='EXACT_MATCH',physical_split=392005,original_channel_seeds_unchanged=True,legacy_recipe_reference_not_initializer=True))
 def finish(self):
  cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if cp['args']['seed']!=self.c['model_seed']:raise ValueError('Actual native training seed differs')
  return super().finish()
def train(c):
 from .source import segment_train
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
