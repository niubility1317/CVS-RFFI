"""Unchanged baseline model; actual F factory seed must follow the row."""
from contextlib import contextmanager
import importlib.util
from .runner import REFERENCE
from . import design as d
from .channel import Practical
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
 if c['As']:m._full_channel=Practical()
 m._replication_factory_seed=c['model_seed'];return m

def build(c,device='cpu'):
 with factory_seed(c):m=impl.build(c,device)
 if c['As']:m._full_channel=Practical()
 m._replication_factory_seed=c['model_seed'];return m

def own_model(c,device='cpu'):
 import torch
 from pathlib import Path
 d.validate(c);p=Path(c['output_root']);done=d.read(p/'completion.json');frozen=d.read(p/'source_frozen.json');init=d.read(p/'initialization.json')
 if done['config']!=c or done['epoch']!=30 or done['optimizer_steps']!=6660 or done['target_access'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Own single30 exactsource provenance differs')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);a=cp['args']
 if cp['epoch']!=30 or cp['a1_ema_successful_updates']!=6660 or cp['run_id']!=d.RUN or cp['candidate_id']!=c['row_id'] or a['epochs']!=30 or a['label_epochs']!=13 or a['pseudo_epochs']!=17 or a['baseline_ckpt'] or a['teacher_ckpt'] or a['seed']!=c['model_seed'] or cp['checkpoint_selection']!='final_only':raise ValueError('Own native final30 identity differs')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
