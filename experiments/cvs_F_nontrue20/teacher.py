"""Frozen historical CSIL; does not consume training RNG or join student state."""
from pathlib import Path
import torch
from . import design as d
from .native_factory import load_factory
from experiments.cvs_sameiq_experts.protocol import provenance,CLASSES

def load(device):
 root=d.TEACHER;contract=d.read(root/'source_contract.json');init=d.read(root/'initialization.json');done=d.read(root/'completion.json');cfg=d.read(root/'resolved_config.json')
 provenance('csil',d.read(d.SOURCE),contract,init,done,cfg)
 if not d.read(d.TEACHER_QUAL)['qualification']['pass_both_draws']:raise ValueError('Fixed source qualification failed')
 cp=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
 if cp['source_contract']['role_ids']!=contract['role_ids'] or cp['classes']!=list(CLASSES) or cp['selection']!='fixed_last_epoch' or cp['epoch']!=200 or cp['initialization']!=init or cp['config']!=cfg:raise ValueError('Teacher payload provenance differs')
 release=Path(d.PROJECT)/'releases/native_comparisons_20260930_r01'
 devices=[] if torch.device(device).type=='cpu' else [torch.device(device).index or 0]
 with torch.random.fork_rng(devices=devices):
  api=load_factory(release)
  m=api.factory('csil',cp['num_classes'],cp['num_receivers'],**cp['model_kwargs']);m.load_state_dict(cp['model'],strict=True);m.to(device).eval().requires_grad_(False)
 if any(not torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('Nonfinite teacher')
 return m
