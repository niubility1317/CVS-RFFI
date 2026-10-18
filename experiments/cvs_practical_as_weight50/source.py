"""Strict own weighted30 parent, including its verified original20 ancestor."""
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata,check_default_head,check_parent_budget
from experiments.cvs_crossview_supcon41 import source as backend

def ancestor_payload(c):
 d.validate(c)
 from experiments.cvs_practical_as_weight30.source import parent_payload as original20
 return original20(d.parent_config(c))

def parent_payload(c):
 d.validate(c);expected=d.parent_config(c);ancestor=ancestor_payload(c);out=Path(c['parent_source'])
 if Path(expected['output_root'])!=out:raise ValueError('Only own weighted FINAL30 path')
 check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),expected,d.read(d.SOURCE)['role_ids'],10,2220)
 result=d.read(out/'result.json')
 check_parent_budget(expected,result)
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,run=d.PARENT_RUN,row=c['row_id'],epoch=10,updates=2220,baseline=expected['parent_source']+'/final_ssdg.pth')
 keys=set(ancestor['model'])
 if set(cp['model'])!=keys or set(cp['ema_model'])!=keys or len(keys)!=206:raise ValueError('Finite own weighted30 student/EMA keys differ')
 check_default_head(cp['model']['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape)
 for k in keys:
  a,b=cp['model'][k],cp['ema_model'][k]
  if a.shape!=ancestor['model'][k].shape or a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Finite own weighted30 state shapes differ')
 if ancestor['args']['epochs']!=20 or cp['args'].get('a1_ema_startup_average',False):raise ValueError('Own original20 augmentation/EMA policy differs')
 return cp

def segment_train(c,design=None,model_api=None,observer=None):
 previous=backend.parent_payload;backend.parent_payload=parent_payload
 try:result=backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=previous
 out=Path(c['output_root']);done=d.read(out/'completion.json');done.update(initialization='own_WEIGHTED_FINAL30+20/resetAdamWcos20',cumulative_epochs=50,cumulative_optimizer_steps=11100,parent_epochs=30,parent_optimizer_steps=6660);d.write(out/'completion.json',done)
 return result
