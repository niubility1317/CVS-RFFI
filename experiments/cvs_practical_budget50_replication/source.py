"""Strict real seed02/03 own FINAL30 student/EMA and own scratch20 ancestry."""
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata
from experiments.cvs_crossview_supcon41 import source as backend

def ancestor_payload(c):
 d.validate(c)
 from experiments.cvs_As_replicate49 import source as s
 expected=d.reference_config(c,30);ancestor=s.parent_payload(expected)
 check_checkpoint(ancestor,seed=c['model_seed'],run=d.REFERENCE_RUN,row=c['row_id'],epoch=20,updates=4440,baseline='')
 return ancestor

def parent_payload(c):
 d.validate(c);expected=d.reference_config(c,30);ancestor=ancestor_payload(c);p=Path(c['parent_source'])
 if c['parent_source']!=expected['output_root']:raise ValueError('Actual seed02/03 fixedFINAL30 path differs')
 check_source_metadata(d.read(p/'completion.json'),d.read(p/'source_frozen.json'),d.read(p/'initialization.json'),d.read(p/'source_contract.json'),expected,d.read(d.SOURCE)['role_ids'],10,2220)
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,seed=c['model_seed'],run=d.REFERENCE_RUN,row=c['row_id'],epoch=10,updates=2220,baseline=expected['parent_source']+'/final_ssdg.pth')
 keys=set(ancestor['model'])
 if set(cp['model'])!=keys or set(cp['ema_model'])!=keys or len(keys)!=206 or cp['model']['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape!=(6,160):raise ValueError('Fixed student/ownEMA/defaultcos state keys or head differ')
 for key in keys:
  if cp['model'][key].shape!=ancestor['model'][key].shape or cp['ema_model'][key].shape!=cp['model'][key].shape or cp['ema_model'][key].dtype!=cp['model'][key].dtype or not torch.isfinite(cp['model'][key]).all() or not torch.isfinite(cp['ema_model'][key]).all():raise ValueError('Fixed finite student/ownEMA state shapes differ')
 if ancestor['args']['epochs']!=20 or ancestor['args'].get('a1_ema_startup_average',False) or cp['args'].get('a1_ema_startup_average',False):raise ValueError('Original own scratch20 augmentation/EMA policy differs')
 return cp

def segment_train(c,design=None,model_api=None,observer=None):
 previous=backend.parent_payload;backend.parent_payload=parent_payload
 try:result=backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=previous
 p=Path(c['output_root']);done=d.read(p/'completion.json');done.update(initialization='own_FIXED_FINAL30+20/resetoptimizer',cumulative_epochs=50,cumulative_optimizer_steps=11100,parent_epochs=30,parent_optimizer_steps=6660);d.write(p/'completion.json',done)
 return result
