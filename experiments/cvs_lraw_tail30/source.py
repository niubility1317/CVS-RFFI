"""Reuse exact P43 strict original own20 loader; never relax parent configuration."""
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_default_head
from experiments.cvs_crossview_supcon41 import source as backend

def parent_payload(c):
 d.validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b,source as s
 row=next(r for r in b.rows() if r['row_id']==c['row_id']);expected=b.config(row,30)
 if b.config(row,20)!=d.reference_config(next(r for r in d.rows() if r['row_id']==c['row_id']),20) or expected!=d.reference_config(next(r for r in d.rows() if r['row_id']==c['row_id']),30) or c['parent_source']!=expected['parent_source']:raise ValueError('Original own20 actual config/lineage differs')
 cp=s.parent_payload(expected)
 check_checkpoint(cp,run=d.REFERENCE_RUN,row=c['row_id'],epoch=20,updates=4440,baseline='')
 keys=set(cp['model'])
 if set(cp['ema_model'])!=keys or len(keys)!=206:raise ValueError('Actual F/defaultcos student/ownEMA state differs')
 check_default_head(cp['model']['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape)
 for key in keys:
  a,b=cp['model'][key],cp['ema_model'][key]
  if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Finite matched ownEMA shape differs')
 if cp['args'].get('a1_ema_startup_average',False):raise ValueError('Original EMA policy differs')
 return cp

def segment_train(c,design=None,model_api=None,observer=None):
 previous=backend.parent_payload;backend.parent_payload=parent_payload
 try:return backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=previous
