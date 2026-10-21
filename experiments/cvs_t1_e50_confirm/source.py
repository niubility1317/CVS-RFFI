from functools import lru_cache
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_crossview_supcon41 import source as backend

@lru_cache(maxsize=2)
def original_parent(seed):
 from experiments.cvs_same_as_kd_confirm86 import model as pm
 if seed not in d.NEW_SEEDS:raise ValueError('Only own seed02/03 parents for new training')
 c=d.parent_config(seed);p=d.parent_source(seed)
 # Existing checker validates source physical roles and scratch20 -> own10 chain.
 checked=pm.own_model(c,'cpu');del checked
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['args']['seed']!=seed or cp['candidate_id']!='T1-s'+str(seed) or cp['run_id']!=c['run_id'] or cp['checkpoint_selection']!='final_only' or cp.get('ema_model') is None:
  raise ValueError('Same-seed fixed-final T1/EMA provenance differs')
 if set(cp['model'])!=set(cp['ema_model']):raise ValueError('Parent EMA tensor structure differs')
 return cp

def parent_payload(c):
 d.validate(c)
 if c['parent_source']!=str(d.parent_source(c['model_seed'])):raise ValueError('Cross-seed parent source forbidden')
 return original_parent(c['model_seed'])

def segment_train(c,design=None,model_api=None,observer=None):
 previous=backend.parent_payload;backend.parent_payload=parent_payload
 try:return backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=previous
