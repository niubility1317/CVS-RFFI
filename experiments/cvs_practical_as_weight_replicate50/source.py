"""Own original20 seed-aware parent, followed by this row's fixed weighted30 parent."""
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_metadata
from experiments.cvs_crossview_supcon41 import source as backend

def ancestor_payload(c):
 d.validate(c)
 from experiments.cvs_As_replicate49 import source as original
 cp=original.parent_payload(d.reference_config(c,30))
 check_checkpoint(cp,seed=c['model_seed'],run=d.PARENT_RUN,row=c['row_id'],epoch=20,updates=4440,baseline='')
 return cp

def parent_payload(c):
 d.validate(c);ancestor=ancestor_payload(c)
 if c['target_budget']==30:return ancestor
 row=next(r for r in d.rows() if r['row_id']==c['row_id']);expected=d.config(row,30);out=Path(c['parent_source'])
 if out!=Path(expected['output_root']):raise ValueError('Only own new weighted30 initializer')
 check_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),expected,d.read(d.SOURCE)['role_ids'])
 result=d.read(out/'result.json')
 if result['cumulative_epochs']!=30 or result['cumulative_optimizer_steps']!=6660 or result['actual_F_factory_seed']!=c['model_seed'] or result['U_truth_read'] or result['target_access'] or result['head_fit_updates']:raise ValueError('Own weighted30 cumulative/actualfactory metadata differs')
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,seed=c['model_seed'],run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=expected['parent_source']+'/final_ssdg.pth')
 keys=set(ancestor['model'])
 if set(cp['model'])!=keys or set(cp['ema_model'])!=keys or len(keys)!=206 or cp['model']['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape!=(6,160):raise ValueError('Own finite student/EMA/default head keys differ')
 for k in keys:
  a,b=cp['model'][k],cp['ema_model'][k]
  if a.shape!=ancestor['model'][k].shape or a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Own weighted30 finite state shapes differ')
 return cp

def segment_train(c,design=None,model_api=None,observer=None):
 prior=backend.parent_payload;backend.parent_payload=parent_payload
 try:result=backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=prior
 out=Path(c['output_root']);done=d.read(out/'completion.json');done.update(cumulative_epochs=c['target_budget'],cumulative_optimizer_steps=c['cumulative_updates'],parent_epochs=c['parent_epochs'],parent_optimizer_steps=c['parent_optimizer_steps'],initialization='own_seed_stage_resetAdamWcos'+str(c['epochs']));d.write(out/'completion.json',done)
 return result
