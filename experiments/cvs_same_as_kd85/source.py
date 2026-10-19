import torch
from pathlib import Path
from .design import *
from experiments.cvs_crossview_supcon41 import source as backend

def parent_payload(c):
 p=Path(c['parent_source']);row=next(r for r in rows() if r['row_id']==c['row_id']);expected=config(row,20);done=read(p/'completion.json');init=read(p/'initialization.json')
 if done['config']!=expected or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or read(p/'source_contract.json')['role_ids']!=read(SOURCE)['role_ids']:raise ValueError('Own fixed scratch20 parent contract differs')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=20 or cp['a1_ema_successful_updates']!=4440 or cp['run_id']!=c['run_id'] or cp['candidate_id']!=c['row_id'] or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt'] or cp['checkpoint_selection']!='final_only' or cp.get('ema_model') is None:raise ValueError('Own parent/EMA provenance differs')
 return cp

def segment_train(c,design=None,model_api=None,observer=None):
 previous=backend.parent_payload;backend.parent_payload=parent_payload
 try:return backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=previous
