from pathlib import Path
from . import design as d
from experiments.cvs_evidence_responsibility_train20 import provenance as parent_proof,model as parent_model

def parent_source(c):
    d.validate(c);return parent_proof.source_proof(d.parent_config(c['model_seed']))

def frozen_proof(c):
    source=parent_source(c)
    freeze=d.read(d.BASE/c['row_id']/'source_export_frozen.json')
    expected=dict(status='OWN_EMA_EXPORT_FROZEN',config=c,parent_checkpoint=str(source/'final_ssdg.pth'),
        state_key='ema_model',new_optimizer_steps=0,head_fit_updates=0,target_contact_before_freeze=False)
    if freeze!=expected:raise ValueError('Own EMA export must be frozen before target access')
    return source

def load_parent(c,device,state_key='ema_model',require_frozen=True):
    import torch
    d.validate(c)
    if state_key not in ('model','ema_model'):raise ValueError('Only own student/EMA states allowed')
    source=frozen_proof(c) if require_frozen else parent_source(c)
    old=d.parent_config(c['model_seed']);checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    parent_proof.validate_checkpoint(checkpoint,old)
    if set(checkpoint['model'])!=set(checkpoint['ema_model']):raise ValueError('Own EMA must include the entire model state')
    model=parent_model.build(old,device);model.load_state_dict(checkpoint[state_key],strict=True)
    return model.eval().requires_grad_(False)

def own_model(c,device):return load_parent(c,device,'ema_model',True)

def freeze(c):
    source=parent_source(c);path=d.BASE/c['row_id']/'source_export_frozen.json'
    if path.exists():raise FileExistsError('Preserve frozen export')
    d.write(path,dict(status='OWN_EMA_EXPORT_FROZEN',config=c,parent_checkpoint=str(source/'final_ssdg.pth'),
        state_key='ema_model',new_optimizer_steps=0,head_fit_updates=0,target_contact_before_freeze=False))
