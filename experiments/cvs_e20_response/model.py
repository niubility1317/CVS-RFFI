"""Strict parent initialization; prediction constructs only its own saved state."""
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_nocfo_engineering.teacher import verify,teacher_config,expected_proof
from experiments.cvs_nocfo_foundation.model import build as native
from experiments.cvs_nocfo_engineering.model import groups as native_groups
def groups(model,c):
 pg,names=native_groups(model,c)
 for group in pg:group['peak']=c['optimizer']['peak_lr']
 return pg,names
def expected_parent(c):
 p=expected_proof(c['model_seed'])
 if c['parent_checkpoint']!=p['checkpoint'] or c['parent_training_epochs']!=20 or c['total_epochs']!=40:raise ValueError('Parent differs')
 return p
def build(c,device,checkpoint=None):
 d.validate(c)
 if checkpoint is None:
  proof=verify(c['model_seed']);assert proof==expected_parent(c)
  model=native(teacher_config(c['model_seed']),device,Path(proof['checkpoint']))
 else:
  from .provenance import checkpoint_provenance
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False);checkpoint_provenance(c,ck)
  model=native(teacher_config(c['model_seed']),device)
  model.load_state_dict(ck['model'],strict=True);proof=ck['parent_provenance']
 model._parent_provenance=proof
 if model._identity_input_policy['explicit_CFO_identity_input']:raise ValueError('CFO input enabled')
 return model
