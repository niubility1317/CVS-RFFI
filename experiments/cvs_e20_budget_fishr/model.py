"""Strict native E20 inheritance; auxiliary heads always initialize separately."""
from types import SimpleNamespace
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_e20_foundation.verify import verify
from experiments.cvs_stage2_rx_risk.model import OriginalIdentity
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_early_geometry.model import groups


def build(c,device,checkpoint=None):
 d.validate(c)
 if checkpoint is None:
  provenance=verify(c['model_seed'],build_smoke=False)
  if provenance['status']!='VERIFIED' or Path(provenance['checkpoint']).as_posix()!=c['parent_checkpoint']:
   raise ValueError('Verified native E20 parent differs')
  payload=torch.load(c['parent_checkpoint'],map_location='cpu',weights_only=False)
 else:
  payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
  expected=dict(config=c,epoch=20,total_epochs=40,optimizer_steps=4440,parent_checkpoint=c['parent_checkpoint'],checkpoint_selection='final_only',target_access=False,run_id=c['run_id'],candidate_id=c['row_id'])
  if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('Own E20+20 checkpoint identity differs')
  provenance=payload['parent_provenance']
  if provenance['status']!='VERIFIED' or Path(provenance['checkpoint']).as_posix()!=c['parent_checkpoint'] or provenance['training_target_access']:
   raise ValueError('Own checkpoint lacks original E20 inheritance')
 args=SimpleNamespace(**payload['baseline_args'])
 torch.manual_seed(c['model_seed'])
 with installed(c['reference_config']) as native:base=native.build_baseline_model(args,torch.device(device))
 base._stage2_baseline_args=vars(args).copy()
 if checkpoint is None:base.load_state_dict(payload['model'],strict=True)
 model=OriginalIdentity(base).to(device)
 if c['multilevel']['enabled']:
  from .multilevel import MultiLevel
  model.multilevel=MultiLevel(base,seed=c['model_seed']).to(device)
 if checkpoint is not None:model.load_state_dict(payload['model'],strict=True)
 model._parent_provenance=provenance
 return model
