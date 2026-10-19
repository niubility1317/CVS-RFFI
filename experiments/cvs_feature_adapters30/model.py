from pathlib import Path
import torch
from . import design as d
from .layers import attach
from experiments.cvs_F_nontrue20 import model as base

def is_adapter(name):return '.lora_A.' in name or '.lora_B.' in name or '.feature_adapter.' in name

def configure(m,c):
 d.validate(c);m.requires_grad_(False);core=m.id_backbone.encoder.core.id_backbone
 if c['style'] in ('LORA','TEMP'):
  device=next(m.parameters()).device;devices=[] if device.type=='cpu' else [device.index or 0]
  with torch.random.fork_rng(devices=devices):
   torch.manual_seed(20261019641)
   for name in ('t1','t2'):attach(getattr(core,name),c['style'],d.RANK,d.KERNEL,d.ALPHA)
  m.to(device)
 for name,p in m.named_parameters():
  allowed=name.startswith('id_backbone.encoder.') and '.cls_head.' not in name if c['style']=='FULL' else is_adapter(name)
  p.requires_grad_(allowed)
 if core.cls_head.weight.shape!=(6,160) or core.cls_head.weight.requires_grad or any('CRRAAdapter'==type(x).__name__ for x in m.modules()):raise ValueError('Same fixed default head and no full CRRA required')
 m._source_labeled_feature_hook=None;m._feature_adaptation_config=c;m.eval()
 return m

def from_state(c,parent_state,device='cpu'):
 if parent_state is None:raise ValueError('Verified parent state required; no silent scratch fallback')
 dev=torch.device(device);devices=[] if dev.type=='cpu' else [dev.index or 0]
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(d.SEED);m=base.build(d.parent_config(),'cpu');m.load_state_dict(parent_state,strict=True);m=configure(m,c).to(dev)
 return m

def parent(device='cpu'):
 # Existing full-lineage/source-role/default-head checks precede loading any parent.
 return base.own_model(d.parent_config(),device)

def counts(m):
 named=dict(m.named_parameters());allowed=[n for n,p in named.items() if p.requires_grad]
 if not allowed or named[d.HEAD].requires_grad or any('.cls_head.' in n for n in allowed):raise ValueError('No new or trainable classifier allowed')
 return dict(total_parameters=sum(p.numel() for p in named.values()),trainable_parameters=sum(p.numel() for p in named.values() if p.requires_grad),trainable_names=allowed,default_head_frozen=True,style=m._feature_adaptation_config['style'])

def own_model(c,device='cpu'):
 d.validate(c);out=d.BASE/c['row_id']/'source';done=d.read(out/'completion.json');frozen=d.read(out/'source_frozen.json');init=d.read(out/'initialization.json')
 if done['config']!=c or done['epoch']!=30 or done['optimizer_steps']!=2220 or done['target_access'] or done['U_used'] or done['EMA_used'] or not done['default_head_frozen'] or frozen['config']!=c or frozen['selection']!='fixed FINAL30' or frozen['target_access'] or not init['inheritance_verified'] or not init['source_roles_match'] or init['parent_config']!=d.parent_config() or init['target_contact'] or d.read(out/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Fixed legal E20+10 feature checkpoint required')
 p=parent('cpu');state=p.state_dict();cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['config']!=c or cp['epoch']!=30 or cp['optimizer_steps']!=2220 or cp['checkpoint_selection']!='fixed FINAL30' or not cp['default_head_frozen'] or not torch.equal(cp['model'][d.HEAD],state[d.HEAD]):raise ValueError('Frozen original W/final checkpoint identity differs')
 m=from_state(c,state,device);fixed=[k for k,v in m.named_parameters() if not v.requires_grad]
 if any(not torch.equal(cp['model'][k],state[k]) for k in fixed):raise ValueError('Frozen original parameters changed')
 m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
