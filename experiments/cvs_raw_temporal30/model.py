import torch
from . import design as d
from .layers import attach
from experiments.cvs_feature_adapters30 import model as old
parent=old.parent
def is_adapter(name):return '.raw_temporal.' in name
def from_state(c,parent_state,device='cpu'):
 d.validate(c)
 if parent_state is None:raise ValueError('Verified original B0 parent required')
 dev=torch.device(device);devices=[] if dev.type=='cpu' else [dev.index or 0]
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(d.SEED);m=old.base.build(d.parent_config(),'cpu');m.load_state_dict(parent_state,strict=True);m.requires_grad_(False);torch.manual_seed(d.ADAPTER_SEED);attach(m.id_backbone.encoder.core.id_backbone,c['style']);m.to(dev)
 for name,p in m.named_parameters():p.requires_grad_(is_adapter(name) or name==d.HEAD)
 m._source_labeled_feature_hook=None;m._feature_adaptation_config=c;m.eval();counts(m);return m
def counts(m):
 c=m._feature_adaptation_config;d.validate(c);named=dict(m.named_parameters());names=[n for n,p in named.items() if p.requires_grad];expected=[n for n in named if is_adapter(n) or n==d.HEAD]
 if set(names)!=set(expected) or named[d.HEAD].shape!=(6,160) or sum(named[n].numel() for n in names)!=3212 or sum(p.numel() for p in named.values())!=902974:raise ValueError('Same original W and2252 supplemental feature parameters required')
 return dict(total_parameters=902974,trainable_parameters=3212,trainable_names=names,default_head_frozen=False,default_head_joint=True,style=c['style'])
def own_model(c,device='cpu'):
 d.validate(c);out=d.BASE/c['row_id']/'source';done=d.read(out/'completion.json');f=d.read(out/'source_frozen.json');init=d.read(out/'initialization.json')
 if done['config']!=c or done['epoch']!=30 or done['optimizer_steps']!=2220 or done['target_access'] or done['U_used'] or done['EMA_used'] or not done['default_head_joint'] or done['head_joint_optimizer_steps']!=2220 or done['raw_supplement_training_NN']!=568320 or f['config']!=c or f['selection']!='fixed FINAL30' or f['target_access'] or not init['inheritance_verified'] or not init['source_roles_match'] or init['parent_config']!=d.parent_config() or init['target_contact'] or d.read(out/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Fixed legal default-head/raw feature checkpoint required')
 p=parent('cpu');state=p.state_dict();cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['config']!=c or cp['epoch']!=30 or cp['optimizer_steps']!=2220 or cp['checkpoint_selection']!='fixed FINAL30' or not cp['default_head_joint'] or torch.equal(cp['model'][d.HEAD],state[d.HEAD]):raise ValueError('Original W did not jointly update')
 m=from_state(c,state,device)
 if any(not torch.equal(cp['model'][n],state[n]) for n,p in m.named_parameters() if not p.requires_grad):raise ValueError('Frozen original F changed')
 if not all(bool(cp['model']['id_backbone.encoder.core.id_backbone.raw_temporal.'+n+'.weight'].abs().sum()>0) for n in ('B1','B2')):raise ValueError('Real t1/t2 raw feature heads must both update;head-only training forbidden')
 m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
