import torch
from . import design as d
from experiments.cvs_feature_adapters30 import model as fixed

parent=fixed.parent

def from_state(c,parent_state,device='cpu'):
 d.validate(c);row=next(r for r in d.rows() if r['row_id']==c['row_id']);m=fixed.from_state(d.frozen_config(row),parent_state,device)
 m.get_parameter(d.HEAD).requires_grad_(True);m._feature_adaptation_config=c;m._default_head_joint=True
 counts(m);return m

def counts(m):
 c=m._feature_adaptation_config;d.validate(c);named=dict(m.named_parameters());names=[n for n,p in named.items() if p.requires_grad]
 expected=[n for n in named if (n.startswith('id_backbone.encoder.') and '.cls_head.' not in n if c['style']=='FULL' else fixed.is_adapter(n)) or n==d.HEAD]
 if set(names)!=set(expected) or named[d.HEAD].shape!=(6,160) or not named[d.HEAD].requires_grad or len(names)<2:raise ValueError('Same default W and real features must jointly train')
 return dict(total_parameters=sum(p.numel() for p in named.values()),trainable_parameters=sum(p.numel() for p in named.values() if p.requires_grad),trainable_names=names,default_head_frozen=False,default_head_joint=True,style=c['style'])

def own_model(c,device='cpu'):
 d.validate(c);out=d.BASE/c['row_id']/'source';done=d.read(out/'completion.json');f=d.read(out/'source_frozen.json');init=d.read(out/'initialization.json')
 if done['config']!=c or done['epoch']!=30 or done['optimizer_steps']!=2220 or done['target_access'] or done['U_used'] or done['EMA_used'] or not done['default_head_joint'] or done['head_joint_optimizer_steps']!=2220 or f['config']!=c or f['selection']!='fixed FINAL30' or f['target_access'] or not init['inheritance_verified'] or init['parent_config']!=d.parent_config() or init['target_contact'] or d.read(out/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Fixed legal joint feature/default-head checkpoint required')
 p=parent('cpu');state=p.state_dict();cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['config']!=c or cp['epoch']!=30 or cp['optimizer_steps']!=2220 or cp['checkpoint_selection']!='fixed FINAL30' or not cp['default_head_joint'] or torch.equal(cp['model'][d.HEAD],state[d.HEAD]):raise ValueError('Existing W must have jointly updated, never reset')
 m=from_state(c,state,device);fixed_names=[n for n,p in m.named_parameters() if not p.requires_grad]
 if any(not torch.equal(cp['model'][n],state[n]) for n in fixed_names):raise ValueError('Frozen original F changed')
 if not any(not torch.equal(cp['model'][n],state[n]) for n,p in m.named_parameters() if p.requires_grad and n!=d.HEAD and n in state):
  if c['style']=='FULL' or not any(bool(v.abs().sum()>0) for n,v in cp['model'].items() if n.endswith('lora_B.weight') or n.endswith('feature_adapter.up.weight')):raise ValueError('Head-only training is forbidden')
 m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
