"""Persistent feature statistics and explicitly L-only evidence competition.

No class-indexed feature parameters, RX routing, or query fitting.
"""
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
import math
import torch
from torch import nn
from experiments.cvs_feature_multiscale50 import design as parent
from experiments.cvs_feature_multiscale50.model import build as baseline_build
from experiments.cvs_e20_cross_style.style import _paths

GROUPS=('time','frequency','PA','reference')
WARMUP=5*222
MASK_P=.1

class RelativeNormBlock(nn.Module):
 def __init__(self,old,kind,seed):
  super().__init__()
  if kind not in ('ordinary','restore') or not isinstance(old.norm,nn.GroupNorm):raise ValueError('Expected one original GN time block')
  self.kind=kind
  for name in ('dw','pw','norm','act','pool','drop'):setattr(self,name,deepcopy(getattr(old,name)))
  c,g=old.norm.num_channels,old.norm.num_groups;self.channels=c;self.groups=g
  # Do not alter the native global dropout/augmentation RNG stream.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(int(seed)+78347)
   self.projection=(nn.Linear(2*g,c) if kind=='restore' else nn.Sequential(nn.Linear(c,g,bias=False),nn.SiLU(),nn.Linear(g,c)))
  self.alpha_raw=nn.Parameter(torch.tensor(math.atanh(.05/.2)))
 def statistics(self,h):
  b,c,t=h.shape
  if c!=self.channels or not h.is_floating_point() or h.is_complex():raise ValueError('Expected real native feature maps')
  grouped=h.reshape(b,self.groups,c//self.groups,t)
  mu=grouped.mean((2,3));sigma=torch.sqrt(grouped.var((2,3),unbiased=False)+self.norm.eps)
  rms=torch.sqrt(h.square().mean((1,2))+self.norm.eps)[:,None]
  return torch.cat((mu/rms,torch.log(sigma/rms)),1)
 def forward(self,x):
  h=self.pw(self.dw(x));normalized=self.norm(h)
  q=self.statistics(h) if self.kind=='restore' else normalized.mean(-1)
  delta=self.projection(q).unsqueeze(-1)
  restored=normalized+.2*self.alpha_raw.tanh()*delta
  return self.drop(self.pool(self.act(restored)))


def layout(model):
 modules,paths=_paths(model);prefix=paths['time_down'].rsplit('.',1)[0];core=modules[prefix]
 refs=[(n,m) for n,m in model.named_modules() if n.endswith('response_projection')]
 if len(refs)!=1:raise ValueError('One real reference-response projection required')
 return core,prefix,refs[0]

@contextmanager
def evidence_mask(model,group,role='L_s',diagnostic=False):
 if group not in GROUPS or role not in (('L_s','V') if diagnostic else ('L_s',)):raise ValueError('Evidence masks only consume source L; frozen source diagnostics may use V')
 if getattr(model,'_evidence_active',False):raise RuntimeError('Nested evidence mask')
 if not diagnostic and not model.training:raise ValueError('Training mask requires L student train mode')
 core,prefix,(reference_path,reference)=layout(model)
 for key in ('use_time_path','use_freq_path','use_pa_path','use_stats_path','use_circularity'):
  if not getattr(core,key):raise ValueError('Native evidence graph differs')
 if core.use_dac_path or core.branch_drop_p!=0:raise ValueError('Unregistered native DAC/drop policy')
 names={'time':['t_proj'],'frequency':['f_proj','freq_stats_proj','dac_subband_agg','pa_stats_proj'],'PA':['pa_proj','pa_stats_proj'],'reference':[]}[group]
 targets=[(prefix+'.'+n,getattr(core,n)) for n in names if isinstance(getattr(core,n,None),nn.Module)]
 if group=='reference':targets=[(reference_path,reference)]
 details=dict(group=group,role=role,diagnostic=diagnostic,hook_calls=[],frequency_sidepaths_zeroed=False,target_access=False)
 handles=[];original=core._mirror_compressed_features;model._evidence_active=True
 def hook(module,args,output,path):
  if not isinstance(output,torch.Tensor):raise ValueError('Expected real evidence tensor')
  details['hook_calls'].append(path)
  return output*0
 try:
  for path,module in targets:
   handles.append(module.register_forward_hook(lambda m,a,o,p=path:hook(m,a,o,p)))
  if group=='frequency':
   def mirror(self,*args,**kwargs):
    feat,rho,dac_stats,pa_stats=original(*args,**kwargs)
    details['frequency_sidepaths_zeroed']=True
    return feat,None if rho is None else rho*0,dac_stats*0,pa_stats*0
   core._mirror_compressed_features=MethodType(mirror,core)
  yield details
  if any(path not in details['hook_calls'] for path,_ in targets):raise RuntimeError('Selected actual evidence path was bypassed')
  if group=='frequency' and not details['frequency_sidepaths_zeroed']:raise RuntimeError('Frequency sidepath hook was bypassed')
 finally:
  core._mirror_compressed_features=original
  for handle in handles:handle.remove()
  model._evidence_active=False

@contextmanager
def labeled_scope(model):
 # The native loop calls this at the real out_l entry, never U or teacher.
 if not model.training:
  yield
  return
 model._feature_l_steps+=1;step=model._feature_l_steps;group=None
 if model._feature_competition and step>WARMUP:
  rng=torch.Generator().manual_seed(model._feature_seed+step*701+10007)
  if float(torch.rand((),generator=rng))<MASK_P:group=GROUPS[int(torch.randint(len(GROUPS),(1,),generator=rng))]
 if group is None:
  yield
 else:
  with evidence_mask(model,group) as detail:
   model._feature_mask_counts[group]+=1
   yield
  model._last_l_mask=detail


def install(model,seed,norm='none',competition=False):
 if norm not in ('none','ordinary','restore'):raise ValueError('Unregistered normalization policy')
 core,_,_=layout(model)
 if norm!='none':core.t1=RelativeNormBlock(core.t1,norm,seed)
 model._feature_seed=int(seed);model._feature_competition=bool(competition)
 model._feature_l_steps=0;model._feature_mask_counts={g:0 for g in GROUPS};model._evidence_active=False
 model._phase1_labeled_feature_context=MethodType(labeled_scope,model)
 extra=0 if norm=='none' else sum(p.numel() for p in core.t1.projection.parameters())+core.t1.alpha_raw.numel()
 model._feature_learning_report=dict(norm=norm,layer='t1' if norm!='none' else None,groups=core.t1.norm.num_groups,channels=core.t1.norm.num_channels,extra_parameters=extra,alpha_initial=.05,alpha_bound=.2,competition=bool(competition),mask_groups=list(GROUPS),mask_probability=MASK_P,warmup_steps=WARMUP,mask_role='L_s student only',classifier_changed=False,deployed_embedding=160,per_packet_statistics=True)
 return model

def build(seed,norm='none',competition=False,device='cpu'):
 row=next(r for r in parent.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed)
 model=baseline_build(parent.config(row),device)
 return install(model,seed,norm,competition).to(device)

def embedding(model,x):return model.forward_identity_only(x)['id_feat_joint']
