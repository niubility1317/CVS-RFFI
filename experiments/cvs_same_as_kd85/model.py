"""One frozen external teacher on the exact existing As tensor; deployed student unchanged."""
from copy import deepcopy
from types import MethodType
from pathlib import Path
import torch
from . import design as d
from .kd import objective
from experiments.cvs_as_temperature84 import model as base
classifier,source_domain_count=base.classifier,base.source_domain_count
TEACHERS={}
class Captured:
 def __init__(self,obj):self.obj=obj;self.last=None
 def __getattr__(self,k):
  if k=='obj':raise AttributeError(k)
  return getattr(self.obj,k)
 def __deepcopy__(self,memo):
  v=type(self)(deepcopy(self.obj,memo));memo[id(self)]=v;return v
 def apply(self,*a,**kw):
  view,meta=self.obj.apply(*a,**kw);self.last=view.detach();return view,meta
def teacher(device):
 key=str(device)
 if key not in TEACHERS:
  from experiments.cvs_as_temperature84.source_comparison import inherited
  devices=[torch.cuda.current_device()] if torch.device(device).type=='cuda' else []
  with torch.random.fork_rng(devices=devices):m,provenance=inherited('csil');m=m.to(device).eval().requires_grad_(False)
  TEACHERS[key]=dict(model=m,before={k:v.clone() for k,v in m.state_dict().items()},packets=0,provenance=provenance)
 return TEACHERS[key]
def hook(self,**kw):
 old=base.mixed;calls=0
 def mixed(a,b,y,mask,T,ordinary_terms):
  nonlocal calls
  calls+=1
  if T!=1. or calls!=1:raise ValueError('KD builds on native-scale T1 CE-only As')
  result=old(a,b,y,mask,T,ordinary_terms);x=self._full_channel.last
  if x is None or x.shape!=(64,2,256):raise ValueError('Exact existing full As64 teacher input missing')
  info=teacher(x.device)
  with torch.no_grad():z=info['model'](x)['tx_logits']
  info['packets']+=64;value,use=objective(a[mask],b[mask],z,y[mask]);delta=.5*value
  self._kd_records.append(dict(step=kw['optimizer_step'],teacher_correct=int(use.sum()),packets=64,teacher_NN=64,student_extra_NN=0,KL=float(value.detach()),total_L_delta=.25*float(value.detach()),common_scale=5.,beta=1.,all_As_CE=True,teacher_input_same_As=True,teacher_grad_enabled=False))
  return (result[0]+delta,*result[1:])
 base.mixed=mixed
 try:return base.hook(self,**kw)
 finally:
  base.mixed=old;self._full_channel.last=None
  if calls!=1:raise ValueError('Exactly one same-As transfer per existing step required')
hook._accepts_labeled_inputs=True
def attach(m,c):
 d.validate(c)
 if c['arm']=='KD':m._full_channel=Captured(m._full_channel);m._kd_records=[];m._source_labeled_feature_hook=MethodType(hook,m);m._feature_learning_report['sameAsKD85']=c['knowledge_transfer']
 return m
def install(m,c):return attach(base.install(m,d.reference_config(c['segment'])),c)
def build(c,device='cpu'):return attach(base.build(d.reference_config(c['segment']),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 if c['segment']!=30:raise ValueError('Only own cumulative FINAL30')
 from .source import parent_payload
 parent_payload(c)
 p=Path(c['output_root']);done=d.read(p/'completion.json');frozen=d.read(p/'source_frozen.json');init=d.read(p/'initialization.json')
 if done['config']!=c or done['epoch']!=10 or done['optimizer_steps']!=2220 or done['target_access'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['checkpoint_sources']!=c['checkpoint_sources'] or d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Own exactsource final provenance differs')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=10 or cp['a1_ema_successful_updates']!=2220 or cp['run_id']!=c['run_id'] or cp['candidate_id']!=c['row_id'] or cp['args']['baseline_ckpt']!=c['parent_source']+'/final_ssdg.pth' or cp['args']['teacher_ckpt']:raise ValueError('Own source final identity differs')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
