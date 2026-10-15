"""Paired feature dropout: native independent masks or disjoint dropped sets."""
import copy,torch
from torch.nn import functional as F
PREFIX='id_backbone.encoder.core.id_backbone.'
SITES={PREFIX+k:p for k,p in [('t1.drop',.1),('t2.drop',.1),('t3.drop',.1),('f1.drop',.05),('f2.drop',.05),('f3.drop',.05),('freq_stats_proj.2',.1125),('pa_b1.drop',.08),('pa_b2.drop',.08),('pa_b3.drop',.08),('pa_proj.2',.1125),('pa_stats_proj.2',.1125),('fuse.2',.45)]}
KINDS=('INDEPENDENT','ANTITHETIC')
class NativeMask(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,noise,p):ctx.save_for_backward(noise);ctx.p=p;return x*noise
 @staticmethod
 def backward(ctx,g):
  noise,=ctx.saved_tensors
  # CUDA native dropout backward uses its own reciprocal rounding, notably p=.45.
  gx=torch.ops.aten.native_dropout_backward(g,noise.ne(0),1/(1-ctx.p)) if g.is_cuda else g*noise
  return gx,None,None
def draw_second(first,p,kind):
 if kind not in KINDS or not 0<p<.5 or first.dtype!=torch.bool:raise ValueError('Fixed p<.5 paired feature masks required')
 q=p if kind=='INDEPENDENT' else p/(1-p)
 noise=F.dropout(torch.ones_like(first,dtype=torch.float32),q,True,False)
 candidate=noise.ne(0)
 return candidate if kind=='INDEPENDENT' else (~first)|candidate
class PairState:
 def __init__(self,kind):
  if kind not in KINDS:raise ValueError('Unknown mask coupling')
  self.kind=kind;self.stage='idle';self.cache={};self.seen=[]
  self.audit=dict(calls=0,sites={n:dict(p=p,calls=0,elements=0,first_drop=0,second_drop=0,both_drop=0,different=0) for n,p in SITES.items()})
 def start(self):
  if self.stage!='idle' or self.cache:raise ValueError('Previous paired L masks not consumed')
  self.stage='first';self.seen=[]
 def between(self):
  if self.stage!='first' or self.seen!=list(SITES) or set(self.cache)!=set(SITES):raise ValueError('First L feature-dropout inventory differs')
  self.stage='between';self.seen=[]
 def second(self):
  if self.stage!='between' or set(self.cache)!=set(SITES):raise ValueError('Only paired second L pass may consume masks')
  self.stage='second';self.seen=[]
 def finish(self):
  if self.stage!='second' or self.seen!=list(SITES) or self.cache:raise ValueError('Second L mask inventory differs')
  self.audit['calls']+=1;self.stage='idle';self.seen=[]
 def snapshot(self):return copy.deepcopy(self.audit)
 def apply(self,name,x,p):
  if name not in SITES or p!=SITES[name] or x.ndim<2 or x.shape[0]!=128 or not x.is_floating_point() or x.dtype!=torch.float32 or len(self.seen)>=len(SITES) or name!=list(SITES)[len(self.seen)]:raise ValueError('Actual fixed L128 feature-dropout path required')
  self.seen.append(name)
  if self.stage=='first':
   if name in self.cache:raise ValueError('Duplicate first mask')
   if x.is_cuda:
    output,keep=torch.ops.aten.native_dropout(x,p,True);scale=x.new_tensor(1-p).reciprocal()
   else:
    noise=F.dropout(torch.ones_like(x),p,True,False);keep=noise.ne(0);scale=noise.amax().detach();output=NativeMask.apply(x,noise,p)
   self.cache[name]=(keep,scale,tuple(x.shape));return output
  if self.stage!='second' or name not in self.cache:raise ValueError('Unmatched paired feature dropout')
  first,scale,shape=self.cache.pop(name)
  if tuple(x.shape)!=shape or first.device!=x.device:raise ValueError('Paired feature shape/device differs')
  if x.is_cuda and self.kind=='INDEPENDENT':output,second=torch.ops.aten.native_dropout(x,p,True)
  else:
   second=draw_second(first,p,self.kind)
   output=NativeMask.apply(x,second.to(x.dtype)*scale,p)
  # Keep the exact native forward scale, separate from its CUDA backward scale.
  a=self.audit['sites'][name];n=x.numel();a['calls']+=1;a['elements']+=n;a['first_drop']+=n-int(first.sum());a['second_drop']+=n-int(second.sum());a['both_drop']+=int((~first&~second).sum());a['different']+=int((first^second).sum())
  if self.kind=='ANTITHETIC' and bool((~first&~second).any()):raise ValueError('Antithetic masks dropped the same coordinate twice')
  return output
class PairedDropout(torch.nn.Dropout):
 def forward(self,x):
  state=self._pair_state
  if not self.training or state.stage=='idle':return F.dropout(x,self.p,self.training,self.inplace)
  if self.inplace or state.stage=='between':raise ValueError('Only actual paired L feature forwards may use mask cache')
  return state.apply(self._pair_name,x,self.p)
def install(m,kind):
 modules=dict(m.named_modules());actual={n:mod.p for n,mod in modules.items() if n.startswith(PREFIX) and isinstance(mod,torch.nn.modules.dropout._DropoutNd)}
 if actual!=SITES or any(type(modules[n]) is not torch.nn.Dropout or modules[n].inplace for n in SITES):raise ValueError('Original13 feature dropout sites/rates required')
 state=PairState(kind);m._pair_state=state
 for name in SITES:
  mod=modules[name];mod.__class__=PairedDropout;mod._pair_state=state;mod._pair_name=name
 return m
def validate_audit(a,kind,steps):
 if kind not in KINDS or a.get('calls')!=steps or set(a.get('sites',{}))!=set(SITES):raise ValueError('Complete paired feature mask audit required')
 for name,s in a['sites'].items():
  if s.get('p')!=SITES[name] or s.get('calls')!=steps or s.get('elements',0)<=0:raise ValueError('Fixed feature site/step budget differs')
  if any(not 0<=s[k]<=s['elements'] for k in ('first_drop','second_drop','both_drop','different')) or s['different']!=s['first_drop']+s['second_drop']-2*s['both_drop']:raise ValueError('Mask pair count identities differ')
  if any(abs(s[k]/s['elements']-s['p'])>.005 for k in ('first_drop','second_drop')):raise ValueError('Observed mask marginals differ from original rates')
  if kind=='ANTITHETIC' and s['both_drop']!=0:raise ValueError('Repeated antithetic drops detected')
  if kind=='INDEPENDENT' and abs(s['both_drop']/s['elements']-s['p']**2)>.005:raise ValueError('Independent double-drop frequency differs')
