import math
from contextlib import contextmanager
import torch
from torch import nn
from . import design as d
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_source_identity_readout.input_policy import apply

class FrequencyContext(nn.Module):
 """Packet-local residual attention: local/global have exactly the same weights."""
 def __init__(self,channels,mode,radius=1):
  super().__init__()
  if channels not in (16,32) or mode not in ('local','global') or radius!=1:raise ValueError('Fixed spectral context required')
  self.channels=channels;self.mode=mode;self.radius=radius;self.heads=4;self.width=2*channels
  self.norm1=nn.LayerNorm(channels);self.qkv=nn.Linear(channels,3*self.width);self.out=nn.Linear(self.width,channels)
  self.norm2=nn.LayerNorm(channels);self.ffn=nn.Sequential(nn.Linear(channels,4*channels),nn.GELU(),nn.Linear(4*channels,channels))
  self.residual_scale=.01
 def mask(self,n,device):
  if n<2:raise ValueError('At least two ordered frequency positions required')
  t=torch.arange(n,device=device)
  return torch.ones(n,n,dtype=torch.bool,device=device) if self.mode=='global' else (t[:,None]-t[None,:]).abs()<=self.radius
 def forward(self,x):
  if x.ndim!=3 or x.shape[1]!=self.channels:raise ValueError('Frequency channel interface differs')
  v=x.transpose(1,2);n=v.shape[1];h=self.norm1(v)
  index=torch.arange(n,device=x.device,dtype=x.dtype)[:,None];rate=torch.exp(torch.arange(0,self.channels,2,device=x.device,dtype=x.dtype)*(-math.log(10000)/self.channels))
  p=torch.stack(((index*rate).sin(),(index*rate).cos()),-1).flatten(1);h=h+p[None]
  q,k,value=self.qkv(h).chunk(3,-1)
  def heads(t):return t.reshape(len(x),n,self.heads,self.width//self.heads).transpose(1,2)
  q,k,value=map(heads,(q,k,value));score=q@k.transpose(-2,-1)/math.sqrt(self.width//self.heads)
  score=score.masked_fill(~self.mask(n,x.device)[None,None],float('-inf'));prob=score.softmax(-1)
  context=(prob@value).transpose(1,2).reshape(len(x),n,self.width)
  v=v+self.residual_scale*self.out(context);v=v+self.residual_scale*self.ffn(self.norm2(v))
  return v.transpose(1,2)

def install(model,c):
 d.validate(c);b=model.id_backbone.encoder.core.id_backbone;head=b.cls_head
 if type(b.freq_gate).__name__!='FreqBandGate1d' or type(b.pa_gate).__name__!='EnvelopeGate1d' or head.weight.shape!=(6,160) or head.scale!=30.:raise ValueError('Baseline gates/cosine interface differs')
 original_head=head
 if c['frequency_context']['enabled']:
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(c['model_seed']+41137)
   for level,ch in [('f1',16),('f2',32)]:setattr(b,level,nn.Sequential(getattr(b,level),FrequencyContext(ch,c['frequency_context']['mode'])))
 if b.cls_head is not original_head:raise ValueError('Feature experiment changed classifier')
 model._identity_input_policy=apply(model)
 model._frequency_context_contract=dict(c['frequency_context'],head='unchanged cosine6x160',old_pseudo_gating=True,EMA=True)
 return model

@contextmanager
def installed(c):
 d.validate(c)
 with native_installed(c) as n:
  previous=n.build_baseline_model
  def build(args,device):return install(previous(args,device),c).to(device)
  n.build_baseline_model=build
  try:yield n
  finally:n.build_baseline_model=previous

def build(c,device='cpu'):
 a=d.make_args(c,str(device));a.num_domains=5;a.input_len=256
 with installed(c) as n:return n.build_baseline_model(a,torch.device(device))
