from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F

class RawTemporal(nn.Module):
 def __init__(self,condition):
  super().__init__()
  if condition not in ('RAW','PAIR'):raise ValueError('Registered input condition required')
  self.condition=condition;self.packets=0
  self.down=nn.Conv1d(2,8,5,stride=2,padding=2,bias=False);self.mix=nn.Conv1d(8,12,1,bias=False);self.temporal=nn.Conv1d(12,12,5,padding=2,groups=12,bias=False)
  self.B1=nn.Conv1d(12,72,1,bias=False);self.B2=nn.Conv1d(12,96,1,bias=False);nn.init.zeros_(self.B1.weight);nn.init.zeros_(self.B2.weight)
 def forward(self,x):
  if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Same single packet2x256 core IQ required')
  v=x if self.condition=='RAW' else F.avg_pool1d(x,2).repeat_interleave(2,dim=-1)
  h=F.silu(self.temporal(self.mix(F.silu(self.down(v)))));self.packets+=len(x)
  return self.B1(h),self.B2(F.avg_pool1d(h,2))

def prepare_raw(module,args):
 if not args or any(getattr(module,n).__dict__.get('_raw_correction') is not None for n in ('t1','t2')):raise ValueError('No stale or unconsumed supplemental feature permitted')
 a,b=module.raw_temporal(args[0]);module.t1.__dict__['_raw_correction']=a;module.t2.__dict__['_raw_correction']=b

def raw_block_forward(self,x):
 correction=self.__dict__.pop('_raw_correction',None)
 if correction is None:raise ValueError('Supplemental feature missing from actual block')
 u=self.pw(self.dw(x))
 if correction.shape!=u.shape:raise ValueError('Actual preGN feature shape differs')
 return self.drop(self.pool(self.act(self.norm(u+correction))))

def finish_raw(module,args,output):
 if any(getattr(module,n).__dict__.get('_raw_correction') is not None for n in ('t1','t2')):raise ValueError('Actual t1/t2 did not consume raw features')

def attach(core,condition):
 for name,ci,co in (('t1',48,72),('t2',72,96)):
  block=getattr(core,name)
  if type(block).__name__!='DSConvBlock1d' or block.pw.in_channels!=ci or block.pw.out_channels!=co or block.dw.stride!=(1,):raise ValueError('Original F time block layout differs')
  block.forward=MethodType(raw_block_forward,block)
 core.raw_temporal=RawTemporal(condition);core.register_forward_pre_hook(prepare_raw);core.register_forward_hook(finish_raw)
 return core
