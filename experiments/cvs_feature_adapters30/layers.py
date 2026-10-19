"""Pre-pool rectangular feature adapters; no full CRRA gate or whitening."""
import math
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F

class InputTemporalResidual(nn.Module):
 def __init__(self,cin,cout,rank=8,kernel=5):
  super().__init__()
  if min(cin,cout,rank,kernel)<=0 or kernel%2!=1:raise ValueError('Positive channels/rank and odd temporal kernel required')
  self.down=nn.Conv1d(cin,rank,1,bias=False)
  self.temporal=nn.Conv1d(rank,rank,kernel,padding=kernel//2,groups=rank,bias=False)
  self.up=nn.Conv1d(rank,cout,1,bias=False);nn.init.zeros_(self.up.weight)
 def forward(self,x):return self.up(F.silu(self.temporal(self.down(x))))

def temporal_forward(self,x):
 # Keep the original block object/hooks, so t2/SIG_TIME reads the corrected result.
 u=self.pw(self.dw(x))+self.feature_adapter(x)
 return self.drop(self.pool(self.act(self.norm(u))))

def lora_forward(self,x):return self._conv_forward(x,self.weight,self.bias)+self._lora_scale*self.lora_B(self.lora_A(x))

def attach(block,style,rank=8,kernel=5,alpha=8.):
 if type(block).__name__!='DSConvBlock1d' or not isinstance(block.pw,nn.Conv1d) or block.pw.kernel_size!=(1,) or block.pw.groups!=1 or block.dw.stride!=(1,):raise ValueError('Verified original F depthwise/pointwise time block required')
 cin,cout=block.pw.in_channels,block.pw.out_channels
 if style=='LORA':
  p=block.pw;p.lora_A=nn.Conv1d(cin,rank,1,bias=False);p.lora_B=nn.Conv1d(rank,cout,1,bias=False);nn.init.zeros_(p.lora_B.weight);p._lora_scale=float(alpha)/rank;p.forward=MethodType(lora_forward,p)
 elif style=='TEMP':
  block.feature_adapter=InputTemporalResidual(cin,cout,rank,kernel);block.forward=MethodType(temporal_forward,block)
 else:raise ValueError('Only the registered shallow/middle feature adapter styles')
 return block
