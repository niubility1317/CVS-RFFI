"""Source-trained clean IQ denoising prior, independent of DURA."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
 def __init__(self,width,dilation):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),nn.SiLU(),nn.Conv1d(width,width,1))
 def forward(self,x):return x+self.net(x)*.2

class SignalPrior(nn.Module):
 def __init__(self,width=64):
  super().__init__();self.entry=nn.Conv1d(3,width,9,padding=4);self.blocks=nn.Sequential(*(Block(width,d) for d in (1,2,4,8,16,32)));self.attention=nn.MultiheadAttention(width,4,batch_first=True);self.output=nn.Conv1d(width,2,3,padding=1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,noisy,sigma):
  if noisy.ndim!=3 or noisy.shape[1:]!=(2,256) or sigma.shape!=(len(noisy),):raise ValueError('Expected IQ [N,2,256] and per-record sigma')
  rms=noisy.square().mean((1,2),keepdim=True).clamp_min(1e-8).sqrt();s=sigma[:,None,None].expand(-1,1,256);h=F.silu(self.entry(torch.cat((noisy/rms,s),1)));h=self.blocks(h);q=F.avg_pool1d(h,4).transpose(1,2);a,_=self.attention(q,q,q,need_weights=False);h=h+F.interpolate(a.transpose(1,2),size=256,mode='linear',align_corners=False);return noisy+rms*self.output(h)

def observation_step(received,state,action,step=.8):
 if received.shape!=state.shape or state.ndim!=3 or state.shape[1:]!=(2,256) or action.shape!=(len(state),512,512):raise ValueError('Expected paired IQ and action matrix')
 if not 0<step<=1:raise ValueError('Step must be in (0,1]')
 residual=received.reshape(len(state),512,1)-torch.bmm(action,state.reshape(len(state),512,1));bound=action.abs().sum(1).amax(1)*action.abs().sum(2).amax(1);update=torch.bmm(action.transpose(1,2),residual).squeeze(-1)/bound.clamp_min(1e-8)[:,None];return state+step*update.reshape_as(state)

def restore(prior,received,initial,action):
 state=initial
 for value in (.16,.08,.03):
  state=observation_step(received,state,action);state=prior(state,state.new_full((len(state),),value))
 return state
