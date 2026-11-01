"""Input-conditioned continuous IQ transport, in the frozen blind model's coordinates."""
import torch
from torch import nn
from torch.nn import functional as F

def backproject(received,state,action):
 if received.shape!=state.shape or state.ndim!=3 or state.shape[1:]!=(2,256) or action.shape!=(len(state),512,512):raise ValueError('Expected paired IQ and real action matrix')
 residual=received.reshape(len(received),512,1)-torch.bmm(action,state.reshape(len(state),512,1))
 scale=action.square().sum((1,2)).div(512).clamp_min(1e-6)
 return (torch.bmm(action.transpose(1,2),residual).squeeze(-1)/scale[:,None]).reshape_as(state)

class ConditionalTransport(nn.Module):
 def __init__(self,width=64):
  super().__init__();self.entry=nn.Conv1d(8,width,7,padding=3);self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,5,padding=2*d,dilation=d),nn.SiLU(),nn.Conv1d(width,width,1)) for d in (1,2,4,8)]);self.attention=nn.MultiheadAttention(width,4,batch_first=True);self.output=nn.Conv1d(width,2,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,state,parent,residual,time,log_variance):
  if state.shape!=parent.shape or state.shape!=residual.shape or state.ndim!=3 or state.shape[1:]!=(2,256):raise ValueError('Expected [B,2,256] IQ tensors')
  if time.shape!=(len(state),) or log_variance.shape!=(len(state),):raise ValueError('Expected per-record time and log variance')
  rms=parent.square().sum(1).mean(1).sqrt().clamp_min(1e-8);scale=rms[:,None,None];tt=time[:,None,None].expand(-1,1,256);noise=(log_variance-2*rms.log()).clamp(-12,8)[:,None,None].expand(-1,1,256)
  h=F.silu(self.entry(torch.cat((state/scale,parent/scale,(residual/scale).clamp(-8,8),tt,noise),1)))
  for block in self.blocks:h=h+block(h)
  reduced=F.avg_pool1d(h,2).transpose(1,2);attended,_=self.attention(reduced,reduced,reduced,need_weights=False);h=h+F.interpolate(attended.transpose(1,2),size=256,mode='linear',align_corners=False)
  return 2*scale*torch.tanh(self.output(h)/2)

def integrate(net,received,parent,action,log_variance,steps):
 if not isinstance(steps,int) or steps<1:raise ValueError('Positive integer integration steps required')
 state=parent
 for k in range(steps):
  t=parent.new_full((len(parent),),k/steps);b=backproject(received,state,action);state=state+net(state,parent,b,t,log_variance)/steps
 return state
