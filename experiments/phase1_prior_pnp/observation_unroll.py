"""Independent end-to-end, observation-conditioned unrolled recovery."""
import torch
from torch import nn
from torch.nn import functional as F
from signal_prior import Block,observation_step

class ObservationCorrector(nn.Module):
 def __init__(self,width=48):
  super().__init__();self.entry=nn.Conv1d(6,width,9,padding=4);self.blocks=nn.Sequential(*(Block(width,d) for d in (1,2,4,8,16,32)));self.attention=nn.MultiheadAttention(width,4,batch_first=True);self.output=nn.Conv1d(width,2,3,padding=1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,state,direction,logvar,stage):
  if state.shape!=direction.shape or state.ndim!=3 or state.shape[1:]!=(2,256) or logvar.shape!=(len(state),):raise ValueError('Observation-conditioned IQ shape required')
  rms=state.square().mean((1,2),keepdim=True).clamp_min(1e-8).sqrt();lv=(logvar.clamp(-12,4)/12)[:,None,None].expand(-1,1,256);step=state.new_full((len(state),1,256),stage/3);h=F.silu(self.entry(torch.cat((state/rms,direction/rms,lv,step),1)));h=self.blocks(h);q=F.avg_pool1d(h,4).transpose(1,2);v,_=self.attention(q,q,q,need_weights=False);h=h+F.interpolate(v.transpose(1,2),size=256,mode='linear',align_corners=False);return state+.15*rms*torch.tanh(self.output(h))

def restore_observation_conditioned(model,received,initial,action,logvar):
 z=initial
 for stage in (1,2,3):
  observed=observation_step(received,z,action);direction=observed-z;z=model(observed,direction,logvar,stage)
 return z
