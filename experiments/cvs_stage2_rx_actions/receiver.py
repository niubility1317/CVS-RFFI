import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_rx_spectral_audit.spectral import observations,grid,fit_contrast
from experiments.cvs_rx_intervention_audit.actions import apply
from experiments.cvs_report_ssl.physical import signal_state

class ReceiverActions(nn.Module):
 def __init__(self,c,L):
  super().__init__();self.plan=c['rx_action'];self.linear=None;self.repetition=None;self.critic=None
  if not self.plan['enabled']:return
  values=observations(L['x'],'rectangular')['DCT1to8'];means,_,labels=grid(values,L['y'],L['rx'],L['day'])
  effects=torch.stack([torch.stack([fit_contrast(means,t,d) for d in range(3)]) for t in range(6)])
  ueffects=[]
  for day in range(3):
   value=means[:,torch.arange(3)!=day].mean((0,1));ueffects.append(value-value.mean(0,keepdim=True))
  for name,tensor in [('effects',effects.float()),('U_effects',torch.stack(ueffects).float()),('TX',torch.tensor(labels['TX'])),('RX',torch.tensor(labels['RX'])),('day',torch.tensor(labels['day']))]:self.register_buffer(name,tensor)
  if self.plan['critic']:self.critic=nn.Sequential(nn.Linear(48,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),nn.Linear(48,1))

 def candidates(self,cache,indices,labeled):
  device=self.effects.device;rx=cache['rx'][indices].to(device);day=cache['day'][indices].to(device)
  r=torch.searchsorted(self.RX,rx);d=torch.searchsorted(self.day,day);targets=(r[:,None]+torch.arange(1,5,device=device))%5
  if labeled:
   y=cache['y'][indices].to(device);t=torch.searchsorted(self.TX,y);base=self.effects[t,d,r];destination=self.effects[t[:,None],d[:,None],targets]
  else:base=self.U_effects[d,r];destination=self.U_effects[d[:,None],targets]
  delta=destination-base[:,None]
  if self.plan['kind']=='rotated':delta=delta.roll(4,-1)
  return torch.stack((.5*delta,delta),dim=2).reshape(len(indices),8,8)

 def risks(self,x,delta):
  state=signal_state(x).detach();base=self.critic(torch.cat((state,torch.zeros_like(delta)),1))
  return (self.critic(torch.cat((state,delta.detach()),1))-base).flatten()

 @torch.no_grad()
 def choose(self,x,cache,indices,labeled,step,generator,uniform=False):
  candidates=self.candidates(cache,indices,labeled);probs=x.new_full((len(x),8),1/8)
  active=self.critic is not None and step>888+self.plan['critic_uniform_warmup_updates'] and not uniform
  if active:
   state=signal_state(x).detach();inputs=torch.cat((state[:,None].expand(-1,8,-1),candidates),2)
   risk=self.critic(inputs).squeeze(-1);soft=(risk/self.plan['critic_temperature']).softmax(1)
   probs=self.plan['critic_uniform_probability']/8+(1-self.plan['critic_uniform_probability'])*soft
  selected=torch.multinomial(probs.cpu(),1,generator=generator).flatten().to(x.device)
  delta=candidates[torch.arange(len(x),device=x.device),selected]
  changed,info=apply(x,delta,1.,self.plan['taps'],self.plan['max_amplitude_dB'])
  return changed,delta,dict(critic_selection_active=active,full_strength_count=int((selected%2==1).sum()),selection_entropy=float(-(probs*probs.log()).sum(1).mean()),clipped_bin_fraction=float(info['clipped_fraction'].mean()))

 def critic_objective(self,model,x,y,cache,indices,step,generator):
  changed,delta,_=self.choose(x,cache,indices,True,step,generator,uniform=True)
  modes=[(m,m.training) for m in model.modules()]
  try:
   model.eval()
   with torch.no_grad():target=F.cross_entropy(model(changed),y,reduction='none')-F.cross_entropy(model(x),y,reduction='none')
  finally:
   for m,mode in modes:m.training=mode
  predicted=self.risks(x,delta);loss=F.smooth_l1_loss(predicted,target)
  return loss,dict(batch=len(x),target_CE_increment_mean=float(target.mean()),prediction_mean=float(predicted.detach().mean()),absolute_error_mean=float((predicted.detach()-target).abs().mean()),identity_target_detached=not target.requires_grad)
