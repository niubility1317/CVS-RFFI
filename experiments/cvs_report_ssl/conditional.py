"""Snapshot-coherent class/condition means and shrunk diagonal metric.

 L and U means remain separate; U never replaces L priors and alpha<=.2.
 No high-dimensional bare inverse, estimated CFO or U TX labels.
 """
from copy import deepcopy
import torch
from torch.nn import functional as F
from .physical import nonlinear_memory_features
from .auxiliary import observable_weights

@torch.no_grad()
def features(model,x):
 _,z,h=model.representation(x)
 return torch.cat((z,h[:,-29:-5],nonlinear_memory_features(x)),1)

class ConditionalTeacher:
 def __init__(self,teacher,L,U,device,alpha_cap=.2):
  if not 0<=alpha_cap<=.2:raise ValueError('U correction cannot dominate L anchors')
  self.model=deepcopy(teacher.model).eval().requires_grad_(False)
  self.keys=sorted(set(zip(L['rx'].tolist(),L['day'].tolist())))
  if len(self.keys)!=15:raise ValueError('Expected15 legal source RX/day conditions')
  lv=[];uv=[];uq=[];uw=[]
  for start in range(0,len(L['x']),256):lv.append(features(self.model,L['x'][start:start+256].to(device)).cpu())
  for start in range(0,len(U['x']),256):
   x=U['x'][start:start+256].to(device);uv.append(features(self.model,x).cpu());uq.append(self.model(x).softmax(1).cpu());uw.append(observable_weights(x,'repetition').cpu())
  lv=torch.cat(lv);uv=torch.cat(uv);uq=torch.cat(uq);uw=torch.cat(uw)
  self.offset=lv.mean(0);self.scale=lv.std(0,unbiased=False).clamp_min(1e-3)
  lv=(lv-self.offset)/self.scale;uv=(uv-self.offset)/self.scale
  ly=L['y'];global_mu=torch.stack([lv[ly==y].mean(0) for y in range(6)])
  global_var=(lv-global_mu[ly]).square().mean(0).clamp_min(.01)
  self.shared=global_mu;self.means_L=[];self.means_U=[];self.means=[];self.variances=[];self.priors=[];self.alphas=[];self.counts=[]
  for rx,day in self.keys:
   lm=(L['rx']==rx)&(L['day']==day);um=(U['rx']==rx)&(U['day']==day)
   counts=torch.bincount(ly[lm],minlength=6)
   if not bool((counts==70).all()):raise ValueError('L condition label roles differ')
   # First condition model: shared TX prototype + additive RX/date offsets.
   # Unrestricted TX/RX interaction is deliberately deferred.
   rx_offset=(lv[L['rx']==rx]-global_mu[ly[L['rx']==rx]]).mean(0)
   day_offset=(lv[L['day']==day]-global_mu[ly[L['day']==day]]).mean(0)
   mu=global_mu+rx_offset+day_offset
   variance=(lv[lm]-mu[ly[lm]]).square().mean(0)
   variance=.5*variance+.5*global_var+.01
   weight=uq[um]*uw[um,None];mass=weight.sum(0).clamp_min(1e-8)
   mu_u=weight.T@uv[um]/mass[:,None]
   disagreement=((mu_u-mu).square()/variance).mean(1)
   alpha=alpha_cap*torch.exp(-disagreement)
   self.means_L.append(mu);self.means_U.append(mu_u);self.means.append((1-alpha[:,None])*mu+alpha[:,None]*mu_u)
   self.variances.append(variance);self.priors.append(counts.float()/counts.sum());self.alphas.append(alpha)
   self.counts.append(dict(rx=rx,day=day,L=counts.tolist(),U_count=int(um.sum()),soft_mass=mass.tolist(),ESS=(mass.square()/weight.square().sum(0).clamp_min(1e-8)).tolist(),alpha=alpha.tolist(),U_L_disagreement=disagreement.tolist()))
  for key in ('shared','means_L','means_U','means','variances','priors','alphas'):
   value=getattr(self,key);setattr(self,key,(torch.stack(value) if isinstance(value,list) else value).to(device))
  self.offset=self.offset.to(device);self.scale=self.scale.to(device)

 @torch.no_grad()
 def probabilities(self,x,rx,day,mode='condition_U'):
  value=(features(self.model,x)-self.offset)/self.scale
  indices=torch.tensor([self.keys.index((r,d)) for r,d in zip(rx.cpu().tolist(),day.cpu().tolist())],device=x.device)
  mu=self.shared[None].expand(len(x),-1,-1) if mode=='shared' else (self.means_L if mode=='condition_L' else self.means)[indices]
  variance=self.variances[indices]
  scores=-.5*((value[:,None]-mu).square()/variance[:,None]).sum(2)+self.priors[indices].log()
  return scores.softmax(1)

 def state(self):
  return dict(version='report_no_explicit_cfo_v1',model=self.model.state_dict(),keys=self.keys,offset=self.offset.cpu(),scale=self.scale.cpu(),means_L=self.means_L.cpu(),means_U=self.means_U.cpu(),means=self.means.cpu(),diagonal_covariance=self.variances.cpu(),fixed_L_priors=self.priors.cpu(),alphas=self.alphas.cpu(),statistics=self.counts,single_packet_repetition_covariance_rank_max=3,L_cell_covariance_rank_max=69,coordinate_snapshot_fixed=True,identity_confidence_mask_reused=False)
