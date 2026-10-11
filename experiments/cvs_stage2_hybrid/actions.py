"""Separate version; remove sigmoid offset without changing physical limits."""
import math
import torch
from . import design as d
from experiments.cvs_stage2_geometry import actions as a
from experiments.cvs_stage2_geometry.policy_audit import search_loss
from experiments.cvs_stage2_decouple.common import margins
from experiments.cvs_stage2_decouple.source_counterfactual_audit import ReadOnlyState

VERSION=3


class BalancedAction(torch.nn.Module):
 def __init__(self,legacy):
  super().__init__();self.kind=legacy.kind;self.state_norm=legacy.state_norm;self.net=legacy.net
 def forward(self,state,code):
  if state.shape!=(len(code),40) or code.shape!=(len(state),8):raise ValueError('State40/code8 required')
  raw=self.net(torch.cat((self.state_norm(state),code),1)).tanh()
  if self.kind=='linear':return raw
  return torch.stack((raw[:,0]*a.old.PHASE_LIMIT_RAD,raw[:,1]*a.old.CFO_LIMIT_HZ,
   raw[:,2].abs()*a.old.PHASE_NOISE_LIMIT_RAD,torch.zeros_like(raw[:,0]),raw[:,3]*a.old.DELAY_LIMIT_SAMPLES),1)


def new_net(kind,generator,device):return BalancedAction(a.old._new_net(kind,generator,device))


def contract():
 return dict(version=VERSION,physical=a.old.action_contract(),coordinates='all four tanh(raw); temporal noise abs; then registered Linfinity normalization',
  fit='Original E200 frozen, fresh initialization, full L_s raw-margin pass then finite-risk pass',
  fit_budget=dict(passes=2,batch_size=128,lr=2e-4,weight_decay=1e-4),mixture=.5,
  identity_training='frozen actions; no online optimizer, U, teacher or satellite')


def fit(parent,cache,seed,device):
 a.old._validate_cache(cache)
 if cache.get('role')!='L_s':raise ValueError('L_s only auxiliary fit')
 networks={};logs=[]
 with ReadOnlyState(parent) as state:
  parent.eval().requires_grad_(False)
  for ki,kind in enumerate(a.KINDS):
   g=a._generator(seed+17041*(ki+1));net=new_net(kind,g,device)
   optimizer=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
   for epoch,objective in enumerate(('raw','risk'),1):
    order=torch.randperm(len(cache['y']),generator=g)
    for start in range(0,len(order),128):
     ix=order[start:start+128];x=cache['x'][ix].to(device);y=cache['y'][ix].to(device)
     optimizer.zero_grad(set_to_none=True)
     proposal=a._proposal(net,x,kind,g,'conditional',cache['state'][ix].to(device))
     ax=a.apply_proposal(x,proposal);u=margins(parent(ax),y)
     loss=search_loss(cache['parent_margin'][ix].to(device),u,objective)
     loss.backward();norm=sum(p.grad.detach().square().sum() for p in net.parameters() if p.grad is not None).sqrt()
     if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite balanced action fit')
     optimizer.step();logs.append(dict(kind=kind,pass_index=epoch,objective=objective,samples=len(ix),loss=float(loss.detach()),gradient_norm=float(norm)))
   networks[kind]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
 return dict(version=VERSION,contract=contract(),seed=seed,source_ids=list(cache['ids']),networks=networks,
  fit_logs=logs,auxiliary_updates=4*math.ceil(len(cache['y'])/128),source_occurrences=4*len(cache['y']),
  identity_state=state.report,target_access=False,identity_updates=0)


class FrozenBalanced:
 def __init__(self,payload,device):
  if payload.get('contract')!=contract() or payload.get('version')!=VERSION:raise ValueError('Balanced action contract differs')
  self.networks={kind:new_net(kind,a._generator(payload['seed']+ki),device) for ki,kind in enumerate(a.KINDS)}
  for kind,net in self.networks.items():net.load_state_dict(payload['networks'][kind]);net.eval().requires_grad_(False)
 @torch.no_grad()
 def transform(self,x,kind,generator,policy='conditional'):
  p=a._proposal(self.networks[kind],x,kind,generator,policy);ax=a.apply_proposal(x,p)
  stats=a.proposal_statistics(x,ax,p)
  if not stats['finite'] or not stats['within_registered_bounds']:raise ValueError('Action outside old physical bounds')
  return ax.detach(),stats
