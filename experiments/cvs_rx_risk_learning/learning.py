import torch
from torch import nn
from torch.nn import functional as F

class Risk(nn.Module):
 def __init__(self,state_dim):
  super().__init__();self.net=nn.Sequential(nn.Linear(state_dim+8,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),nn.Linear(48,1))
 def forward(self,state,delta):
  repeated=state[:,None].expand(-1,8,-1);value=self.net(torch.cat((repeated,delta),2)).squeeze(-1)
  zero=self.net(torch.cat((repeated,torch.zeros_like(delta)),2)).squeeze(-1)
  return value-zero

def normalizer(state,delta):
 return dict(mean=state.mean(0),std=state.std(0,unbiased=False).clamp_min(1e-4),delta_std=delta.flatten(0,1).std(0,unbiased=False).clamp_min(1e-4))

def normalize(state,delta,n):return (state-n['mean'])/n['std'],delta/n['delta_std']

def loss(scores,CE,margin,kind):
 if kind=='CE':return F.smooth_l1_loss(scores,CE)
 if kind=='MARGIN':return F.smooth_l1_loss(scores,margin)
 if kind!='RANK':raise ValueError('Unknown supervision')
 i,j=torch.triu_indices(8,8,1,device=scores.device);diff=margin[:,i]-margin[:,j];mask=diff.abs()>.01
 penalties=F.softplus(-diff.sign()*(scores[:,i]-scores[:,j]))
 return (penalties*mask).sum()/mask.sum().clamp_min(1)

def probabilities(scores,standardized):
 if standardized:scores=(scores-scores.mean(1,keepdim=True))/scores.std(1,unbiased=False,keepdim=True).clamp_min(1e-6)
 else:scores=scores/.25
 return .5/8+.5*scores.softmax(1)

def fit(data,arm,seed,device,steps=2000):
 kind,state_kind=arm.split('_');dim=40 if state_kind=='PHYS' else 206
 state=data['state'][:,:dim].to(device);delta=data['delta'].to(device);CE=data['CE'].to(device);margin=data['margin'].to(device)
 n=normalizer(state,delta);state,delta=normalize(state,delta,n)
 torch.manual_seed(seed);model=Risk(dim).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 generator=torch.Generator().manual_seed(seed+101);order=torch.randperm(len(state),generator=generator);cursor=0;records=[]
 for step in range(1,steps+1):
  if cursor+32>len(order):order=torch.randperm(len(state),generator=generator);cursor=0
  idx=order[cursor:cursor+32].to(device);cursor+=32;optimizer.zero_grad(set_to_none=True)
  scores=model(state[idx],delta[idx]);value=loss(scores,CE[idx],margin[idx],kind)
  if not torch.isfinite(value):raise FloatingPointError('Nonfinite auxiliary loss')
  value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True);optimizer.step()
  record=dict(step=step,loss=float(value.detach()),gradient_norm=float(norm),L_packets=32,learning_rate=optimizer.param_groups[0]['lr']);records.append(record)
  if step%200==0:print('AUX_STEP',arm,record,flush=True)
 return model.eval(),n,records

@torch.no_grad()
def predict(model,n,data,arm,device,shuffled=False):
 dim=40 if arm.endswith('PHYS') else 206;state=data['state'][:,:dim].clone()
 if shuffled:
  if dim!=206:raise ValueError('Only decision condition can be shuffled')
  state[:,40:]=state[:,40:].roll(1,0)
 result=[]
 for start in range(0,len(state),512):
  s,delta=normalize(state[start:start+512].to(device),data['delta'][start:start+512].to(device),n);result.append(model(s,delta).cpu())
 return torch.cat(result)
