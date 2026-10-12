"""Fit-only standardization; strong bilinear and lowrank controls vs MLP."""
import math,time,torch
from torch import nn
def normfit(v):return dict(mean=v.mean(0),scale=v.std(0,unbiased=False).clamp_min(1e-4))
def normalize(v,s):return (v-s['mean'])/s['scale']
def inputs(h,q,a,state,bilinear=False,parameter=False):
 hh=normalize(h,state['h']);qq=normalize(q,state['q']);aa=normalize(a,state['a'])
 if parameter:return torch.cat((torch.ones_like(aa[:,:1]),aa),1)
 values=[hh,qq,aa]
 if bilinear:values.extend([(hh[:,:,None]*aa[:,None,:]).flatten(1),(qq[:,:,None]*aa[:,None,:]).flatten(1)])
 return torch.cat((torch.ones_like(aa[:,:1]),*values),1)
def ridge(x,y,relative):
 xx=x.double();yy=y.double();gram=xx.T@xx/len(xx);penalty=relative*gram.diag().mean().clamp_min(1e-12)
 return torch.linalg.solve(gram+penalty*torch.eye(len(gram),device=x.device,dtype=torch.float64),xx.T@yy/len(xx)).float()
class Action(nn.Module):
 def __init__(self,n,d,width,noise):
  super().__init__();self.noise=noise;self.layers=nn.Sequential(nn.Linear(n,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,d*(2 if noise else 1)))
 def forward(self,x):
  v=self.layers(x)
  if not self.noise:return v,torch.zeros_like(v)
  mu,logv=v.chunk(2,1);return mu,logv.clamp(-10,6).exp()
def fit(h,q,a,delta,var,c,kind,seed,device,progress=None):
 # Every input is already a cached detached reference readout.
 if any(v.requires_grad for v in (h,q,a,delta,var)):raise ValueError('Frozen observations required')
 state={k:normfit(v.to(device)) for k,v in [('h',h),('q',q),('a',a),('y',delta)]}
 hd,qd,ad=h.to(device),q.to(device),a.to(device);target=normalize(delta.to(device),state['y']);variance=var.to(device)/state['y']['scale'].square()
 x=inputs(hd,qd,ad,state);xb=inputs(hd,qd,ad,state,bilinear=True);xp=inputs(hd,qd,ad,state,parameter=True)
 bp=ridge(xp,target,c['ridge_relative']);bb=ridge(xb,target,c['ridge_relative'])
 # Reduced-rank output directions are fitted to the same training targets.
 _,vectors=torch.linalg.eigh(target.double().T@target.double()/len(target));rank=min(c['lowrank'],target.shape[1]);proj=vectors[:,-rank:].float();bl=ridge(xb,target@proj,c['ridge_relative'])@proj.T
 gen=torch.Generator().manual_seed(seed);torch.manual_seed(seed);net=Action(x.shape[1],target.shape[1],c['hidden_width'],kind=='noise').to(device);opt=torch.optim.AdamW(net.parameters(),lr=c['fit_lr'],weight_decay=c['fit_weight_decay'])
 logs=[];updates=0
 for epoch in range(1,c['fit_epochs']+1):
  epoch_start=time.perf_counter();order=torch.randperm(len(x),generator=gen);loss_sum=0.;packets=0;terms=[0.,0.,0.]
  for start in range(0,len(order),c['fit_batch_size']):
   idx=order[start:start+c['fit_batch_size']].to(device);mu,v=net(x[idx]);error=(mu-target[idx]).square()
   if kind=='noise':
    vv=v.clamp_min(1e-8);components=(error.div(vv).mean()/2,variance[idx].div(vv).mean()/2,vv.log().mean()/2);value=sum(components)
   else:components=(error.mean(),torch.zeros((),device=device),torch.zeros((),device=device));value=components[0]
   if not torch.isfinite(value):raise FloatingPointError('Action predictor loss')
   opt.zero_grad(set_to_none=True);value.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
   if not torch.isfinite(norm):raise FloatingPointError('Action predictor gradient')
   opt.step();updates+=1;loss_sum+=float(value.detach())*len(idx);packets+=len(idx)
   for i,component in enumerate(components):terms[i]+=float(component.detach())*len(idx)
  record=dict(epoch=epoch,loss=loss_sum/packets,mean_error_term=terms[0]/packets,repeat_variance_term=terms[1]/packets,log_variance_term=terms[2]/packets,gradient_norm_last=float(norm),optimizer_steps=updates,fit_observations=len(x),fit_packets=packets,lr=c['fit_lr'],epoch_seconds=time.perf_counter()-epoch_start,identity_updates=0,U_fit=False);logs.append(record)
  if progress:progress(record)
 net.eval().requires_grad_(False)
 pooled_var=variance.mean(0)
 return dict(state=state,parameter=bp,bilinear=bb,lowrank=bl,net=net,pooled_var=pooled_var,logs=logs,updates=updates,rank=rank,parameters=sum(p.numel() for p in net.parameters()))
@torch.no_grad()
def predict(model,h,q,a,method,device,batch=512):
 state=model['state'];mus=[];vs=[]
 for start in range(0,len(h),batch):
  hh,qq,aa=h[start:start+batch].to(device),q[start:start+batch].to(device),a[start:start+batch].to(device)
  if method=='zero':mu=-state['y']['mean'][None]/state['y']['scale'][None];mu=mu.expand(len(hh),-1);v=model['pooled_var'][None].expand_as(mu)
  elif method=='MLP':mu,v=model['net'](inputs(hh,qq,aa,state))
  else:
   key={'parameter_linear':'parameter','bilinear_ridge':'bilinear','lowrank8':'lowrank'}[method];mu=inputs(hh,qq,aa,state,bilinear=key!='parameter',parameter=key=='parameter')@model[key];v=model['pooled_var'][None].expand_as(mu)
  mus.append((mu*state['y']['scale']+state['y']['mean']).cpu());vs.append((v*state['y']['scale'].square()).cpu())
 return torch.cat(mus),torch.cat(vs)
