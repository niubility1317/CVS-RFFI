import torch
from torch.nn import functional as F

def sample(cache,rxs,days,count,generator):
 result=[]
 for rx in rxs:
  for day in days:
   for y in range(6):
    pool=torch.where((cache['rx']==rx)&(cache['day']==day)&(cache['y']==y))[0]
    if len(pool)<count:raise ValueError('Insufficient source cell')
    result.append(pool[torch.randperm(len(pool),generator=generator)[:count]])
 return torch.cat(result)

def gradient(loss,parameters):
 g=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True)
 return torch.cat([(torch.zeros_like(p) if x is None else x).detach().reshape(-1) for p,x in zip(parameters,g)])

def direction(g,lr_scale):
 v=g*lr_scale
 if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite update direction')
 return v/v.norm().clamp_min(1e-20)

@torch.no_grad()
def metrics(model,x,y,base_prediction=None):
 z=model(x);pred=z.argmax(1);correct=pred==y
 rival=z.masked_fill(F.one_hot(y,z.shape[1]).bool(),-torch.inf).max(1).values
 value=dict(CE=float(F.cross_entropy(z,y)),accuracy=float(correct.double().mean()),margin=float((z.gather(1,y[:,None]).flatten()-rival).mean()))
 if base_prediction is not None:
  old=base_prediction==y;value.update(fix_count=int((~old&correct).sum()),break_count=int((old&~correct).sum()))
 return value,pred

def virtual(model,parameters,before,vector,delta,probes,baseline):
 offset=0
 try:
  with torch.no_grad():
   for p,v in zip(parameters,before):
    p.copy_(v-delta*vector[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
  result={}
  for role,(x,y) in probes.items():
   m,_=metrics(model,x,y,baseline[role][1]);m['CE_change']=m['CE']-baseline[role][0]['CE'];m['margin_change']=m['margin']-baseline[role][0]['margin'];result[role]=m
  return result
 finally:
  with torch.no_grad():
   for p,v in zip(parameters,before):p.copy_(v)

def select_from_L(candidate_results):
 return min(range(len(candidate_results)),key=lambda i:(candidate_results[i]['L']['CE'],i))
