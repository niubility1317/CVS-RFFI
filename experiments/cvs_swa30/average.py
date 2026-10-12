import torch
class MeanParameters:
 def __init__(self,model,pg):
  owned={id(p) for g in pg for p in g['params']};self.names=[n for n,p in model.named_parameters() if id(p) in owned];self.mean={};self.count=0;self.snapshots=[]
 @torch.no_grad()
 def update(self,model):
  self.count+=1;params=dict(model.named_parameters());self.snapshots.append({name:params[name].detach().cpu().clone() for name in self.names})
  for name in self.names:
   value=params[name].detach()
   if name not in self.mean:self.mean[name]=value.clone()
   else:self.mean[name].add_(value-self.mean[name],alpha=1/self.count)
 def state(self,model):
  if self.count!=10:raise ValueError('SWA requires all10registered snapshots')
  out={k:v.detach().clone() for k,v in model.state_dict().items()}
  for name in self.names:out[name]=self.mean[name].clone()
  return out
