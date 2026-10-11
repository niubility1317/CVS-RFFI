import torch


class TemporarySAM:
 """Exact parameter restore, then one AdamW update owned by the caller."""
 def __init__(self,groups,rho):
  self.parameters=[p for group in groups for p in group['params']];self.rho=rho;self.saved=[]
 def perturb(self):
  if self.saved:raise RuntimeError('SAM perturbation already active')
  norm=sum(p.grad.detach().square().sum() for p in self.parameters if p.grad is not None).sqrt()
  if not torch.isfinite(norm):raise FloatingPointError('Nonfinite SAM gradient')
  with torch.no_grad():
   for p in self.parameters:
    self.saved.append((p,p.detach().clone()))
    if p.grad is not None:p.add_(p.grad,alpha=self.rho/(float(norm)+1e-12))
  return float(norm)
 def restore(self):
  if not self.saved:raise RuntimeError('No SAM perturbation to restore')
  with torch.no_grad():
   for p,value in self.saved:p.copy_(value)
  self.saved=[]


class WeightAverage:
 def __init__(self,model,parameter_names):
  self.names=set(parameter_names);self.values={};self.count=0;self.epochs=[]
  if not self.names:raise ValueError('Empty average parameter set')
 def add(self,model,epoch):
  if epoch in self.epochs:raise ValueError('Repeated fixed average epoch')
  with torch.no_grad():
   for name,p in model.named_parameters():
    if name in self.names:
     if not self.count:self.values[name]=p.detach().clone()
     else:self.values[name].add_((p.detach()-self.values[name])/(self.count+1))
  self.count+=1;self.epochs.append(epoch)
 def apply(self,model,expected_epochs):
  if self.epochs!=expected_epochs or set(self.values)!=self.names:raise ValueError('Incomplete fixed averaging trajectory')
  with torch.no_grad():
   for name,p in model.named_parameters():
    if name in self.values:p.copy_(self.values[name])
