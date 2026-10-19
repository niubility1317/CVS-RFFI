"""Discarded fresh-AdamW responses using the real fixed F physical Sinc policy.

This is a reset-state counterfactual, not a reconstruction of the old optimizer
trajectory or a trained model. No original parameter, grad, EMA or RNG is written.
"""
from copy import deepcopy
import math
import torch
from experiments.cvs_rdrop_sinc_units import optimizer as units

def discardable_step(model,gradients,*,lr=2e-4,weight_decay=1e-4):
 named=dict(model.named_parameters())
 if set(gradients)!=set(named) or not math.isfinite(lr) or lr!=2e-4 or weight_decay!=1e-4:
  raise ValueError('Fixed complete named-gradient map and preregistered reset AdamW required')
 if any(p.grad is not None for p in named.values()):raise ValueError('Source checkpoint .grad must remain empty')
 for k,g in gradients.items():
  if g is not None and (g.shape!=named[k].shape or g.device!=named[k].device or g.dtype!=named[k].dtype or not torch.isfinite(g).all()):
   raise ValueError('Finite shape/device/dtype-matched source gradients required')
 clone=deepcopy(model);cn=dict(clone.named_parameters())
 if any(cn[k].data_ptr()==p.data_ptr() for k,p in named.items()):raise ValueError('Discarded model aliases the frozen source')
 for k,g in gradients.items():cn[k].grad=None if g is None else g.detach().clone()
 opt=units.build(clone,clone.parameters(),'CENTER_UNITS',lr=lr,weight_decay=weight_decay,betas=(.9,.999),eps=1e-8)
 if opt.state:raise ValueError('Probe optimizer must start with no moments')
 norm=units.clip_physical(opt,clone.parameters(),5.)
 opt.step();opt.zero_grad(set_to_none=True)
 if opt._sinc_unit_steps!=1 or opt._sinc_physical_clip_calls!=1 or any(p.grad is not None for p in clone.parameters()):raise ValueError('Exactly one physical-unit reset update required')
 changed={k:float((p.detach()-named[k].detach()).float().norm()) for k,p in cn.items() if not torch.equal(p,named[k])}
 if not changed or not any(k.startswith(units.PREFIX) for k in changed) or not any(k.endswith('cls_head.weight') for k in changed):raise ValueError('Virtual Sinc/body/defaulthead response must actually update')
 return clone,dict(virtual_optimizer='fresh AdamW; not saved original optimizer',virtual_updates=1,persistent_original_updates=0,base_lr=lr,weight_decay=weight_decay,global_physical_clip=5.,physical_gradient_norm=float(norm),unit_step=opt._sinc_unit_lr_records[0],changed_parameter_norms=changed,default_head_joint=True,deploy_parameters_added=0)
