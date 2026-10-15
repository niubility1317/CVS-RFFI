"""Reuse the released reference-gradient operator; add one fixed-gain control."""
import torch
from experiments.cvs_sigtime_rdrop_center20 import mechanism as base
from .design import ARMS,FIXED_GAIN
trace,penalty,boundary_geometry=base.trace,base.penalty,base.boundary_geometry

def new_audit():
 return dict(base.new_audit(),applied_gain_sum=0.,applied_gain_min=None,applied_gain_max=None)

def consistency(a,b,z1,z2,W,arm,audit):
 if arm not in ARMS:raise ValueError('Only preregistered gain arms allowed')
 before=audit['auxiliary_loss_sum']
 mapped={'FIXED11':'POINT','DYNAMIC':'NORM_MATCH'}.get(arm,arm)
 loss,parts=base.consistency(a,b,z1,z2,W,mapped,audit)
 if arm=='FIXED11':
  aux=FIXED_GAIN*parts['auxiliary_graph']
  # Recompute the differentiable original KL; its returned metric is detached.
  p,q=a.log_softmax(1),b.log_softmax(1)
  native=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
  loss=native+aux
  audit['auxiliary_loss_sum']=before+float(aux.detach())
  parts.update(auxiliary_loss=aux.detach(),consistency_loss=loss.detach(),auxiliary_graph=aux)
 gain=parts['norm_ratio'] if arm=='DYNAMIC' else z1.new_tensor(0 if arm=='FULL' else FIXED_GAIN if arm=='FIXED11' else 1)
 gain=float(gain)
 audit['applied_gain_sum']+=gain
 audit['applied_gain_min']=gain if audit['applied_gain_min'] is None else min(audit['applied_gain_min'],gain)
 audit['applied_gain_max']=gain if audit['applied_gain_max'] is None else max(audit['applied_gain_max'],gain)
 if not bool(torch.isfinite(loss)):raise ValueError('Finite gain consistency required')
 parts['actual_auxiliary_gain']=gain
 return loss,parts
