import torch
from . import design as d
from experiments.pamf_r38_prior_compact.models import CompactSuccessor as Dense,compact_from_parent as narrow_parent,paired_stages
from experiments.pamf_r38_factor_prior.models import factor_modules

class CompactSuccessor(Dense):
 def __init__(self,arm,*,share=True):
  if arm not in d.ARMS:raise ValueError('Registered fidelity-only arm required')
  super().__init__('prior_three_quarter' if arm=='dense750_fidelity' else 'prior_keep',share=share);self.arm=arm
  if d.FRACTIONS[arm] is not None:factor_modules(self.prior,d.FRACTIONS[arm],initialize=False)

def compact_from_parent(parent,arm):
 if arm not in d.ARMS:raise ValueError('Registered fidelity-only arm required before initialization')
 if getattr(parent,'arm',None)!='dense_half' or sum(p.numel() for p in parent.parameters())!=794529:
  raise ValueError('Original designated R38 parent required')
 if arm=='dense750_fidelity':
  model=narrow_parent(parent,'prior_three_quarter');model.arm=arm
 else:
  model=Dense('prior_keep');model.load_state_dict(parent.state_dict(),strict=True);model.arm=arm
  audit=factor_modules(model.prior,d.FRACTIONS[arm],initialize=True)
  model.initialization_report=dict(mode=d.INITIALIZATION_MODE[arm],level=d.LEVEL[arm],group_count=4,
   IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,
   exact_function_equivalence=False,context_exact=True,channel_dimensions_unchanged=True,normalization_values_exact=True,factor_layers=audit)
 return model

def objective(outputs,desired,clean):
 if len(outputs)!=3 or len(desired)!=3 or any(x.shape!=clean.shape for x in (*outputs,*desired)):
  raise ValueError('Three matched source IQ stages required')
 energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
 fidelity=torch.stack([(x-y.detach()).square().mean((1,2))/energy for x,y in zip(outputs,desired)]).mean(0)
 paired=(outputs[-1].detach()-clean.detach()).square().mean((1,2))/energy
 return fidelity,dict(iq_nmse=paired,teacher_stage_nmse=fidelity)
