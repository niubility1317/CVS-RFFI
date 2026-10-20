import torch
from torch import nn
from . import design as d
from experiments.pamf_r38_prior_compact.models import CompactSuccessor as Dense,paired_stages,objective
from experiments.pamf_r17_compact.models import FactorConv

def factor_modules(priors,fraction,*,initialize):
 maps=[dict(core.named_modules()) for core in priors];audit=[]
 for name,conv in list(maps[0].items()):
  if not isinstance(conv,nn.Conv1d) or conv.kernel_size[0]<=1:continue
  rank=max(1,int(min(conv.out_channels,conv.in_channels*conv.kernel_size[0])*fraction))
  cost=rank*(conv.in_channels*conv.kernel_size[0]+conv.out_channels)
  if cost>=conv.weight.numel():continue
  first=FactorConv.from_conv(conv,fraction=fraction,initialize=initialize)
  for index,mapping in enumerate(maps):
   old=mapping[name];new=first if index==0 else FactorConv.from_conv(old,fraction=fraction,initialize=False)
   if index:
    new.left.weight=first.left.weight;new.right.weight=first.right.weight
    if initialize and old.bias is not None:
     with torch.no_grad():new.right.bias.copy_(old.bias)
   container,_,leaf=name.rpartition('.');setattr(mapping[container],leaf,new)
  if initialize:
   with torch.no_grad():
    reconstructed=first.right.weight[:,:,0]@first.left.weight.flatten(1)
    relative=float((reconstructed-conv.weight.flatten(1)).square().sum()/conv.weight.square().sum().clamp_min(1e-30))
  else:relative=None
  audit.append(dict(module=name,rank=rank,original_weight_parameters=conv.weight.numel(),factor_weight_parameters=cost,relative_squared_weight_error=relative))
 return audit

class CompactSuccessor(Dense):
 def __init__(self,arm,*,share=True):
  if arm not in d.ARMS:raise ValueError('Registered factor-prior arm required')
  super().__init__('prior_keep',share=share);self.arm=arm
  factor_modules(self.prior,d.FRACTIONS[arm],initialize=False)

def compact_from_parent(parent,arm):
 if getattr(parent,'arm',None)!='dense_half' or sum(p.numel() for p in parent.parameters())!=794529:
  raise ValueError('Original user-designated R38 parent required')
 model=Dense('prior_keep');model.load_state_dict(parent.state_dict(),strict=True);model.arm=arm
 audit=factor_modules(model.prior,d.FRACTIONS[arm],initialize=True)
 model.initialization_report=dict(mode='R38_context_exact_prior_SVD_preserve_channels',level=d.LEVEL[arm],group_count=4,
  IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,
  exact_function_equivalence=False,context_exact=True,channel_dimensions_unchanged=True,normalization_values_exact=True,factor_layers=audit)
 return model
