import torch
from torch import nn
from . import design as d
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer,transplant_core
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r17_compact.models import stages
class CompactSuccessor(JointCompactRestorer):
 def __init__(self,arm,*,share=True):
  if arm not in d.ARMS:raise ValueError('Registered prior compression arm required')
  super().__init__('dense_half',share=False);self.arm=arm
  self.prior=nn.ModuleList([Core(incoming=9,widths=d.WIDTHS[arm]) for _ in range(3)])
  if share:share_long_weights(self.prior)
  self.frozen.eval().requires_grad_(False);self.prior.requires_grad_(True)
 def train(self,mode=True):
  super().train(mode);self.frozen.eval();return self
 @torch.no_grad()
 def estimate_context(self,received):return super().estimate_context(received)
def compact_from_parent(parent,arm):
 if getattr(parent,'arm',None)!='dense_half' or sum(p.numel() for p in parent.parameters())!=794529:
  raise ValueError('Original designated 794529 R38 parent required')
 model=CompactSuccessor(arm,share=d.LEVEL[arm]=='keep')
 model.frozen.load_state_dict(parent.frozen.state_dict(),strict=True)
 if d.LEVEL[arm]=='keep':model.prior.load_state_dict(parent.prior.state_dict(),strict=True)
 else:
  for source,target in zip(parent.prior,model.prior):transplant_core(source,target)
  share_long_weights(model.prior)
 model.initialization_report=dict(mode='R38_context_exact_prior_GN4_slice',level=d.LEVEL[arm],group_count=4,
  IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,
  exact_function_equivalence=d.LEVEL[arm]=='keep',context_exact=True)
 return model
def paired_stages(model,teacher,received):
 with torch.no_grad():
  context=teacher.estimate_context(received)
  desired=stages(teacher.prior,received,context)[0]
 return stages(model.prior,received,context)[0],desired
def objective(outputs,desired,clean):
 if len(outputs)!=3 or len(desired)!=3 or any(x.shape!=clean.shape for x in (*outputs,*desired)):
  raise ValueError('Three matched source IQ stages required')
 energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
 paired=(outputs[-1]-clean.detach()).square().mean((1,2))/energy
 fidelity=torch.stack([(x-y.detach()).square().mean((1,2))/energy for x,y in zip(outputs,desired)]).mean(0)
 return paired+fidelity,dict(iq_nmse=paired,teacher_stage_nmse=fidelity)
