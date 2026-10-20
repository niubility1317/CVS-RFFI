from . import design as d
from experiments.pamf_r38_fidelity_only.models import CompactSuccessor as Previous,compact_from_parent as original_parent,paired_stages,objective
class CompactSuccessor(Previous):
 def __init__(self,arm,*,share=True):
  if arm not in d.ARMS:raise ValueError('Registered dense-step arm required')
  super().__init__(d.STRUCTURE[arm],share=share);self.arm=arm
def compact_from_parent(parent,arm):
 if arm not in d.ARMS:raise ValueError('Registered dense-step arm required before IO')
 model=original_parent(parent,d.STRUCTURE[arm]);model.arm=arm
 return model
