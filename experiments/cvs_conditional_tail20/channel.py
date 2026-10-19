from experiments.cvs_F_nontrue20.channel import tensor_iq
from experiments.cvs_fullsource_concat20.channel import Practical

class SecondPractical(Practical):
 def apply(self,x,meta,epoch,step):
  if step not in range(1,4441):raise ValueError('Declared native E20 optimizer step required')
  return super().apply(x,meta,epoch,4441-step)
