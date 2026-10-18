"""Use unchanged original complete physical/residual channel and tensor bridge."""
from experiments.cvs_clock_factorial30.channel import Practical
from experiments.cvs_crossview_supcon41.channel import tensor_iq
from .protocol import cumulative
class Shifted:
 def __init__(self,obj):self.obj=obj
 def __getattr__(self,k):
  if k=='obj':raise AttributeError(k)
  return getattr(self.obj,k)
 def apply(self,x,meta,epoch,step):
  epoch,step=cumulative(epoch,step)
  if len(x)!=128:raise ValueError('Only real nonempty full128 on-step channel call')
  return self.obj.apply(x,meta,epoch=epoch,step=step)
