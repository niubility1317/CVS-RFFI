"""Keep native RNG draws;selectively bypass dropout only inside satellite L views."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
class Engine:
 def __init__(self,mode):
  if mode not in ('AS_CE','AS_DET'):raise ValueError('Fixed satellite mode required')
  self.mode=mode;self.active=False;self.frame=0;self.mask=None;self.calls=0;self.bypassed_elements=0
 def apply(self,module,x):
  dropped=F.dropout(x,module.p,module.training,module.inplace)
  if self.active and self.frame in (2,3) and module.training and module.p>0:
   if x.ndim<1 or len(x)!=128 or module.inplace:raise ValueError('Actual perpacket L128 noninplace dropout required')
   self.calls+=1
   if self.mode=='AS_DET':
    mask=self.mask.reshape((128,)+(1,)*(x.ndim-1));self.bypassed_elements+=int(x.numel()//2);return torch.where(mask,x,dropped)
  return dropped
 @contextmanager
 def scope(self,mask):
  if self.active or mask.shape!=(128,) or mask.dtype!=torch.bool or int(mask.sum())!=64:raise ValueError('Only64 actual satellite L rows')
  self.active=True;self.mask=mask;self.frame=0;self.calls=0;self.bypassed_elements=0
  try:yield
  finally:self.active=False;self.mask=None;self.frame=0

def forward(self,x):return self._sat_engine.apply(self,x)
def install(model,mode):
 e=Engine(mode);inventory=[]
 for name,module in model.named_modules():
  if isinstance(module,torch.nn.Dropout) and name.startswith('id_backbone.encoder.'):
   if module.inplace or not 0<=module.p<1 or hasattr(module,'_sat_engine'):raise ValueError('Unsupported/repeated dropout installation')
   inventory.append(dict(name=name,p=module.p));module._sat_engine=e;module.forward=MethodType(forward,module)
 if not inventory:raise ValueError('Existing dropout required')
 model._sat_engine=e;model._sat_inventory=inventory;return e
