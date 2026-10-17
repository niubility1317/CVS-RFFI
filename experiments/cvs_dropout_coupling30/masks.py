"""L-only mask coupling with identical finite-uniform marginals and RNG work."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
SIZE=2**24
VIEWS=('O1','O2','S1','S2')
class Engine:
 def __init__(self,mode,seed):
  if mode not in ('INDEPENDENT','ANTITHETIC') or type(seed) is not int:raise ValueError('Fixed mask law required')
  self.mode=mode;self.seed=seed;self.step=None;self.slot=None;self.next_slot=0;self.cache={};self.trace=[];self.cursor=0;self.occ={};self.steps=0;self.records=[];self.random_elements=0
 def begin(self,step):
  if self.step is not None or self.cache or type(step) is not int or step!=self.steps+1:raise ValueError('One ordered L mask context required')
  self.step=step;self.next_slot=0;self.trace=[];self.random_elements=0
 @contextmanager
 def view(self,name):
  if self.step is None or self.slot is not None or self.next_slot>=4 or name!=VIEWS[self.next_slot]:raise ValueError('Four ordered L views required')
  self.slot=self.next_slot;self.cursor=0;self.occ={}
  try:
   yield
   if self.slot and self.cursor!=len(self.trace):raise ValueError('Dropout call trace differs between L views')
   self.next_slot+=1
  finally:self.slot=None
 def apply(self,index,x,p):
  if self.step is None or self.slot is None or not 0<p<1:raise ValueError('Active nondegenerate L dropout only')
  occ=self.occ.get(index,0);self.occ[index]=occ+1;key=(index,occ);shape=tuple(x.shape)
  if self.slot==0:
   if key in self.cache:raise ValueError('Duplicate first-view mask')
   g=torch.Generator(device=x.device);g.manual_seed(self.seed+self.step*100000000+index*100000+occ*100)
   # Both modes pay for all four private draws, even when S uses reflection.
   r=[torch.randint(SIZE,shape,device=x.device,generator=g,dtype=torch.int32) for _ in range(4)]
   self.cache[key]=(shape,x.device,p,r);self.trace.append(key);self.random_elements+=4*x.numel()
  else:
   if self.cursor>=len(self.trace) or self.trace[self.cursor]!=key:raise ValueError('Dropout module order differs')
  oldshape,device,oldp,r=self.cache[key]
  if oldshape!=shape or device!=x.device or p!=oldp:raise ValueError('Dropout shape/device/probability differs between views')
  self.cursor+=1
  value=r[self.slot]
  if self.slot>=2:
   reflected=(SIZE-1)-r[self.slot-2]  # paid by both matched modes
   if self.mode=='ANTITHETIC':value=reflected
  mask=(value<round((1-p)*SIZE)).to(x.dtype)
  return x*mask/(1-p)
 def finish(self):
  if self.step is None or self.slot is not None or self.next_slot!=4 or not self.trace:raise ValueError('Incomplete four-view mask context')
  record=dict(step=self.step,mode=self.mode,active_calls_per_view=len(self.trace),random_tensors=4*len(self.trace),random_elements=self.random_elements,view_count=4,additional_NN_forwards=0)
  self.steps=self.step;self.records.append(record);self.abort();return record
 def abort(self):
  self.step=None;self.slot=None;self.next_slot=0;self.cache={};self.trace=[];self.occ={};self.cursor=0
 def summary(self):
  if self.step is not None or self.cache:raise ValueError('L masks leaked beyond loss construction')
  return dict(mode=self.mode,steps=self.steps,random_tensors=sum(r['random_tensors'] for r in self.records),random_elements=sum(r['random_elements'] for r in self.records),active_calls_per_view=sorted(set(r['active_calls_per_view'] for r in self.records)),global_RNG_consumption=0,extra_parameters=0,extra_NN_forwards=0)

def forward(self,x):
 engine=self._coupling_engine
 if self.training and self.p>0 and engine.step is not None:
  if self.inplace:raise ValueError('Inplace coupled dropout unsupported')
  return engine.apply(self._coupling_index,x,self.p)
 return F.dropout(x,self.p,self.training,self.inplace)

def install(model,mode,seed,prefix=''):
 engine=Engine(mode,seed);inventory=[]
 for name,module in model.named_modules():
  if isinstance(module,torch.nn.Dropout) and name.startswith(prefix):
   if module.inplace or not 0<=module.p<1 or hasattr(module,'_coupling_engine'):raise ValueError('Unsupported or already installed dropout')
   index=len(inventory);inventory.append(dict(name=name,index=index,p=module.p))
   module._coupling_engine=engine;module._coupling_index=index;module.forward=MethodType(forward,module)
 if not inventory:raise ValueError('No existing dropout found')
 model._coupling_engine=engine;model._coupling_inventory=inventory
 return engine
