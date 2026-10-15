"""SIG_TIME plus independently learned moment/covariance feature views."""
import math
from types import MethodType
import torch
from torch import nn
from experiments.cvs_reference_identity.model import ReferenceCVS
from experiments.cvs_e20_cross_style.style import _paths
from .descriptors import distribution_descriptor,moment_descriptor,path_signature
MODES={'SIG_TIME':('SIG_TIME',),'SIG_TIME_X3':('SIG_TIME','SIG_TIME','SIG_TIME'),'SIG_TIME_MV_COV':('SIG_TIME','MEAN_VAR','COV')}

def labeled_forward(model,x,y_tx=None,*args,**kwargs):
 if model.training and y_tx is not None and torch.is_tensor(y_tx) and bool((y_tx>=0).all()):
  # Count replay observations separately from native optimizer-step contexts.
  if getattr(model,"_feature_extra_l_forward",False):model._feature_extra_l_steps=getattr(model,"_feature_extra_l_steps",0)+1
  else:model._feature_l_steps+=1
 return model._signature_native_forward(x,y_tx,*args,**kwargs)

class DescriptorView(nn.Module):
 def __init__(self,channels,mode,seed,index):
  super().__init__();self.mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(int(seed)+80411+index*19001)
   self.local_projection=nn.Conv1d(channels,6,1,bias=False)
  rng=torch.Generator().manual_seed(2026101307)
  omega=torch.randn(14,6,generator=rng);omega=omega/omega.norm(dim=1,keepdim=True)*torch.tensor([.5,1.,2.,.5,1.,2.,.5,1.,2.,.5,1.,2.,.5,1.])[:,None]
  self.register_buffer('omega',omega)
  if mode=='COV':
   q,_=torch.linalg.qr(torch.randn(28,28,generator=rng));lift=q[:,:27].T*math.sqrt(28/27)
  else:lift=torch.eye(28)
  self.register_buffer('descriptor_lift',lift)
 def forward(self,h):
  u=torch.tanh(self.local_projection(h)).transpose(1,2);b,t,_=u.shape;windows=u.reshape(b,4,t//4,6).reshape(b*4,t//4,6)
  if self.mode=='SIG_TIME':v=path_signature(windows,True)
  elif self.mode=='MEAN_VAR':v=distribution_descriptor(windows,'MEAN_VAR',self.omega)
  elif self.mode=='COV':v=moment_descriptor(windows)
  else:raise ValueError('Unregistered descriptor view')
  return (v@self.descriptor_lift).reshape(b,112)

class CompositePool(nn.Module):
 def __init__(self,channels,arm,seed):
  super().__init__()
  if arm not in MODES:raise ValueError('Unregistered feature combination')
  self.arm=arm;self.channels=channels;self.modes=MODES[arm]
  self.views=nn.ModuleList([DescriptorView(channels,mode,seed,i) for i,mode in enumerate(self.modes)])
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(int(seed)+80411)
   # Preserve the historical first projection RNG draw before output init.
   nn.Conv1d(channels,6,1,bias=False)
   self.output_projection=nn.Linear(112*len(self.views),160,bias=False)
  self.output_norm=nn.LayerNorm(160);self.gain_raw=nn.Parameter(torch.full((160,),math.atanh(.05/.2)))
 def forward(self,h):
  if h.ndim!=3 or h.shape[1]!=self.channels or h.shape[2]<8 or h.shape[2]%4:raise ValueError('Native t2 four equal windows required')
  descriptor=torch.cat([v(h) for v in self.views],1)
  return .2*self.gain_raw.tanh()*self.output_norm(self.output_projection(descriptor))

class CompositeReference(ReferenceCVS):
 def features(self,x):
  local={}
  position=getattr(self,'sequence_read_position','t2')
  if position not in ('t2','t2.pool'):raise ValueError('Unregistered temporal descriptor read position')
  target=self.core.id_backbone.t2.pool if position=='t2.pool' else self.core.id_backbone.t2
  handle=target.register_forward_hook(lambda module,args,out:local.update(h=out))
  try:base=super().features(x)
  finally:handle.remove()
  if set(local)!={'h'}:raise ValueError('Actual t2 feature path bypassed')
  return base+self.sequence_pool(local['h'])

def install(model,seed,arm):
 modules,paths=_paths(model);core=modules[paths['time_down'].rsplit('.',1)[0]]
 refs=[m for m in model.modules() if type(m) is ReferenceCVS]
 if len(refs)!=1:raise ValueError('Exactly one default reference extractor required')
 ref=refs[0];ref.sequence_pool=CompositePool(core.t2.pw.out_channels,arm,seed);ref.__class__=CompositeReference
 model._feature_l_steps=0;model._feature_mask_counts={n:0 for n in ('time','frequency','PA','reference')};model._signature_native_forward=MethodType(type(model).forward,model);model.forward=MethodType(labeled_forward,model)
 extra=sum(p.numel() for p in ref.sequence_pool.parameters())
 model._feature_learning_report=dict(arm=arm,modes=list(MODES[arm]),layer='t2',windows=4,local_dimension_each=6,independent_local_projections=True,descriptor_width_each=28,output_embedding=160,extra_parameters=extra,classifier_changed=False,default_head_only=True,separate_head_training=False,gain_initial=.05,gain_bound=.2,per_packet=True,teacher_and_U_features_identical_policy=True,intrinsic_descriptor_rank_matched=False)
 return model

class ExportedComposite(nn.Module):
 """Copy actual composite f_theta and discard class W alone."""
 def __init__(self,reference):
  super().__init__()
  from copy import deepcopy
  from experiments.cvs_source_identity_readout.input_policy import NoExplicitCFO
  from experiments.cvs_residual_identity.model import ResidualPhysicalHead
  if type(reference) is not CompositeReference or type(reference.response) is not NoExplicitCFO or type(reference.core.id_backbone.cls_head) is not ResidualPhysicalHead:raise ValueError('Recognized composite NoExplicitCFO default-head boundary required')
  if reference.core.id_backbone.cls_head.weight.shape!=(6,160):raise ValueError('Class W boundary differs')
  self.encoder=deepcopy(reference);del self.encoder.core.id_backbone.cls_head.weight
 def forward_features(self,x):
  if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('IQ2x256 required')
  with torch.autocast(device_type=x.device.type,enabled=False):z=self.encoder.features(x.float())
  if z.shape!=(len(x),160):raise ValueError('Actual composite joint160 differs')
  return z
 def forward(self,x):return self.forward_features(x)
def export(model):return ExportedComposite(model.id_backbone.encoder)
