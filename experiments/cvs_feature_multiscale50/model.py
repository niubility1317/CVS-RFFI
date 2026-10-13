from copy import deepcopy
import torch
from torch import nn
from . import design as d
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_stage2_decouple.common import parameter_groups
from experiments.cvs_e20_cross_style.style import _paths

class MultiscaleFeatureBlock(nn.Module):
 def __init__(self,old,kind):
  super().__init__();self.kind=kind
  if kind=='time':
   ci,co=old.dw.in_channels,old.pw.out_channels;settings=((3,1),(5,1),(5,2))
   self.branches=nn.ModuleList([nn.Conv1d(ci,ci,k,padding=(k-1)//2*rate,dilation=rate,groups=ci,bias=False) for k,rate in settings]);fuse_in=3*ci
  elif kind=='PA':
   ci,co=old.conv.in_channels,old.conv.out_channels;rate=old.conv.dilation[0];settings=tuple((k,rate) for k in (3,5,7))
   self.branches=nn.ModuleList([nn.Conv1d(ci,co,k,padding=(k-1)//2*rate,dilation=rate,bias=False) for k,rate in settings]);fuse_in=3*co
  else:raise ValueError('Unknown feature block')
  self.fuse=nn.Conv1d(fuse_in,co,1,bias=False);self.skip=nn.Conv1d(ci,co,1,bias=False)
  self.norm=deepcopy(old.norm);self.act=deepcopy(old.act);self.pool=deepcopy(old.pool);self.drop=deepcopy(old.drop)
  self._settings=settings
 def forward(self,x):
  y=self.fuse(torch.cat([branch(x) for branch in self.branches],1))+self.skip(x)
  return self.drop(self.pool(self.act(self.norm(y))))

def install(model,c):
 settings=c['feature_extractor'];modules,paths=_paths(model);b=modules[paths['time_down'].rsplit('.',1)[0]];changed=[]
 for kind,layers in (('time',d.TIME_LAYERS),('PA',d.PA_LAYERS)):
  if settings[kind]:
   for layer in layers:setattr(b,layer,MultiscaleFeatureBlock(getattr(b,layer),kind));changed.append(layer)
 model._feature_extractor=dict(changed_layers=changed,classifier_type=type(b.cls_head).__name__,classifier_weight_shape=list(b.cls_head.weight.shape),classifier_scale=float(b.cls_head.scale),output_embedding=160,structural_classifier_changes=False,multiscale={n:[list(v) for v in getattr(b,n)._settings] for n in changed})
 return model

def classifier(model):
 modules,paths=_paths(model);return modules[paths['time_down'].rsplit('.',1)[0]].cls_head

def build(c,device):
 d.validate(c);a=d.make_args(c,str(device));a.num_domains=5;a.input_len=256
 torch.manual_seed(c['model_seed'])
 with installed(c['legacy_reference_config']) as native:model=native.build_baseline_model(a,torch.device(device))
 from experiments.cvs_source_identity_readout.input_policy import apply
 model._identity_input_policy=apply(model)
 return install(model,c).to(device)
def groups(model,c):
 pg,names=parameter_groups(model)
 for group in pg:group['lr']=d.make_args(c,'cpu').lr
 return [group for group in pg if group['params']],{key:value for key,value in names.items() if value}
