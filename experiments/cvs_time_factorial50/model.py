"""Only temporal blocks change; all four conditions share initialization draws."""
from copy import deepcopy
import torch
from torch import nn
from . import design as d
from experiments.cvs_feature_multiscale50.model import source_domain_count,classifier
from experiments.cvs_feature_restitution.features import install as install_counter,layout
from experiments.cvs_phase1_stack.runtime import installed
class TemporalBlock(nn.Module):
 def __init__(self,old,long,nonlinear):
  super().__init__();ci,co=old.dw.in_channels,old.pw.out_channels
  self.settings=((3,1),(5,1),(5,2 if long else 1));self.nonlinear=nonlinear
  self.branches=nn.ModuleList([nn.Conv1d(ci,ci,k,padding=(k-1)//2*r,dilation=r,groups=ci,bias=False) for k,r in self.settings])
  self.fuse=nn.Conv1d(3*ci,co,1,bias=False);self.skip=nn.Conv1d(ci,co,1,bias=False)
  self.branch_act=nn.SiLU() if nonlinear else nn.Identity()
  for n in ('norm','act','pool','drop'):setattr(self,n,deepcopy(getattr(old,n)))
 def mapped(self,x):
  outputs=[self.branch_act(b(x)) for b in self.branches];ci=x.shape[1]
  return [torch.nn.functional.conv1d(v,self.fuse.weight[:,j*ci:(j+1)*ci]) for j,v in enumerate(outputs)]
 def forward(self,x):
  y=self.fuse(torch.cat([self.branch_act(b(x)) for b in self.branches],1))+self.skip(x)
  return self.drop(self.pool(self.act(self.norm(y))))
def install(model,c):
 d.validate(c);model=install_counter(model,c['model_seed'],norm='none',competition=False);core,_,_=layout(model)
 if c['arm']!='BASE_REF':
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(c['model_seed']+48213)
   for layer in ('t1','t2','t3'):setattr(core,layer,TemporalBlock(getattr(core,layer),c['arm'].startswith('LONG'),c['arm'].endswith('SILU')))
 model._feature_learning_report=dict(c['feature_extractor'],arm=c['arm'],default_head_only=True,separate_head_training=False,classifier_type=type(classifier(model)).__name__,classifier_weight_shape=list(classifier(model).weight.shape),classifier_scale=float(classifier(model).scale))
 return model
def build(c,device):
 d.validate(c);a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256;torch.manual_seed(c['model_seed'])
 with installed(c) as n:model=n.build_baseline_model(a,torch.device(device))
 from experiments.cvs_source_identity_readout.input_policy import apply
 model._identity_input_policy=apply(model);return install(model,c).to(device)
def validate_report(report,c):
 if report!=dict(c['feature_extractor'],arm=c['arm'],default_head_only=True,separate_head_training=False,classifier_type='ResidualPhysicalHead',classifier_weight_shape=[6,160],classifier_scale=30.):raise ValueError('Actual temporal/default head path differs')
