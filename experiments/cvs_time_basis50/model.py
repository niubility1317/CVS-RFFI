"""One k5 temporal basis plus rank-neutral pointwise overparameterization."""
from copy import deepcopy
import torch
from torch import nn
from . import design as d
from experiments.cvs_feature_multiscale50.model import source_domain_count,classifier
from experiments.cvs_feature_restitution.features import install as install_counter,layout
from experiments.cvs_phase1_stack.runtime import installed
class SingleBasisBlock(nn.Module):
 def __init__(self,old,hidden):
  super().__init__();ci,co=old.dw.in_channels,old.pw.out_channels
  self.dw=nn.Conv1d(ci,ci,5,padding=2,groups=ci,bias=False)
  self.pw_expand=nn.Conv1d(ci,hidden,1,bias=False);self.pw=nn.Conv1d(hidden,co,1,bias=False);self.skip=nn.Conv1d(ci,co,1,bias=False)
  for key in ('norm','act','pool','drop'):setattr(self,key,deepcopy(getattr(old,key)))
 def preGN(self,x):return self.pw(self.pw_expand(self.dw(x)))+self.skip(x)
 def forward(self,x):return self.drop(self.pool(self.act(self.norm(self.preGN(x)))))
def equivalent_kernel(block):
 p=block.pw.weight[:,:,0].double()@block.pw_expand.weight[:,:,0].double()
 k=p[:,:,None]*block.dw.weight[:,0].double()[None,:,:];k[:,:,2]+=block.skip.weight[:,:,0].double();return k
def install(model,c):
 d.validate(c);model=install_counter(model,c['model_seed'],norm='none',competition=False);core,_,_=layout(model)
 if c['arm']!='BASE_REF':
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(c['model_seed']+48213)
   for layer,h in zip(('t1','t2','t3'),d.HIDDEN):setattr(core,layer,SingleBasisBlock(getattr(core,layer),h))
 model._feature_learning_report=dict(c['feature_extractor'],arm=c['arm'],default_head_only=True,separate_head_training=False,classifier_type=type(classifier(model)).__name__,classifier_weight_shape=list(classifier(model).weight.shape),classifier_scale=float(classifier(model).scale))
 return model
def build(c,device):
 d.validate(c);a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256;torch.manual_seed(c['model_seed'])
 with installed(c) as n:model=n.build_baseline_model(a,torch.device(device))
 from experiments.cvs_source_identity_readout.input_policy import apply
 model._identity_input_policy=apply(model);return install(model,c).to(device)
def validate_report(report,c):
 if report!=dict(c['feature_extractor'],arm=c['arm'],default_head_only=True,separate_head_training=False,classifier_type='ResidualPhysicalHead',classifier_weight_shape=[6,160],classifier_scale=30.):raise ValueError('Actual single-basis/default-head report differs')
