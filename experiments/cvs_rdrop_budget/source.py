from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path
from . import design as d,model
from experiments.cvs_feature_sequence50.source import train as shared_train,source_validation
class Observer:
 def __init__(self,c):self.c=c
 def initialize(self,m,c,out):self.model=m;self.out=Path(out)
 @contextmanager
 def native_scope(self,native):yield
 def epoch(self,m,ctx,rows,out,device):
  steps=len(rows)*222
  if m._feature_l_steps!=steps or m._rdrop_calls!=steps or m._rdrop_extra_packets!=steps*128:raise ValueError('Actual double-L-forward/update budget differs')
  d.write(self.out/'source_training_mechanism.json',dict(epoch=len(rows),optimizer_steps=steps,L_context_steps=m._feature_l_steps,paired_calls=m._rdrop_calls,extra_L_identity_forwards=m._rdrop_extra_packets,alpha=m._rdrop_alpha,full_five_RX=True,original_U_sampler_unchanged=True,default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False,config=self.c))
def train(c):
 d.validate(c);view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT')},UPDATES=c['epochs']*222,source_validation=source_validation)
 shared_train(c,design=view,model_api=model,observer=Observer(c))
