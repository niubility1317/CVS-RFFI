"""Same source trainer/optimizer/teacher;L-only native carrier organization."""
import gzip,json,sys
from contextlib import contextmanager
import torch
from . import design as d,model as api
from .carrier import install as install_carrier
from experiments.cvs_clock_factorial30.source import Observer as OriginalObserver
from experiments.cvs_view_moment30.source import _train
class Observer(OriginalObserver):
 @contextmanager
 def native_scope(self,native):
  with super().native_scope(native):
   original=native.make_augmentor;self.carrier=None
   def make(cfg):
    if self.carrier is not None:raise ValueError('Only one L augmentor construction expected;U must remain unchanged')
    obj=original(cfg);self.carrier=install_carrier(obj,self.c['carrier_mode'],d.CARRIER_SEED);return obj
   native.make_augmentor=make
   try:yield
   finally:native.make_augmentor=original
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);n=len(rows)*222
  if self.carrier is None or self.carrier.phase_calls!=n or self.carrier.cfo_calls!=n or self.carrier.pending is not None:raise ValueError('Exactly one complete L-only carrier call per update required')
  d.write(self.out/'carrier_progress.json',dict(epoch=len(rows),cumulative_epoch=20+len(rows),**self.carrier.summary(),U_rule_unchanged=True,target_access=False))
 def finish(self):
  result=super().finish()
  if self.carrier is None or len(self.carrier.records)!=2220:raise ValueError('All2220L augmentations required')
  with gzip.open(self.out/'carrier_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
   for r in self.carrier.records:f.write(json.dumps(r,allow_nan=False)+'\n')
  result.update(carrier_organization=self.carrier.summary(),one_shared_NoAs20_parent=True,not_new_IQ_Ag=True);d.write(self.out/'training_result.json',result);return result
def train(c):
 d.validate(c);p=d.parent_proof();parent=d.validate_parent_checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c);observer=Observer(c,parent);_train(c,design=d,model_api=api,observer=observer,parent=parent);return observer.finish()
