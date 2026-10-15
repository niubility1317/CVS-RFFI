from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d,model as api
from experiments.cvs_feature_sequence50 import source as shared
class Observer:
 def initialize(self,m,c,out):self.m=m;self.c=c;self.out=Path(out)
 def native_scope(self,native):
  from contextlib import nullcontext
  return nullcontext()
 def epoch(self,m,ctx,rows,out,device):
  steps=len(rows)*222;extra=0 if self.c['mode']=='ORIGINAL' else steps*128
  if m._feature_l_steps!=steps or m._legacy_calls!=steps or m._legacy_extra_packets!=extra or m._legacy_actual_L_input is not None:raise ValueError('Actual old SIG_TIME/RDrop call or augmented input lifetime differs')
  d.write(self.out/'source_training_mechanism.json',dict(epoch=len(rows),optimizer_steps=steps,L_context_steps=steps,objective_calls=steps,extra_L_identity_forwards=extra,positive_KL_calls=m._legacy_positive_KL_calls,actual_augmented_input_reused=self.c['mode']!='ORIGINAL',augmentation_redraws=0,label_smoothing=.01,use_aug=True,sinc_coordinates='RAW_HZ',original_U_sampler_unchanged=True,default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False,config=self.c))
def train(c):
 view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT')},UPDATES=c['epochs']*222)
 observer=Observer();shared.train(c,design=view,model_api=api,observer=observer)
 from .provenance import source_proof,validate_checkpoint
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c)
 return d.read(p/'completion.json')
