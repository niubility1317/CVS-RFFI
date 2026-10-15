from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d,model as api
from experiments.cvs_sigtime_rdrop_legacy.source import Observer as NativeObserver
from experiments.cvs_sigtime_rdrop_masks20.mechanism import validate_audit
from experiments.cvs_feature_sequence50 import source as shared
class Observer(NativeObserver):
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222;state=m._pair_state
  if state.stage!='idle' or state.cache:raise ValueError('Paired masks leaked outside actual L')
  audit=state.snapshot();validate_audit(audit,'ANTITHETIC',steps)
  d.write(self.out/'rdrop_masks_progress.json',dict(epoch=len(rows),actual_steps=steps,extra_L_identity_packets=m._legacy_extra_packets,mask_audit=audit,cache_empty=True,config=self.c,target_access=False,head_fit_updates=0))
def train(c):
 view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT')},UPDATES=c['epochs']*222)
 shared.train(c,design=view,model_api=api,observer=Observer())
 from .provenance import source_proof,validate_checkpoint
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c)
 return d.read(p/'completion.json')
