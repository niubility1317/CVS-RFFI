from contextlib import contextmanager
from pathlib import Path
import copy,torch
from . import design as d,model as api
from experiments.cvs_sigtime_locked20.source import Observer as OldObserver
from experiments.cvs_feature_sequence50.source import train as shared_train

class Observer:
 data=OldObserver.data
 source_contract=OldObserver.source_contract
 evaluate=OldObserver.evaluate
 def __init__(self,c):self.c=c
 def initialize(self,model,c,out):
  if c!=self.c:raise ValueError('One fixed source trajectory required')
  self.model=model;self.reader=copy.deepcopy(model).eval();self.out=Path(out)
  (self.out/'source_readouts').mkdir()
 @contextmanager
 def native_scope(self,native):yield
 def epoch(self,model,ctx,rows,out,device):
  steps=len(rows)*222
  if model._feature_l_steps!=steps or model._rdrop_calls!=steps or model._rdrop_extra_packets!=steps*128:
   raise ValueError('Matched two-L-forward update budget differs')
  d.write(self.out/'rdrop_progress.json',dict(status='RUNNING',epoch=len(rows),actual_steps=steps,extra_L_identity_packets=model._rdrop_extra_packets,arm=self.c['arm'],target_access=False,head_fit_updates=0))
 def finish(self):
  done=d.read(self.out/'completion.json')
  if done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or self.model._rdrop_calls!=4440 or self.model._rdrop_extra_packets!=568320:
   raise ValueError('Fixed scratch20 two-forward training required')
  payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if any(not torch.equal(v,self.model.state_dict()[k].detach().cpu()) for k,v in payload['model'].items()):raise ValueError('Live/final checkpoint differ')
  value=self.evaluate(payload['model']);d.write(self.out/'source_readouts/FINAL.json',value)
  compact={k:v for k,v in value.items() if k not in ('physical_ids','labels','predictions')}
  result=dict(status='SOURCE_RDROP_TRAINED',arm=self.c['arm'],epoch=20,actual_steps=4440,source_health=done['final_training_L_CE']<=.5,
    final_training_L_CE=done['final_training_L_CE'],exports=dict(FINAL=compact),held_L_forward_packets=1260,
    extra_L_identity_packets=568320,paired_objective_calls=4440,smoke_forward_packets=2,additional_L_IQ_reads=1404,
    target_access=False,U_truth_read=False,head_fit_updates=0,default_head_joint_training=True,checkpoint_sources=[])
  d.write(self.out/'rdrop_result.json',result);print('RDROP_RESULT '+str(result),flush=True);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
