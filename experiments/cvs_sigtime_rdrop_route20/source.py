from contextlib import contextmanager
from pathlib import Path
import copy,torch
from . import design as d,model as api
from experiments.cvs_sigtime_locked20.source import Observer as OldObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer:
 data=OldObserver.data;source_contract=OldObserver.source_contract;evaluate=OldObserver.evaluate
 def __init__(self,c):self.c=c
 def initialize(self,model,c,out):
  if c!=self.c:raise ValueError('One fixed RDrop trajectory required')
  self.model=model;self.reader=copy.deepcopy(model).eval();self.out=Path(out);(self.out/'source_readouts').mkdir()
 @contextmanager
 def native_scope(self,native):yield
 def epoch(self,m,ctx,rows,out,device):
  steps=len(rows)*222;a=m._rdrop_route_audit
  if m._feature_l_steps!=steps or m._legacy_calls!=steps or m._rdrop_route_calls!=steps or m._legacy_extra_packets!=steps*128 or m._legacy_actual_L_input is not None or a['calls']!=steps:raise ValueError('Actual RDrop forward/backward budget differs')
  progress=dict(epoch=len(rows),actual_steps=steps,extra_L_identity_packets=m._legacy_extra_packets,backward_audit=a.copy(),removed_common_energy_fraction=a['common_energy']/max(a['raw_energy'],1e-300),config=self.c,target_access=False,head_fit_updates=0)
  d.write(self.out/'rdrop_route_progress.json',progress)
 def finish(self):
  done=d.read(self.out/'completion.json')
  if done['optimizer_steps']!=4440 or done['checkpoint_sources'] or done['initialization']!='scratch':raise ValueError('Own complete scratch20 trajectory required')
  from .provenance import source_proof,validate_checkpoint
  source_proof(self.c);payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(payload,self.c)
  if any(not torch.equal(v,self.model.state_dict()[k].detach().cpu()) for k,v in payload['model'].items()):raise ValueError('Frozen live model/checkpoint differs')
  readout=self.evaluate(payload['model']);d.write(self.out/'source_readouts/FINAL.json',readout)
  compact={k:v for k,v in readout.items() if k not in ('physical_ids','labels','predictions')};progress=d.read(self.out/'rdrop_route_progress.json')
  result=dict(status='SOURCE_RDROP_ROUTE_COMPLETE',arm=self.c['arm'],epoch=20,actual_steps=4440,source_health=done['final_training_L_CE']<=.5,final_training_L_CE=done['final_training_L_CE'],exports={'FINAL':compact},
   removed_common_energy_fraction=progress['removed_common_energy_fraction'],backward_audit=progress['backward_audit'],extra_L_identity_packets=568320,held_L_forward_packets=1260,additional_L_IQ_reads=1404,unused_anchor_L_packets=144,
   default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False)
  d.write(self.out/'rdrop_route_result.json',result);print('SOURCE_RDROP_ROUTE_COMPLETE '+self.c['row_id'],flush=True);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
