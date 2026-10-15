"""Reuse native RX1 filtering and released default-head-only evaluation."""
from pathlib import Path
import torch
from . import design as d,model as api
from experiments.cvs_rx_u_increment20.fold import prepare
from experiments.cvs_sigtime_rdrop_center20.source import Observer as BaseObserver
from experiments.cvs_feature_sequence50.source import train as shared_train

class Observer:
 initialize=BaseObserver.initialize
 evaluate=BaseObserver.evaluate
 native_scope=BaseObserver.native_scope
 def __init__(self,c):self.c=c
 def data(self,native,ctx,out):
  unused,self.held=prepare(native,ctx)
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  self.roles=role_ids_from_native(ctx)
  d.write(Path(out)/'source_fold.json',dict(held_RX=d.HELD,fit_RX=list(d.FIT),effective_counts={k:len(v) for k,v in self.roles.items()},all_training_calibration_probe_loaders_held_RX_excluded=True,held_L_packets=1260,additional_L_IQ_reads=1404,target_access=False,U_truth_read=False,discarded_unused_anchor_packets=len(unused['y'])))
 def source_contract(self,expected,ctx):
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  if role_ids_from_native(ctx)!=self.roles or {k:len(v) for k,v in self.roles.items()}!={'L_s':5040,'U_s':45360,'V':21600} or any(not set(ids)<=set(expected['role_ids'][k]) for k,ids in self.roles.items()):raise ValueError('Effective source roles escape original contract')
  return dict(expected,role_ids=self.roles,source_receivers=list(d.FIT),held_RX=d.HELD,carrier_contract_ref=d.SOURCE,native_role_comparison='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD',checkpoint_effective_training_contract='FourRX only,no inherited trained checkpoint')
 def epoch(self,m,ctx,rows,out,device):
  steps=len(rows)*222;a=m._gain_audit
  if m._feature_l_steps!=steps or m._legacy_calls!=steps or m._gain_calls!=steps or m._legacy_extra_packets!=steps*128 or m._legacy_actual_L_input is not None or a['calls']!=steps or a['packets']!=steps*128:raise ValueError('Actual paired gain budget differs')
  d.write(self.out/'rdrop_gain_progress.json',dict(epoch=len(rows),actual_steps=steps,extra_L_identity_packets=m._legacy_extra_packets,evidence_audit=a.copy(),config=self.c,target_access=False,head_fit_updates=0))
 def finish(self):
  from .provenance import source_proof,validate_checkpoint
  done=d.read(self.out/'completion.json');source_proof(self.c)
  payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(payload,self.c)
  if any(not torch.equal(v,self.model.state_dict()[k].detach().cpu()) for k,v in payload['model'].items()):raise ValueError('Frozen live/default checkpoint differs')
  held=self.evaluate(payload['model']);d.write(self.out/'source_readouts/FINAL.json',held)
  private=('physical_ids','labels','predictions','per_packet_CE','per_packet_true_margin','per_packet_true_probability')
  compact={k:v for k,v in held.items() if k not in private};progress=d.read(self.out/'rdrop_gain_progress.json')
  result=dict(status='SOURCE_RDROP_GAIN_COMPLETE',arm=self.c['arm'],epoch=20,actual_steps=4440,source_health=done['final_training_L_CE']<=.5,final_training_L_CE=done['final_training_L_CE'],exports={'FINAL':compact},evidence_audit=progress['evidence_audit'],extra_L_identity_packets=568320,held_L_forward_packets=1260,additional_L_IQ_reads=1404,unused_anchor_L_packets=144,default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False)
  d.write(self.out/'rdrop_gain_result.json',result);print('SOURCE_RDROP_GAIN_COMPLETE '+self.c['row_id'],flush=True);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
