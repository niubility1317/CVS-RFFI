from contextlib import contextmanager
from pathlib import Path
import copy,torch
from torch.nn import functional as F
from . import design as d,model as api
from .fold import prepare
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng
class Observer:
 def __init__(self,c):self.c=c
 def data(self,native,ctx,out):
  unused,self.held=prepare(native,ctx)
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  self.roles=role_ids_from_native(ctx)
  d.write(Path(out)/'source_fold.json',dict(held_RX=d.HELD,fit_RX=list(d.FIT),effective_counts={k:len(v) for k,v in self.roles.items()},
   all_training_calibration_probe_loaders_held_RX_excluded=True,held_L_packets=1260,additional_L_IQ_reads=1404,
   target_access=False,U_truth_read=False,discarded_unused_anchor_packets=len(unused['y'])))
 def source_contract(self,expected,ctx):
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  if role_ids_from_native(ctx)!=self.roles or {k:len(v) for k,v in self.roles.items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Four RX roles differ')
  if any(not set(ids)<=set(expected['role_ids'][key]) for key,ids in self.roles.items()):raise ValueError('Effective roles escape original source contract')
  return dict(expected,role_ids=self.roles,source_receivers=list(d.FIT),held_RX=d.HELD,carrier_contract_ref=d.SOURCE,
   native_role_comparison='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD',checkpoint_effective_training_contract='Four RX only;never full-five-RX initialization')
 def initialize(self,model,c,out):
  if c!=self.c:raise ValueError('One fixed source trajectory required')
  self.model=model;self.reader=copy.deepcopy(model).eval();self.out=Path(out);(self.out/'source_readouts').mkdir()
 @contextmanager
 def native_scope(self,native):yield
 def epoch(self,m,ctx,rows,out,device):
  steps=len(rows)*222;a=m._center_audit
  if m._feature_l_steps!=steps or m._legacy_calls!=steps or m._center_calls!=steps or m._legacy_extra_packets!=steps*128 or m._legacy_actual_L_input is not None or a['calls']!=steps or a['packets']!=steps*128:raise ValueError('Actual RDrop evidence forward budget differs')
  d.write(self.out/'rdrop_center_progress.json',dict(epoch=len(rows),actual_steps=steps,extra_L_identity_packets=m._legacy_extra_packets,evidence_audit=a.copy(),config=self.c,target_access=False,head_fit_updates=0))
 def evaluate(self,state):
  self.reader.load_state_dict(state,strict=True);self.reader.eval();device=next(self.reader.parameters()).device;predictions=[];labels=[];CE=[];margins=[];probabilities=[]
  with _preserve_rng(),torch.no_grad(),fresh_parameter_caches(self.reader):
   for start in range(0,1260,128):
    x=self.held['x'][start:start+128].to(device);y=self.held['y'][start:start+128].to(device);logits=self.reader.forward_identity_only(x)['tx_logits'].double()
    if logits.shape!=(len(y),6) or not bool(torch.isfinite(logits).all()):raise ValueError('Finite native held source logits required')
    other=logits.clone();other.scatter_(1,y[:,None],float('-inf'));margin=logits.gather(1,y[:,None]).flatten()-other.max(1).values
    predictions+=logits.argmax(1).cpu().tolist();labels+=y.cpu().tolist();CE+=F.cross_entropy(logits,y,reduction='none').cpu().tolist();margins+=margin.cpu().tolist();probabilities+=logits.softmax(1).gather(1,y[:,None]).flatten().cpu().tolist()
  errors=sum(a!=b for a,b in zip(labels,predictions))
  return dict(count=1260,mean_CE=sum(CE)/1260,errors=errors,accuracy=1-errors/1260,mean_margin=sum(margins)/1260,physical_ids=self.held['ids'],labels=labels,predictions=predictions,per_packet_CE=CE,per_packet_true_margin=margins,per_packet_true_probability=probabilities,head_fit_updates=0)
 def finish(self):
  done=d.read(self.out/'completion.json')
  from .provenance import source_proof,validate_checkpoint
  source_proof(self.c);payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(payload,self.c)
  if any(not torch.equal(v,self.model.state_dict()[k].detach().cpu()) for k,v in payload['model'].items()):raise ValueError('Frozen live weights/checkpoint differ')
  held=self.evaluate(payload['model']);d.write(self.out/'source_readouts/FINAL.json',held)
  compact={k:v for k,v in held.items() if k not in ('physical_ids','labels','predictions','per_packet_CE','per_packet_true_margin','per_packet_true_probability')};p=d.read(self.out/'rdrop_center_progress.json')
  result=dict(status='SOURCE_RDROP_CENTER_COMPLETE',arm=self.c['arm'],epoch=20,actual_steps=4440,source_health=done['final_training_L_CE']<=.5,final_training_L_CE=done['final_training_L_CE'],exports={'FINAL':compact},evidence_audit=p['evidence_audit'],extra_L_identity_packets=568320,held_L_forward_packets=1260,additional_L_IQ_reads=1404,unused_anchor_L_packets=144,default_head_joint_training=True,head_fit_updates=0,U_truth_read=False,target_access=False)
  d.write(self.out/'rdrop_center_result.json',result);print('SOURCE_RDROP_CENTER_COMPLETE '+self.c['row_id'],flush=True);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
