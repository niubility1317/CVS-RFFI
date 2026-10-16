"""Fresh20 source-RX qualification, unchanged default head and hidden U."""
from contextlib import contextmanager
import copy,gzip,json
import torch
from . import design as d,model as api
from .optimizer import attach
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
from experiments.cvs_sigtime_locked20.source import Observer as FoldObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer:
 data=FoldObserver.data;source_contract=FoldObserver.source_contract;evaluate=FoldObserver.evaluate
 def __init__(self,c):self.c=c;self.optimizer=None
 def initialize(self,m,c,out):
  self.model=m;self.reader=copy.deepcopy(m).eval();self.out=out;(out/'source_readouts').mkdir()
 @contextmanager
 def native_scope(self,native):
  with native_scope(self,'CENTER_UNITS'):yield
 def optimizer_created(self,opt):
  self.policy=attach(opt,self.model,self.c['arm']);d.write(self.out/'band_gain_policy.json',self.policy)
 def epoch(self,m,ctx,rows,out,device):
  steps=len(rows)*222
  if m._feature_l_steps!=steps or m._legacy_calls!=steps or m._legacy_extra_packets!=steps*128 or m._legacy_actual_L_input is not None or self.optimizer._sinc_unit_steps!=steps or self.optimizer._band_gain_steps!=steps or self.optimizer._sinc_physical_clip_calls!=steps:raise ValueError('Actual RDrop/gain/physicalclip budgets differ')
  d.write(out/'gain_progress.json',dict(epoch=len(rows),actual_steps=steps,RDrop_calls=m._legacy_calls,extra_L_identity_packets=m._legacy_extra_packets,mode=self.c['arm'],default_head_joint=True,head_fit_updates=0,U_truth_read=False,target_access=False,config=self.c))
  if len(rows) in d.WITNESS_EPOCHS:
   witness=[v for v in self.optimizer._band_gain_raw if v['epoch']==len(rows)]
   if len(witness)!=13:raise ValueError('Missing actual gain step witnesses')
   torch.save(witness,out/f'gain_update_witness_e{len(rows)}.pt')
 def finish(self):
  done=d.read(self.out/'completion.json');payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or payload['args'].get('source_fold_held_rx')!=1 or payload['checkpoint_selection']!='final_only' or payload['a1_ema_successful_updates']!=4440 or payload.get('ema_model') is None or len(self.optimizer._band_gain_records)!=4440 or len(self.optimizer._band_gain_raw)!=52:raise ValueError('Fixed complete own source/EMA/optimizer trajectory required')
  if any(not torch.equal(v,self.model.state_dict()[n].detach().cpu()) for n,v in payload['model'].items()):raise ValueError('Frozen default model changed')
  initial=torch.load(self.out/'initial_smoke.pt',map_location='cpu',weights_only=False)
  if not initial['scratch_only']:raise ValueError('No ancestor permitted')
  from experiments.cvs_rdrop_sinc_units.feature_check import measure
  feature=measure(d.base_model_config(),initial['model'],payload['model'],payload['ema_model']);feature['row_id']=self.c['row_id'];feature['actual_training_epochs']=20
  d.write(self.out/'feature_witness.json',feature)
  held=self.evaluate(payload['model']);d.write(self.out/'source_readouts/FINAL.json',held);compact={k:v for k,v in held.items() if k not in ('physical_ids','labels','predictions')}
  with gzip.open(self.out/'gain_steps.jsonl.gz','xt',encoding='utf8') as f:
   for v in self.optimizer._band_gain_records:f.write(json.dumps(v,allow_nan=False)+'\n')
  with gzip.open(self.out/'unit_steps.jsonl.gz','xt',encoding='utf8') as f:
   for v in self.optimizer._sinc_unit_lr_records:f.write(json.dumps(v,allow_nan=False)+'\n')
  result=dict(status='SOURCE_GAIN_COMPLETE',arm=self.c['arm'],model_seed=d.SEED,epochs=20,actual_steps=4440,physical_clip_calls=self.optimizer._sinc_physical_clip_calls,final_L_CE=done['final_training_L_CE'],held_default_head=compact,feature_witness=feature,policy=self.policy,head_fit_updates=0,optimizer_updates=4440,extra_optimizer_updates=0,U_truth_read=False,target_access=False,extra_L_identity_packets=568320,hidden_U_observations=396368,held_L_forward_packets=1260,unused_anchor_L_packets=144,scratch_initialization=True,checkpoint_sources=[])
  d.write(self.out/'gain_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
