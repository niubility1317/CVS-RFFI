"""Compact full-budget telemetry, retaining original P43 U/EMA and losses."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import gzip,json,sys
import torch
from . import design as d
from .source import parent_payload
from .protocol import check_checkpoint
from .augmentation import ordinary_factory,strong_passthrough,patch_active
from experiments.cvs_lraw_tail30 import observer as old
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
from experiments.cvs_fullsource_concat20 import supcon

class Observer(old.Observer):
 def __init__(self,c):
  self.c=c;self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c);self.ordinary_records=[];self.U_records=[]
 def initialize(self,m,c,out):
  old.Observer.initialize(self,m,c,out);m._oc_ordinary_records=self.ordinary_records
  # Inherited budget checker consumes only these list lengths, not LRaw semantics.
  m._lraw_records=m._oc_records
 @contextmanager
 def native_scope(self,native):
  orig=native.deepcopy;aug=native.configure_augmentor_for_epoch;schedule=supcon.schedule;terminal=native._resolve_phase1_terminal_status;factory=native.make_augmentor;strong=native._strong_augment
  self.ordinary_factory=ordinary_factory(factory,self.ordinary_records);native.make_augmentor=self.ordinary_factory
  native._strong_augment=strong_passthrough(strong,self.U_records,lambda:self.model._feature_l_steps)
  def copied(v,*args,**kw):
   result=orig(v,*args,**kw)
   if v is getattr(self,'model',None):
    result.load_state_dict(self.parent['ema_model'],strict=True);self.teacher=result;e=result.id_backbone.encoder;self.teacher_offset=dict(train=e._full_train_packets,eval=e._full_eval_packets)
    d.write(self.out/'EMA_initialization.json',dict(origin='strict own original P43 stage20 EMA',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True))
   return result
  def terminal_status(**kw):kw['mechanism_gates_required']=False;return terminal(**kw)
  parent_args=SimpleNamespace(**self.parent['args']);parent_aug_base=native.build_aug_base_cfg(parent_args)
  native.deepcopy=copied;native._resolve_phase1_terminal_status=terminal_status;native.configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:aug(obj,parent_aug_base,13,parent_args);supcon.schedule=lambda epoch:schedule(int(epoch)+20)
  try:
   with native_scope(self,'CENTER_UNITS'):yield
  finally:
   native.deepcopy=orig;native.configure_augmentor_for_epoch=aug;supcon.schedule=schedule;native._resolve_phase1_terminal_status=terminal;native.make_augmentor=factory;native._strong_augment=strong
   if hasattr(self,'sequence'):self.sequence.close()
   if patch_active():raise ValueError('L primitive wrapper leaked outside call')
 def epoch(self,m,ctx,rows,out,device):
  old.Observer.epoch(self,m,ctx,rows,out,device)
  steps=len(rows)*222;records=self.ordinary_records
  if len(self.U_records)!=steps or sum(r['packets'] for r in self.U_records)!=len(rows)*56700 or any(r['step']!=i+1 or not r['native_output_passthrough'] or not r['L_primitive_wrappers_inactive'] or not r['post_return_rng_preserved'] for i,r in enumerate(self.U_records)):raise ValueError('Original U strong call sequence/pass-through differs')
  d.write(out/'ordinary_channel_progress.json',dict(epoch=len(rows),cumulative_epoch=20+len(rows),steps=steps,ordinary_calls=len(records),ordinary_packets=steps*128,U_calls=len(self.U_records),U_packets=sum(r['packets'] for r in self.U_records),MP_calls=sum(r['MP_calls'] for r in records),AWGN_calls=sum(r['AWGN_calls'] for r in records),primitive_wrappers_restored=all(r['primitive_restored'] for r in records),target_access=False))
 def finish(self):
  m=self.model;done=d.read(self.out/'completion.json');cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);steps=self.c['updates']
  check_checkpoint(cp,run=d.RUN,row=self.c['row_id'],epoch=10,updates=steps,baseline=self.c['parent_source']+'/final_ssdg.pth')
  if cp['a1_ema_successful_updates']!=steps or done['optimizer_steps']!=steps or len(m._full_records)!=steps or torch.equal(self.initial_head,m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach()):raise ValueError('Default head must jointly train through fixed budget')
  if len(self.ordinary_records)!=steps or len(m._oc_records)!=steps or len(self.U_records)!=steps or self.ordinary_factory.calls!=1:raise ValueError('Original L/U call counts differ')
  for i,(a,u) in enumerate(zip(self.ordinary_records,self.U_records),1):
   if a['step']!=i or a['count']!=128 or not a['enabled'] or not a['primitive_restored'] or not a['post_return_rng_preserved'] or not a['input_unchanged'] or u['step']!=i or not u['native_output_passthrough'] or not u['L_primitive_wrappers_inactive']:raise ValueError('L-only bypass/U passthrough differs')
  for name,records in [('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records),('shape_steps',m._shape_records),('ordinary_channel_steps',m._oc_records),('ordinary_draw_steps',self.ordinary_records),('U_passthrough_steps',self.U_records)]:
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(dict(r,cumulative_step=4440+r['step']),allow_nan=False)+'\n')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  summary=dict(mode='L_ordinary_native_AWGN_MP_execute_then_bypass',ordinary_factory_calls=self.ordinary_factory.calls,ordinary_calls=steps,ordinary_packets=steps*128,U_calls=len(self.U_records),U_packets=sum(r['packets'] for r in self.U_records),U256_calls=sum(r['packets']==256 for r in self.U_records),U124_calls=sum(r['packets']==124 for r in self.U_records),U_native_output_passthrough=True,U_weak_strong_unchanged=True,As_raw_path_unchanged=True,first20_full_ordinary_retained=True,whole_training_RNG_matched=False,full_native_aug_computation=True,new_losses=0,new_parameters=0,extra_NN=0,pair_weights=[.5,.5,.25],total_L_scale=1.25,returned_ordinary_raw_MSE_mean=sum(r['returned_ordinary_raw_MSE'] for r in self.ordinary_records)/steps)
  for key in ('MP_calls','MP_packets','AWGN_calls','AWGN_packets','shadow_MP_packets','shadow_AWGN_packets','primitive_CPU_rng_advanced','primitive_CUDA_rng_advanced','primitive_Python_rng_advanced'):summary[key]=sum(r[key] for r in self.ordinary_records)
  summary.update(MP_hit_steps=sum(r['MP_hit'] for r in self.ordinary_records),AWGN_hit_steps=sum(r['AWGN_hit'] for r in self.ordinary_records),both_hit_steps=sum(r['MP_hit'] and r['AWGN_hit'] for r in self.ordinary_records),all_primitive_restored=all(r['primitive_restored'] for r in self.ordinary_records),all_post_return_rng_preserved=all(r['post_return_rng_preserved'] for r in self.ordinary_records))
  result=dict(status='SEGMENT_TRAINED',config=self.c,segment_epochs=10,optimizer_steps=steps,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,Ag_mode=self.c['Ag_mode'],Ag_packets=m._shape_packets,Ag_control_NN=m._shape_NN,total_L_loss_scale=1.25,extra_U_NN=0,U_As=0,student_train_NN=e._full_train_packets,student_eval_NN=e._full_eval_packets,teacher_train_NN=t._full_train_packets-self.teacher_offset['train'],teacher_eval_NN=t._full_eval_packets-self.teacher_offset['eval'],teacher_counter_offset=self.teacher_offset,source_qualification=None,U_source_qualified=None,U_steps=steps,U_packets=567000,U_student_NN=567000,L_channel=m._full_channel.summary(),U_channel=None,source_RX=list(d.RX),source_RX_holdout=None,U_truth_read=False,target_access=False,head_fit_updates=0,wall_seconds=done['elapsed_seconds'],parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,EMA_local_successful_updates=cp['a1_ema_successful_updates'],EMA_cumulative_updates=4440+cp['a1_ema_successful_updates'],parent_EMA_restored=True,ordinary_channel=summary,coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(m._full_coverage.items())])
  if m._full_channel.calls!=steps or m._full_channel.packets!=(142080 if self.c['As'] else 0) or summary['U_packets']!=567000 or summary['U124_calls']!=10:raise ValueError('Complete original L As and U budget differs')
  n=sys.modules['SSDG.train_ssdg'];a=SimpleNamespace(**self.parent['args']);d.write(self.out/'ordinary_augmentation_actual.json',dict(base=n.build_aug_base_cfg(a),schedule_frozen_at=13,args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},schedule_args_epochs=self.parent['args']['epochs'],source='strict own original P43 stage20 checkpoint args',no_ordinary_before_practical=True,RDrop_same_IQ=True,use_aug=True,removed_primitives=['AWGN','MP'],full_native_aug_computation=True,U_augmentor_wrapped=False,U_strong_guard_only=True,actual_ORD128_anchor_input='native ordinary with MP/AWGN output bypass',whole_training_RNG_matched=False))
  d.write(self.out/'result.json',result);return result

def train(c):
 from . import model
 from .source import segment_train
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
