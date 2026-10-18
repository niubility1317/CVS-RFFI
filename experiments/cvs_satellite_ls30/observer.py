"""Record complete original and extra CE scalar compute costs; never consult target."""
from contextlib import contextmanager
import gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .source import parent_payload
from experiments.cvs_fullsource_concat20.source import RecordingLoader
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
from experiments.cvs_fullsource_concat20 import supcon

class Observer:
 def __init__(self,c):self.c=c;self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c)
 def data(self,native,ctx,out):
  self.out.mkdir(parents=True,exist_ok=True)
  if ctx['num_domains']!=15:raise ValueError('All original RX/day domains required')
  self.sequence=gzip.open(self.out/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  for key,role in [('train_loader','L'),('unlabeled_loader','U')]:ctx[key]=RecordingLoader(ctx[key],role,self.sequence)
 def initialize(self,m,c,out):
  self.model=m;self.initial_head=m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach().clone()
  if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206:raise ValueError('Fixed original F shape differs')
  if self.parent is not None and any(not torch.equal(v.detach().cpu(),self.parent['model'][k]) for k,v in m.state_dict().items()):raise ValueError('Own parent strict state differs before training')
  d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ['SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual']})
  d.write(self.out/'actual_channel.json',m._full_channel.audit())
 def optimizer_created(self,opt):
  if opt.state:raise ValueError('Each original segment resets AdamW state')
  d.write(self.out/'optimizer_initialization.json',dict(state_entries=0,reset=True,schedule='cosine'+str(self.c['epochs']),lr=opt.param_groups[0]['lr'],weight_decay=opt.param_groups[0]['weight_decay']))
 @contextmanager
 def native_scope(self,native):
  orig=native.deepcopy;aug=native.configure_augmentor_for_epoch;schedule=supcon.schedule;terminal=native._resolve_phase1_terminal_status
  def copied(v,*args,**kw):
   result=orig(v,*args,**kw)
   if v is getattr(self,'model',None):
    if self.parent is not None:result.load_state_dict(self.parent['ema_model'],strict=True)
    self.teacher=result;e=result.id_backbone.encoder;self.teacher_offset=dict(train=e._full_train_packets,eval=e._full_eval_packets)
    d.write(self.out/'EMA_initialization.json',dict(origin='own scratch student' if self.parent is None else 'strict own stage20 EMA',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True))
   return result
  def terminal_status(**kw):kw['mechanism_gates_required']=False;return terminal(**kw)
  native.deepcopy=copied;native._resolve_phase1_terminal_status=terminal_status
  if self.parent is not None:
   parent_args=SimpleNamespace(**self.parent['args']);native.configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:aug(obj,cfg,13,parent_args);supcon.schedule=lambda epoch:schedule(int(epoch)+20)
  try:
   with native_scope(self,'CENTER_UNITS'):yield
  finally:
   native.deepcopy=orig;native.configure_augmentor_for_epoch=aug;supcon.schedule=schedule;native._resolve_phase1_terminal_status=terminal
   if hasattr(self,'sequence'):self.sequence.close()
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222;expected_U=ep*56700 if self.c['segment']==30 else max(0,ep-13)*56700;sup=n if self.c['segment']==30 else max(0,ep-5)*222
  if m._full_channel.calls!=n or len(m._satls_records)!=n or m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*512 or m._full_channel.packets!=(n*64 if self.c['As'] else 0) or m._shape_NN!=n*256 or len(m._shape_records)!=n or m._supcon_calls!=n or m._supcon_applied!=sup or self.optimizer._sinc_unit_steps!=n or self.optimizer._sinc_physical_clip_calls!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Real L/U/SupCon/Sinc budget differs')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  if e._full_train_packets!=n*768+expected_U or t._full_train_packets-self.teacher_offset['train']!=0 or t._full_eval_packets-self.teacher_offset['eval']!=expected_U:raise ValueError('Actual original U/student/teacher forward budget differs')
  d.write(out/'reference_progress.json',dict(epoch=ep,cumulative_epoch=ep+self.c['epoch_offset'],steps=n,cumulative_steps=n+self.c['step_offset'],L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,extra_U_NN=0,U_As=0,existing_U_only=True,U_truth_read=False,target_access=False))
 def finish(self):
  m=self.model;done=d.read(self.out/'completion.json');cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);steps=self.c['updates']
  if cp['a1_ema_successful_updates']!=steps or done['optimizer_steps']!=steps or len(m._full_records)!=steps or torch.equal(self.initial_head,m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach()):raise ValueError('Default head must jointly train through fixed budget')
  for name,records in [('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records),('shape_steps',m._shape_records),('satls_steps',m._satls_records)]:
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(dict(r,cumulative_step=r['step'] if name=='U_steps' else self.c['step_offset']+r['step']),allow_nan=False)+'\n')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  result=dict(status='SEGMENT_TRAINED',config=self.c,segment_epochs=self.c['epochs'],optimizer_steps=steps,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,Ag_mode=self.c['Ag_mode'],Ag_packets=m._shape_packets,Ag_control_NN=m._shape_NN,total_L_loss_scale=1.25,extra_U_NN=0,U_As=0,student_train_NN=e._full_train_packets,student_eval_NN=e._full_eval_packets,teacher_train_NN=t._full_train_packets-self.teacher_offset['train'],teacher_eval_NN=t._full_eval_packets-self.teacher_offset['eval'],teacher_counter_offset=self.teacher_offset,source_qualification=None,U_source_qualified=None,U_steps=steps,U_packets=567000,U_student_NN=567000,L_channel=m._full_channel.summary(),U_channel=None,source_RX=list(d.RX),source_RX_holdout=None,U_truth_read=False,target_access=False,head_fit_updates=0,wall_seconds=done['elapsed_seconds'])
  n=sys.modules['SSDG.train_ssdg'];a=SimpleNamespace(**self.parent['args']);d.write(self.out/'ordinary_augmentation_actual.json',dict(base=n.build_aug_base_cfg(a),schedule_frozen_at=13,args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},schedule_args_epochs=self.parent['args']['epochs'],source='strict own original20 checkpoint args',no_ordinary_before_practical=True,RDrop_same_IQ=True))
  if m._full_channel.calls!=steps or len(m._satls_records)!=steps or m._full_channel.packets!=(142080 if self.c['As'] else 0):raise ValueError('Unchanged64 selected mask/channel budget differs')
  for r in m._satls_records:
   if r['selected_packets']!=64 or r['As_packets']!=64 or r['epsilon']!=.05 or r['native_label_smoothing']!=.01 or r['extra_CE_operator_calls']!=4 or r['extra_CE_logit_packets']!=256 or r['extra_NN'] or r['explicit_CUDA_synchronize'] or len(r['selected_L_positions'])!=64:raise ValueError('Fixed As-only label smoothing/extra compute differs')
   if abs(r['loss_delta']-.5*r['mixed_CE_delta'])>3e-6 or abs(r['mixed_RD_new']-r['mixed_RD_old']-r['mixed_CE_delta'])>3e-6:raise ValueError('CE-only normalized delta differs')
  satls=dict(mode='selectedAs_CE05',old_epsilon=.01,epsilon=.05,native_label_smoothing=.01,selected_packets=142080,actual_As_packets=142080,channel_calls=steps,pair_weights=[.5,.5,.25],SupCon='ORD128',total_L_scale=1.25,mixed_KL_unchanged=True,common_raw_U_SupCon_unchanged=True,extra_NN=0,extra_CE_operator_calls=sum(r['extra_CE_operator_calls'] for r in m._satls_records),extra_CE_logit_packets=sum(r['extra_CE_logit_packets'] for r in m._satls_records),baseline_mixed_CE_operator_widths=[128,128],extra_CE_operator_widths=[64,64,64,64],actual_mixed_CE_operator_widths=[128,128,64,64,64,64],CE_operator_cpu_wall_seconds=sum(r['CE_operator_cpu_wall_seconds'] for r in m._satls_records),scalarization_cpu_wall_seconds=sum(r['scalarization_cpu_wall_seconds'] for r in m._satls_records),timing='perf_counter CPUwall,notCUDAevent;no explicitCUDA synchronization;scalar reads may synchronize',confidence_filter=False,all_six_classes_same_formula=True,new_U_views=0,sampler_changed=False)
  result.update(parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,EMA_local_successful_updates=cp['a1_ema_successful_updates'],EMA_cumulative_updates=4440+cp['a1_ema_successful_updates'],satellite_label_smoothing=satls,coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(m._full_coverage.items())])
  d.write(self.out/'result.json',result);return result

def train(c):
 from . import model
 from .source import segment_train
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
