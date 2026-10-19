"""Record complete original and added training costs; never consult target."""
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
 def __init__(self,c):self.c=c;self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c) if c['segment']==30 else None
 def data(self,native,ctx,out):
  self.out.mkdir(parents=True,exist_ok=True)
  if ctx['num_domains']!=15:raise ValueError('All original RX/day domains required')
  self.sequence=gzip.open(self.out/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  for key,role in [('train_loader','L'),('unlabeled_loader','U')]:ctx[key]=RecordingLoader(ctx[key],role,self.sequence)
 def initialize(self,m,c,out):
  from .model import TEACHERS
  self.KD_teacher_offset=sum(v['packets'] for v in TEACHERS.values())
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
  if m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*512 or m._full_channel.packets!=(n*64 if self.c['As'] else 0) or m._shape_NN!=n*256 or len(m._shape_records)!=n or m._supcon_calls!=n or m._supcon_applied!=sup or self.optimizer._sinc_unit_steps!=n or self.optimizer._sinc_physical_clip_calls!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Real L/U/SupCon/Sinc budget differs')
  d.write(out/'reference_progress.json',dict(epoch=ep,cumulative_epoch=ep+self.c['epoch_offset'],steps=n,cumulative_steps=n+self.c['step_offset'],L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,extra_U_NN=0,U_As=0,existing_U_only=True,U_truth_read=False,target_access=False))
 def finish(self):
  m=self.model;done=d.read(self.out/'completion.json');cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);steps=self.c['updates']
  if cp['a1_ema_successful_updates']!=steps or done['optimizer_steps']!=steps or len(m._full_records)!=steps or torch.equal(self.initial_head,m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach()):raise ValueError('Default head must jointly train through fixed budget')
  for name,records in [('KD_steps',m._kd_records),('temperature_steps',m._temperature_records),('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records),('shape_steps',m._shape_records)]:
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(dict(r,cumulative_step=r['step'] if name=='U_steps' else self.c['step_offset']+r['step']),allow_nan=False)+'\n')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  if len(m._temperature_records)!=steps:raise ValueError('Every actual As step must use temperature')
  result=dict(status='SEGMENT_TRAINED',config=self.c,segment_epochs=self.c['epochs'],optimizer_steps=steps,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,Ag_mode=self.c['Ag_mode'],Ag_packets=m._shape_packets,Ag_control_NN=m._shape_NN,total_L_loss_scale=1.25,extra_U_NN=0,U_As=0,student_train_NN=e._full_train_packets,student_eval_NN=e._full_eval_packets,teacher_train_NN=t._full_train_packets-self.teacher_offset['train'],teacher_eval_NN=t._full_eval_packets-self.teacher_offset['eval'],teacher_counter_offset=self.teacher_offset,source_qualification=None,U_source_qualified=None,U_steps=0,L_channel=m._full_channel.summary(),U_channel=None,source_RX=list(d.RX),source_RX_holdout=None,U_truth_read=False,target_access=False,head_fit_updates=0,wall_seconds=done['elapsed_seconds'])
  n=sys.modules['SSDG.train_ssdg'];a=SimpleNamespace(**cp['args']);d.write(self.out/'ordinary_augmentation_actual.json',dict(base=n.build_aug_base_cfg(a),schedule_frozen_at=13,args={k:v for k,v in cp['args'].items() if k.startswith('aug_')},no_ordinary_before_practical=True,RDrop_same_IQ=True))
  from .model import TEACHERS
  actual_teacher=sum(v['packets'] for v in TEACHERS.values())-self.KD_teacher_offset
  if len(m._kd_records)!=steps or actual_teacher!=steps*64 or any(not torch.equal(v,z['model'].state_dict()[k]) for z in TEACHERS.values() for k,v in z['before'].items()) or any(p.requires_grad or p.grad is not None for z in TEACHERS.values() for p in z['model'].parameters()):raise ValueError('Frozen sameAs teacher state/cost changed')
  result['sameAsKD']=dict(teacher_NN=actual_teacher,student_extra_NN=0,steps=len(m._kd_records),teacher_correct=sum(r['teacher_correct'] for r in m._kd_records),teacher_same_IQ=True,teacher_state_unchanged=True,beta=1.,common_scale=5.,head_fit_updates=0)
  d.write(self.out/'result.json',result);return result

def train(c):
 from . import model
 from .source import segment_train
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
