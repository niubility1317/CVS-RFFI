"""Record actual six-L costs and assert zero child U traversal/teacher/EMA activity."""
from contextlib import contextmanager
import gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .source import parent_payload
from .policy import NoUTraversal
from .protocol import check_checkpoint
from experiments.cvs_fullsource_concat20.source import RecordingLoader
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
from experiments.cvs_fullsource_concat20 import supcon

class Observer:
 def __init__(self,c):
  self.c=c;self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c);self.teacher_builds=0;self.EMA_updates=0
 def data(self,native,ctx,out):
  self.out.mkdir(parents=True,exist_ok=True)
  if ctx['num_domains']!=15 or len(ctx['unlabeled_loader'])!=222:raise ValueError('All original source domains/step budget required')
  self.sequence=gzip.open(self.out/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  ctx['train_loader']=RecordingLoader(ctx['train_loader'],'L',self.sequence)
  self.unlabeled=NoUTraversal(ctx['unlabeled_loader']);ctx['unlabeled_loader']=self.unlabeled
 def initialize(self,m,c,out):
  self.model=m;self.initial_head=m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach().clone()
  if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206:raise ValueError('Fixed original F shape differs')
  if any(not torch.equal(v.detach().cpu(),self.parent['model'][k]) for k,v in m.state_dict().items()):raise ValueError('Own parent strict student state differs')
  d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ['SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual']})
  d.write(self.out/'actual_channel.json',m._full_channel.audit())
 def optimizer_created(self,opt):
  if opt.state:raise ValueError('Original tail resets AdamW state')
  d.write(self.out/'optimizer_initialization.json',dict(state_entries=0,reset=True,schedule='cosine10',lr=opt.param_groups[0]['lr'],weight_decay=opt.param_groups[0]['weight_decay']))
 @contextmanager
 def native_scope(self,native):
  orig=native.deepcopy;aug=native.configure_augmentor_for_epoch;schedule=supcon.schedule;terminal=native._resolve_phase1_terminal_status;ema=native._update_ema_model
  def copied(v,*args,**kw):
   if v is getattr(self,'model',None):
    self.teacher_builds+=1
    raise RuntimeError('Child EMA teacher construction is disabled')
   return orig(v,*args,**kw)
  def update_ema(*args,**kw):
   self.EMA_updates+=1
   raise RuntimeError('Child EMA updates are disabled')
  def terminal_status(**kw):kw['mechanism_gates_required']=False;return terminal(**kw)
  parent_args=SimpleNamespace(**self.parent['args'])
  parent_aug_base=native.build_aug_base_cfg(parent_args)
  native.deepcopy=copied;native._update_ema_model=update_ema;native._resolve_phase1_terminal_status=terminal_status
  native.configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:aug(obj,parent_aug_base,13,parent_args)
  supcon.schedule=lambda epoch:schedule(int(epoch)+20)
  d.write(self.out/'EMA_initialization.json',dict(origin='disabled in child;own20 EMA provenance verified',parent_EMA_verified=True,parent_EMA_restored_in_child=False,teacher_constructed=False,teacher_forwards=0,EMA_updates=0,child_ema_model=None,optimizer_segment_counters_reset=True,temporal_cache_reset=True))
  try:
   with native_scope(self,'CENTER_UNITS'):yield
  finally:
   native.deepcopy=orig;native._update_ema_model=ema;native.configure_augmentor_for_epoch=aug;supcon.schedule=schedule;native._resolve_phase1_terminal_status=terminal
   if hasattr(self,'sequence'):self.sequence.close()
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222
  if self.unlabeled.iteration_attempts or self.unlabeled.packets or self.teacher_builds or self.EMA_updates or m.id_backbone.encoder._full_train_packets!=n*768:raise ValueError('Actual child must contain only six L neural forwards/no U or teacher')
  if m._full_channel.calls!=n or len(m._no_u_records)!=n or m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*512 or m._full_channel.packets!=(n*64 if self.c['As'] else 0) or m._shape_NN!=n*256 or len(m._shape_records)!=n or m._supcon_calls!=n or m._supcon_applied!=n or self.optimizer._sinc_unit_steps!=n or self.optimizer._sinc_physical_clip_calls!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Original L/SupCon/Sinc budget differs')
  d.write(out/'reference_progress.json',dict(epoch=ep,cumulative_epoch=ep+20,steps=n,cumulative_steps=n+4440,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,U_packets=0,U_student_NN=0,teacher_train_NN=0,teacher_eval_NN=0,EMA_updates=0,U_iteration_attempts=0,parent_U_epochs=7,whole_trajectory_NoU=False,U_truth_read=False,target_access=False))
 def finish(self):
  m=self.model;done=d.read(self.out/'completion.json');cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);steps=self.c['updates']
  check_checkpoint(cp,run=d.RUN,row=self.c['row_id'],epoch=10,updates=steps,baseline=self.c['parent_source']+'/final_ssdg.pth',child=True)
  if done['optimizer_steps']!=steps or len(m._full_records)!=steps or len(m._no_u_records)!=steps or torch.equal(self.initial_head,m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach()):raise ValueError('Default head must jointly train through complete fixed budget')
  if self.teacher_builds or self.EMA_updates or self.unlabeled.iteration_attempts or m.id_backbone.encoder._full_train_packets!=steps*768:raise ValueError('Actual U/teacher/EMA activity must remain zero')
  for name,records in [('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records),('shape_steps',m._shape_records),('no_u_steps',m._no_u_records)]:
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(dict(r,cumulative_step=4440+r['step']),allow_nan=False)+chr(10))
  e=m.id_backbone.encoder
  result=dict(status='SEGMENT_TRAINED',config=self.c,segment_epochs=10,optimizer_steps=steps,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,Ag_mode=self.c['Ag_mode'],Ag_packets=m._shape_packets,Ag_control_NN=m._shape_NN,total_L_loss_scale=1.25,extra_U_NN=0,U_As=0,U_packets=0,U_student_NN=0,U_iteration_attempts=self.unlabeled.iteration_attempts,student_train_NN=e._full_train_packets,student_eval_NN=e._full_eval_packets,teacher_train_NN=0,teacher_eval_NN=0,teacher_counter_offset=None,teacher_builds=self.teacher_builds,EMA_local_successful_updates=0,EMA_cumulative_updates=4440,parent_EMA_verified=True,parent_EMA_restored_in_child=False,child_ema_model=None,native_a1_ema_successful_updates_semantics='successful optimizer steps,not actual EMA updates',source_qualification=None,U_source_qualified=None,U_steps=0,L_channel=m._full_channel.summary(),U_channel=None,source_RX=list(d.RX),source_RX_holdout=None,U_truth_read=False,target_access=False,head_fit_updates=0,wall_seconds=done['elapsed_seconds'],parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,parent_U_epochs=7,whole_trajectory_NoU=False,tail_no_u=dict(self.c['tail_no_u'],teacher_builds=0,EMA_updates=0,U_iteration_attempts=0),coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(m._full_coverage.items())])
  n=sys.modules['SSDG.train_ssdg'];a=SimpleNamespace(**self.parent['args']);d.write(self.out/'ordinary_augmentation_actual.json',dict(base=n.build_aug_base_cfg(a),schedule_frozen_at=13,args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},schedule_args_epochs=self.parent['args']['epochs'],source='strict own original20 checkpoint args',no_ordinary_before_practical=True,RDrop_same_IQ=True))
  d.write(self.out/'result.json',result);return result

def train(c):
 from . import model
 from .source import segment_train
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
