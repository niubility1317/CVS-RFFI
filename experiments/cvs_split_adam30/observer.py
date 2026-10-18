"""Count original six-view/old-U execution and both extra VJPs each update."""
from contextlib import contextmanager
import gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d,core
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
  if any(not torch.equal(v.detach().cpu(),self.parent['model'][k]) for k,v in m.state_dict().items()):raise ValueError('Own parent strict state differs before training')
  d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ['SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual']});d.write(self.out/'actual_channel.json',m._full_channel.audit())
 def optimizer_created(self,opt):
  if opt.state:raise ValueError('Each original segment resets AdamW state')
  core.attach_optimizer(opt,self.model,amp=False)
  d.write(self.out/'optimizer_initialization.json',dict(state_entries=0,reset=True,schedule='cosine10',lr=opt.param_groups[0]['lr'],weight_decay=opt.param_groups[0]['weight_decay'],mode=self.c['optimizer_mode'],amp=False,one_optimizer=True,independent_rest_As_moments=self.c['optimizer_mode']=='SPLIT',adaptive_weights=[.8,.2]))
 @contextmanager
 def native_scope(self,native):
  orig=native.deepcopy;aug=native.configure_augmentor_for_epoch;schedule=supcon.schedule;terminal=native._resolve_phase1_terminal_status
  def copied(v,*args,**kw):
   result=orig(v,*args,**kw)
   if v is getattr(self,'model',None):
    result.load_state_dict(self.parent['ema_model'],strict=True);self.teacher=result;e=result.id_backbone.encoder;self.teacher_offset=dict(train=e._full_train_packets,eval=e._full_eval_packets)
    d.write(self.out/'EMA_initialization.json',dict(origin='strict own stage20 EMA',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True))
   return result
  def terminal_status(**kw):kw['mechanism_gates_required']=False;return terminal(**kw)
  native.deepcopy=copied;native._resolve_phase1_terminal_status=terminal_status;parent_args=SimpleNamespace(**self.parent['args']);native.configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:aug(obj,cfg,13,parent_args);supcon.schedule=lambda epoch:schedule(int(epoch)+20)
  try:
   with native_scope(self,'CENTER_UNITS'):
    physical_clip=torch.nn.utils.clip_grad_norm_
    def clipped(parameters,max_norm,*args,**kwargs):
     params=list(parameters);total=physical_clip(params,max_norm,*args,**kwargs);core.after_clip(self.model,params,max_norm,total);return total
    torch.nn.utils.clip_grad_norm_=clipped
    try:yield
    finally:torch.nn.utils.clip_grad_norm_=physical_clip
  finally:
   native.deepcopy=orig;native.configure_augmentor_for_epoch=aug;supcon.schedule=schedule;native._resolve_phase1_terminal_status=terminal
   if hasattr(self,'sequence'):self.sequence.close()
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222
  if m._full_channel.calls!=n or m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*512 or m._full_channel.packets!=(n*64 if self.c['As'] else 0) or m._shape_NN!=n*256 or len(m._shape_records)!=n or m._shape_packets!=0 or m._supcon_calls!=n or m._supcon_applied!=n or self.optimizer._sinc_unit_steps!=n or self.optimizer._sinc_physical_clip_calls!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Real L/U/SupCon/Sinc budget differs')
  if m._sa_steps!=n or m._sa_clips!=n or m._sa_vjp_calls!=2*n or len(m._sa_records)!=n or len(m._sa_component_records)!=n or len(m._sa_native_records)!=n or m._sa_as_pending is not None or m._sa_pending is not None or m._sa_ready is not None:raise ValueError('Exactly two VJPs/one clip/one native step per update required')
  actual_U=sum(r['unlabeled_count'] for r in m._sa_native_records);e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  if actual_U!=ep*56700 or e._full_train_packets!=n*768+actual_U or t._full_eval_packets-self.teacher_offset['eval']!=actual_U or t._full_train_packets-self.teacher_offset['train']!=0:raise ValueError('Original active U student/teacher physical count differs')
  d.write(out/'reference_progress.json',dict(epoch=ep,cumulative_epoch=ep+20,steps=n,cumulative_steps=n+4440,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,U_steps=len(m._sa_native_records),U_physical_samples=actual_U,U_student_NN=actual_U,U_teacher_NN=actual_U,extra_U_NN=0,U_As=0,existing_U_only=True,U_truth_read=False,target_access=False,optimizer_mode=self.c['optimizer_mode'],extra_VJP_calls=m._sa_vjp_calls,optimizer_steps=m._sa_steps))
 def finish(self):
  m=self.model;done=d.read(self.out/'completion.json');cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);steps=self.c['updates'];self.epoch(m,None,[None]*10,self.out,None)
  if cp['a1_ema_successful_updates']!=steps or done['optimizer_steps']!=steps or len(m._full_records)!=steps or torch.equal(self.initial_head,m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach()) or cp['args'].get('amp',True):raise ValueError('Default head and native noAMP joint training must complete')
  for name,records in [('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records),('shape_steps',m._shape_records),('split_steps',m._sa_records),('split_components',m._sa_component_records),('split_native',m._sa_native_records)]:
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(dict(r,cumulative_step=4440+r['step']),allow_nan=False)+'\n')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder;actual_U=sum(r['unlabeled_count'] for r in m._sa_native_records)
  result=dict(status='SEGMENT_TRAINED',config=self.c,segment_epochs=10,optimizer_steps=steps,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,Ag_mode='NONE',Ag_packets=0,Ag_control_NN=m._shape_NN,total_L_loss_scale=1.25,extra_U_NN=0,U_As=0,U_steps=len(m._sa_native_records),U_physical_samples=actual_U,U_student_NN=actual_U,U_teacher_NN=actual_U,U_source_qualified_count_recorded=False,student_train_NN=e._full_train_packets,student_eval_NN=e._full_eval_packets,teacher_train_NN=t._full_train_packets-self.teacher_offset['train'],teacher_eval_NN=t._full_eval_packets-self.teacher_offset['eval'],teacher_counter_offset=self.teacher_offset,L_channel=m._full_channel.summary(),U_channel=None,source_RX=list(d.RX),source_RX_holdout=None,U_truth_read=False,target_access=False,head_fit_updates=0,wall_seconds=done['elapsed_seconds'],parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,EMA_local_successful_updates=cp['a1_ema_successful_updates'],EMA_cumulative_updates=4440+cp['a1_ema_successful_updates'],optimizer_mode=self.c['optimizer_mode'],extra_VJP_calls=m._sa_vjp_calls,extra_VJP_explanation='two retained-graph backward traversals each update;zero new neural forward',split_adam=dict(self.c['split_adam']),native_U_unchanged=True,AMP=False,coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(m._full_coverage.items())])
  n=sys.modules['SSDG.train_ssdg'];a=SimpleNamespace(**self.parent['args']);d.write(self.out/'ordinary_augmentation_actual.json',dict(base=n.build_aug_base_cfg(a),schedule_frozen_at=13,args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},schedule_args_epochs=self.parent['args']['epochs'],source='strict own original20 checkpoint args',no_ordinary_before_practical=True,RDrop_same_IQ=True))
  for record in m._sa_component_records:
   if record['As']!=self.c['As'] or record['native_RD_calls']!=3 or record['masked_RD_recomputations']!=int(self.c['As']) or record['extra_NN'] or record['U_changed'] or record['L_forwards']!=6 or (not self.c['As'] and record['As_RD']!=0.):raise ValueError('Original loss decomposition or zero added NN differs')
  d.write(self.out/'result.json',result);return result
def train(c):
 from . import model
 from .source import segment_train
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
