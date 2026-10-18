"""Actual native U qualification/routing and original L/U/EMA budgets;no target access."""
from contextlib import contextmanager
from collections import Counter
import gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .source import parent_payload
from .u_scope import NativeUScope,PendingLoader
from .u_routing import Router
from experiments.cvs_fullsource_concat20.source import RecordingLoader
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
from experiments.cvs_fullsource_concat20 import supcon
class Observer:
 def __init__(self,c):self.c=c;self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c);self.pending_U=None;self.u_records=[]
 def data(self,native,ctx,out):
  self.out.mkdir(parents=True,exist_ok=True)
  if ctx['num_domains']!=15:raise ValueError('All original RX/day domains required')
  self.sequence=gzip.open(self.out/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  ctx['train_loader']=RecordingLoader(ctx['train_loader'],'L',self.sequence)
  ctx['unlabeled_loader']=PendingLoader(RecordingLoader(ctx['unlabeled_loader'],'U',self.sequence),self)
 def initialize(self,m,c,out):
  self.model=m;self.initial_head=m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach().clone();self.u_router=Router(c)
  if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206:raise ValueError('Fixed original F shape differs')
  if any(not torch.equal(v.detach().cpu(),self.parent['model'][k]) for k,v in m.state_dict().items()):raise ValueError('Same own B01 parent strict state differs before training')
  d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ['SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual']})
  d.write(self.out/'actual_channel.json',m._full_channel.audit());d.write(self.out/'actual_U_channel.json',self.u_router.channel.audit())
 def optimizer_created(self,opt):
  if opt.state:raise ValueError('Each original segment resets AdamW state')
  d.write(self.out/'optimizer_initialization.json',dict(state_entries=0,reset=True,schedule='cosine10',lr=opt.param_groups[0]['lr'],weight_decay=opt.param_groups[0]['weight_decay']))
 @contextmanager
 def native_scope(self,native):
  orig=native.deepcopy;aug=native.configure_augmentor_for_epoch;schedule=supcon.schedule;terminal=native._resolve_phase1_terminal_status;u_scope=NativeUScope(self,native);u_scope.install()
  def copied(v,*args,**kw):
   result=orig(v,*args,**kw)
   if v is getattr(self,'model',None):
    result.load_state_dict(self.parent['ema_model'],strict=True);self.teacher=result;e=result.id_backbone.encoder;self.teacher_offset=dict(train=e._full_train_packets,eval=e._full_eval_packets);u_scope.teacher_created(result)
    d.write(self.out/'EMA_initialization.json',dict(origin='strict own stage20 EMA',reference_row_id=self.c['reference_row_id'],extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True))
   return result
  def terminal_status(**kw):kw['mechanism_gates_required']=False;return terminal(**kw)
  parent_args=SimpleNamespace(**self.parent['args']);native.deepcopy=copied;native.configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:aug(obj,cfg,13,parent_args);supcon.schedule=lambda epoch:schedule(int(epoch)+20);native._resolve_phase1_terminal_status=terminal_status
  try:
   with native_scope(self,'CENTER_UNITS'):yield
  finally:
   native.deepcopy=orig;native.configure_augmentor_for_epoch=aug;supcon.schedule=schedule;native._resolve_phase1_terminal_status=terminal;u_scope.restore()
   if hasattr(self,'sequence'):self.sequence.close()
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222;U=ep*56700;AsU=ep*14175 if self.c['arm']=='S01' else 0
  if m._full_channel.calls!=n or m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*512 or m._full_channel.packets!=(n*64 if self.c['As'] else 0) or m._shape_NN!=n*256 or len(m._shape_records)!=n or m._supcon_calls!=n or m._supcon_applied!=n or self.optimizer._sinc_unit_steps!=n or self.optimizer._sinc_physical_clip_calls!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Original real L/SupCon/Sinc budget differs')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  if len(self.u_records)!=n or self.pending_U is not None or self.u_router.calls!=n or self.u_router.packets!=U or self.u_router.selected!=ep*14175 or self.u_router.As_packets!=AsU or self.u_router.channel.packets!=AsU or self.u_router.channel.calls!=(n if AsU else 0) or e._full_train_packets!=n*768+U or t._full_train_packets-self.teacher_offset['train'] or t._full_eval_packets-self.teacher_offset['eval']!=U:raise ValueError('Actual oneUstudent/teacher/quarterPractical counts differ')
  d.write(out/'reference_progress.json',dict(epoch=ep,cumulative_epoch=ep+20,steps=n,cumulative_steps=n+4440,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,U_NN=U,teacher_NN=U,U_As=AsU,U_selected_slots=ep*14175,extra_U_NN=0,U_truth_read=False,target_access=False))
 def finish(self):
  m=self.model;done=d.read(self.out/'completion.json');cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);steps=self.c['updates'];sat=self.c['arm']=='S01';a=cp['args']
  if cp['a1_ema_successful_updates']!=steps or done['optimizer_steps']!=steps or len(m._full_records)!=steps or len(self.u_records)!=steps or torch.equal(self.initial_head,m.id_backbone.encoder.core.id_backbone.cls_head.weight.detach()) or a['pseudo_strong_agreement'] or not a['use_ema_teacher'] or not a['use_unlabeled']:raise ValueError('Fixed nativeU/teacher/defaultheadjoint budget differs')
  for name,records in [('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records),('shape_steps',m._shape_records),('u_steps',self.u_records)]:
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(dict(r,cumulative_step=4440+r['step']),allow_nan=False)+'\n')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder;n=sys.modules['SSDG.train_ssdg'];pa=SimpleNamespace(**self.parent['args'])
  d.write(self.out/'ordinary_augmentation_actual.json',dict(base=n.build_aug_base_cfg(pa),schedule_frozen_at=13,args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},schedule_args_epochs=self.parent['args']['epochs'],source='strict own original20 checkpoint args',use_aug=a['use_aug'],no_ordinary_before_practical=True,RDrop_same_IQ=True,U_strong='nativeGaussian.015 allpackets beforeoptionalquarterrawoverwrite',U_DataAugmentation_used=False))
  if m._full_channel.calls!=steps or m._full_channel.packets!=(142080 if self.c['As'] else 0) or self.u_router.packets!=567000 or self.u_router.selected!=141750 or self.u_router.As_packets!=(141750 if sat else 0):raise ValueError('L/U physical dosage differs')
  exposure=Counter();eligibility=Counter();scene=Counter()
  for r in self.u_records:
   selected=set(r['selected_positions'])
   for i in range(r['U_packets']):
    kind='As' if sat and i in selected else 'Gaussian';key=(r['RX'][i],r['day'][i],kind);exposure[key]+=1;eligibility[key]+=int(r['eligibility'][i])
   scene.update(r['scene_counts'])
  timing={k:sum(r[k] for r in self.u_records) for k in ('selection_cpu_wall_seconds','simulation_adapter_cpu_wall_seconds','replacement_copy_cpu_wall_seconds','route_cpu_wall_seconds','original_strong_cpu_wall_seconds','logging_cpu_wall_seconds')}
  u=dict(self.u_router.summary(),teacher_only_eligibility=True,pseudo_strong_agreement=False,teacher_qualification_before_student_view=True,original_strong_draw_packets=567000,CE_denominator='teacher-qualified slot count acrossallU',entropy_denominator='all U batchslots',lambda_u=.16,lambda_entropy=.01,entropy_includes_unqualified=True,entropy_view_changed=sat,qualified_packets=sum(r['CE_denominator'] for r in self.u_records),qualified_As_packets=sum(r['eligible_As_slots'] for r in self.u_records),qualified_Gaussian_packets=sum(r['eligible_Gaussian_slots'] for r in self.u_records),scene_counts=dict(scene),RX_day_route_exposure=[dict(RX=k[0],day=k[1],route=k[2],count=v,qualified=eligibility[k]) for k,v in sorted(exposure.items())],timing=timing,timing_note='CPU host wall;route inclusive of selection/simulation/copy;do not addinclusive route;no explicit CUDA sync,logging readouts may sync',extra_CE_operator_calls=0,teacher_trajectory_equality_claim=False)
  result=dict(status='SEGMENT_TRAINED',config=self.c,segment_epochs=10,optimizer_steps=steps,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,L_NN=m._full_L_packets+m._shape_NN,L_As=m._full_channel.packets,Ag_mode=self.c['Ag_mode'],Ag_packets=m._shape_packets,Ag_control_NN=m._shape_NN,total_L_loss_scale=1.25,extra_U_NN=0,U_As=self.u_router.As_packets,student_train_NN=e._full_train_packets,student_eval_NN=e._full_eval_packets,teacher_train_NN=t._full_train_packets-self.teacher_offset['train'],teacher_eval_NN=t._full_eval_packets-self.teacher_offset['eval'],teacher_counter_offset=self.teacher_offset,U_steps=steps,U_packets=567000,U_student_NN=567000,L_channel=m._full_channel.summary(),U_channel=self.u_router.channel.summary(),u_practical=u,EMA_local_successful_updates=cp['a1_ema_successful_updates'],EMA_cumulative_updates=4440+cp['a1_ema_successful_updates'],source_RX=list(d.RX),source_RX_holdout=None,U_truth_read=False,target_access=False,head_fit_updates=0,wall_seconds=done['elapsed_seconds'],coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(m._full_coverage.items())])
  d.write(self.out/'result.json',result);return result

def train(c):
 from . import model
 from .source import segment_train
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
