"""Full-role observation, native concat evidence and actual forward/optimizer costs."""
from collections import Counter
from contextlib import contextmanager
import gzip,json,sys
from pathlib import Path
import torch
from . import design as d
from .loader import RecordedLoader
from .streams import PrivateAugment,stream
from .f_base.optimizer import native_scope as sinc_scope
class Observer:
 def __init__(self,c):
  self.c=c;self.model=None;self.optimizer=None;self.steps=0;self.observed=Counter();self.coverage=Counter();self.sat_coverage=Counter();self.concat_calls=0;self.sat_packets=0;self.epochs=[];self.augmentor=None;self.current=None;self.last_physics=None
 def record(self,role,batch):
  x,y,domain,meta=batch
  if set(meta['rx_i'].tolist())-set(d.RX) or (role=='U_s' and bool((y>=0).any())):raise ValueError('Non-source RX or visible U TX entered training')
  self.observed[role]+=len(x)
  keys=('eq_i','rx_i','day_i','sig_i','base_index')
  self.sequence.write(json.dumps(dict(role=role,ids=list(zip(*(meta[k].tolist() for k in keys)))))+'\n')
  if role=='L_s':
   if self.current is not None:raise ValueError('Previous labeled metadata was not consumed')
   self.current=(y.tolist(),meta)
   self.coverage.update((int(t),int(r),int(day)) for t,r,day in zip(y,meta['rx_i'],meta['day_i']))
 def data(self,native,ctx,out):
  self.out=Path(out);self.sequence=gzip.open(self.out/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3);self.physics=gzip.open(self.out/'training_physics.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  actual=role_ids_from_native(ctx);expected=d.read(d.SOURCE)
  if actual!=expected['role_ids'] or {k:len(v) for k,v in actual.items()}!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('All original L/U/V physical IDs required; no source fold')
  if ctx['named_test_loaders'] or len(ctx['unlabeled_loader'])!=222:raise ValueError('No training target loader and fixed full U length required')
  ctx['train_loader']=RecordedLoader(ctx['train_loader'],'L_s',self.record)
  ctx['unlabeled_loader']=RecordedLoader(ctx['unlabeled_loader'],'U_s',self.record)
  d.write(self.out/'full_source_roles.json',dict(source_RX=list(d.RX),held_RX=None,physical_roles='EXACT_MATCH',counts={k:len(v) for k,v in actual.items()},extra_Ldev_split=False,U_truth_hidden=True,target_loaders=0))
 def initialize(self,m,c,out):
  self.model=m
  self.initial={k:p.detach().cpu().clone() for k,p in m.named_parameters()}
  d.write(self.out/'module_files.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.batch','cvsrffi.practical_adapter')})
 def expanded(self,result,clean):
  if self.current is None:raise ValueError('Aligned physical L metadata required at native expansion')
  labels,meta=self.current;self.current=None
  if result.x.shape!=(256,2,256) or not torch.equal(result.x[:128],clean) or not torch.equal(result.y[:128],result.y[128:]):raise ValueError('Native 1:1 concatenation differs')
  if result.applied!=self.c['As'] or (not self.c['As'] and not torch.equal(result.x[:128],result.x[128:])):raise ValueError('B00 exactrepeat or B01 realAs differs')
  self.concat_calls+=1
  if self.c['As']:
   from cvsrffi import practical_adapter as pa
   records=pa._last_meta['records']
   if len(records)!=128:raise ValueError('One native physical record per L satellite required')
   self.sat_packets+=128
   for t,rx,day,r in zip(labels,meta['rx_i'].tolist(),meta['day_i'].tolist(),records):
    self.sat_coverage[(t,rx,day,result.scenario,r['state_end'],r['receiver_lock_status'])]+=1
   if self.last_physics is None:
    self.last_physics=pa._last_meta
    d.write(self.out/'actual_practical_first_batch.json',pa._last_meta)
   self.physics.write(json.dumps(dict(step=self.concat_calls,scene=result.scenario,records=records))+'\n')
 @contextmanager
 def native_scope(self,native):
  original_aug=native.make_augmentor;original_concat=native.ConcatSatChannelAugment;original_copy=native.deepcopy;original_detach=native._detach_log_mapping;owner=self
  class Concat(original_concat):
   def expand(self,x,*args,**kwargs):
    result=super().expand(x,*args,**kwargs);owner.expanded(result,x);return result
  def augment(cfg):
   if self.augmentor is not None:raise ValueError('One original ordinary augmentor required')
   with stream(20261016051,0):inner=original_aug(cfg)
   self.augmentor=PrivateAugment(inner,20261016051);d.write(self.out/'ordinary_base_config.json',cfg);return self.augmentor
  def copied(value,*args,**kwargs):
   result=original_copy(value,*args,**kwargs)
   if value is self.model:
    self.teacher=result;result.id_backbone.encoder._full_train_packets=0;result.id_backbone.encoder._full_eval_packets=0
   return result
  def logged(values):
   result=original_detach(values)
   if 'train/loss' in result:self.steps+=1
   return result
  native.make_augmentor=augment;native.ConcatSatChannelAugment=Concat;native.deepcopy=copied;native._detach_log_mapping=logged
  try:
   with sinc_scope(self,'CENTER_UNITS'):yield
  finally:
   native.make_augmentor=original_aug;native.ConcatSatChannelAugment=original_concat;native.deepcopy=original_copy;native._detach_log_mapping=original_detach
   if hasattr(self,'physics'):self.physics.close()
   if hasattr(self,'sequence'):self.sequence.close()
 def epoch(self,m,ctx,rows,out,device):
  epoch=len(rows);steps=epoch*222;e=m.id_backbone.encoder
  if self.steps!=steps or self.concat_calls!=steps or m._full_calls!=steps or m._feature_l_steps!=steps or m._full_pending is not None or self.current is not None or self.augmentor._calls!=steps or m._full_extra_packets!=steps*256 or m._full_supcon_applied!=max(0,epoch-5)*222 or self.optimizer._sinc_unit_steps!=steps or self.optimizer._sinc_physical_clip_calls!=steps:raise ValueError('Exact concat/defaulthead/SupCon/F optimizer E30 path required')
  if e._full_train_packets!=steps*512+self.observed['U_s'] or self.sat_packets!=(steps*128 if self.c['As'] else 0):raise ValueError('Actual L/U/satellite forward budget differs')
  self.sequence.flush();self.physics.flush()
  params=dict(m.named_parameters());selected=[k for k in params if k.endswith(('cls_head.weight','cls_head.gain','local_projection.weight','low_coordinate','band_coordinate'))]
  if not selected or any(not params[k].requires_grad for k in selected):raise ValueError('F backbone and default head must update jointly')
  self.epochs.append(dict(epoch=epoch,steps=steps,loader_packets=dict(self.observed),L_forward_packets=steps*512,student_train_NN=e._full_train_packets,student_eval_NN=e._full_eval_packets,teacher_train_NN=self.teacher.id_backbone.encoder._full_train_packets,teacher_eval_NN=self.teacher.id_backbone.encoder._full_eval_packets,satellite_packets=self.sat_packets,ordinary=self.augmentor.summary(),parameter_movement={k:float((params[k].detach().cpu()-self.initial[k]).norm()) for k in selected},last_optimizer_mapping=self.optimizer._sinc_unit_lr_records[-1]))
  d.write(self.out/'mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False,target_access=False))
 def finish(self):
  done=d.read(self.out/'completion.json');m=self.model
  if done['epoch']!=30 or done['optimizer_steps']!=6660 or self.steps!=6660 or self.observed!=dict(L_s=852480,U_s=567000) or set(self.coverage)!={(tx,rx,day) for tx in range(6) for rx in d.RX for day in (1,2,3)}:raise ValueError('Complete full-source E30 coverage required')
  if self.c['As'] and ({(t,r,s) for t,r,day,s,z,k in self.sat_coverage}!={(t,r,s) for t in range(6) for r in d.RX for s in d.TRAIN_SCENES} or {(t,r,z) for t,r,day,s,z,k in self.sat_coverage}!={(t,r,z) for t in range(6) for r in d.RX for z in ('LOS','PARTIAL','BLOCKED')}):raise ValueError('Full TX x sourceRX scene/state coverage required')
  with gzip.open(self.out/'SupCon_steps.jsonl.gz','xt',encoding='utf8') as f:
   for r in m._full_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  with gzip.open(self.out/'F_optimizer_steps.jsonl.gz','xt',encoding='utf8') as f:
   for r in self.optimizer._sinc_unit_lr_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  result=dict(status='FULLSOURCE_REFERENCE_TRAINED',config=self.c,source_RX=list(d.RX),held_RX=None,LUV=dict(L_s=6300,U_s=56700,V=27000),updates=6660,epochs=30,L_network_packets=3409920,satellite_packets=self.sat_packets,loader_packets=dict(self.observed),actual_counters=dict(student_train_NN=m.id_backbone.encoder._full_train_packets,student_eval_NN=m.id_backbone.encoder._full_eval_packets,teacher_train_NN=self.teacher.id_backbone.encoder._full_train_packets,teacher_eval_NN=self.teacher.id_backbone.encoder._full_eval_packets),coverage=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(self.coverage.items())],satellite_coverage=[dict(TX=k[0],RX=k[1],day=k[2],scene=k[3],state=k[4],lock=k[5],count=v) for k,v in sorted(self.sat_coverage.items())],default_head_joint=True,new_classifier=False,head_fit_updates=0,U_truth_read=False,target_access=False,checkpoint_sources=[],wall_seconds=done['elapsed_seconds'])
  d.write(self.out/'result.json',result);return result
