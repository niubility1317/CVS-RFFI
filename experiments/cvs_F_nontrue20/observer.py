"""Original source/U trainer, actual F features and backbone update evidence."""
from contextlib import contextmanager
from pathlib import Path
import gzip,json,sys
import torch
from . import design as d,model as api
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_fullsource_concat20.source import RecordingLoader
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
class Observer:
 def __init__(self,c):self.c=c;self.optimizer=None
 def data(self,native,ctx,out):
  Path(out).mkdir(parents=True,exist_ok=True)
  if ctx['num_domains']!=15:raise ValueError('Five source RX x three days required')
  self.sequence=gzip.open(Path(out)/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  for key,role in (('train_loader','L'),('unlabeled_loader','U')):ctx[key]=RecordingLoader(ctx[key],role,self.sequence)
 def initialize(self,m,c,out):
  self.model=m;self.out=Path(out)
  if m._legacy_mode!='ORIGINAL' or m._legacy_alpha!=0. or any('supcon' in k for k in vars(m)):raise ValueError('No RDrop or CROSS_RX allowed in fixed F')
  d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual')})
  if m._nontrue_channel is not None:d.write(self.out/'actual_channel.json',m._nontrue_channel.audit())
 @contextmanager
 def native_scope(self,native):
  original=native.deepcopy
  def copied(v,*a,**kw):
   result=original(v,*a,**kw)
   if v is getattr(self,'model',None):self.ema=result
   return result
  native.deepcopy=copied
  try:
   with native_scope(self,'CENTER_UNITS'):yield
  finally:
   native.deepcopy=original
   if hasattr(self,'sequence'):self.sequence.close()
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222;nf=self.c['training_mechanism']['L_forwards_per_sample'];nt=self.c['training_mechanism']['teacher_packets_per_step']
  if m._feature_l_steps!=n or m._legacy_calls!=n or m._nontrue_steps!=n or m._nontrue_L_packets!=n*128*nf or m._teacher_packets!=n*nt or self.optimizer._sinc_physical_clip_calls!=n or self.optimizer._sinc_unit_steps!=n:raise ValueError('Actual pure F/forward/teacher/optimizer budget differs')
  if m._nontrue_channel is not None and m._nontrue_channel.packets!=n*128:raise ValueError('Physical As budget differs')
  d.write(out/'reference_progress.json',dict(epoch=ep,steps=n,L_packets=m._nontrue_L_packets,teacher_packets=m._teacher_packets,source_RX=list(d.FIT),RDrop=False,CROSS_RX=False,U_truth_read=False,target_access=False))
 def finish(self):
  out=self.out;m=self.model;cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);done=d.read(out/'completion.json');initial=torch.load(out/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];state=cp['model']
  if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['final_source_metrics']['source_val_count']!=27000 or cp.get('ema_model') is None or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt']:raise ValueError('Own scratch20/defaulthead/ownEMA full source required')
  W='id_backbone.encoder.core.id_backbone.cls_head.weight'
  if state[W].shape!=(6,160) or len(m._nontrue_records)!=4440 or any('_class_teacher' in k for k in state):raise ValueError('Default classifier or no-deploy-teacher state differs')
  changed={k:dict(l2=float((v.float()-initial[k].float()).norm()),initial_l2=float(initial[k].float().norm())) for k,v in state.items() if k in dict(m.named_parameters()) and not torch.equal(v,initial[k])}
  if W not in changed or not any('.sinc.' in k for k in changed) or not any('.sequence_pool.' in k for k in changed) or not any('.core.id_backbone.' in k and '.cls_head.' not in k and '.sinc.' not in k for k in changed):raise ValueError('Sinc, signature, body and default head must actually update')
  m.eval();device=next(m.parameters()).device
  with torch.no_grad():feat=m.id_backbone.encoder.features(torch.zeros(2,2,256,device=device))
  if feat.shape!=(2,160) or not torch.isfinite(feat).all():raise ValueError('Actual deployment joint160 required')
  teacher=m.__dict__['_class_teacher']
  if teacher is not None and (any(p.grad is not None or p.requires_grad for p in teacher.parameters()) or any(not torch.equal(v.cpu(),m.__dict__['_teacher_initial'][k]) for k,v in teacher.state_dict().items())):raise ValueError('Frozen CSIL changed')
  for name,records in [('mechanism_steps',m._nontrue_records),('unit_steps',self.optimizer._sinc_unit_lr_records)]:
   with gzip.open(out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
  native=sys.modules['SSDG.train_ssdg'];a=type('ActualArgs',(),cp['args'])()
  d.write(out/'ordinary_augmentation_actual.json',dict(use_aug=cp['args']['use_aug'],base_config=native.build_aug_base_cfg(a),args={k:v for k,v in cp['args'].items() if k.startswith('aug_')},practical_input_is_raw=True,ordinary_before_As=False,As_after_normalization=False,physical_channel_reused_unchanged=True))
  result=dict(status='PURE_F_SOURCE_TRAINED',config=self.c,steps=4440,joint160_shape=list(feat.shape),changed_parameters=changed,trainable_parameters=sum(p.numel() for p in m.parameters()),student_train_packets=m.id_backbone.encoder._full_train_packets,student_eval_packets=m.id_backbone.encoder._full_eval_packets,L_packets=m._nontrue_L_packets,teacher_packets=m._teacher_packets,teacher_frozen=True,channel=None if m._nontrue_channel is None else m._nontrue_channel.summary(),source_metrics=done['final_source_metrics'],CROSS_RX=False,RDrop=False,head_fit_updates=0,target_access=False)
  d.write(out/'training_result.json',result);return result
def train(c):
 o=Observer(c);shared_train(c,design=d,model_api=api,observer=o);return o.finish()
