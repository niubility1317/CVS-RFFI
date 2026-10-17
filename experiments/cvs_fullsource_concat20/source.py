"""Original full-source trainer and U; fixed final20, no leave-one-RX observer."""
from contextlib import contextmanager
import gzip,json,sys
from pathlib import Path
import torch
from . import design as d,model as api
from experiments.cvs_feature_sequence50.source import train as shared_train,source_validation
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope

class RecordingLoader:
 def __init__(self,loader,role,stream):self.loader=loader;self.role=role;self.stream=stream
 def __getattr__(self,k):return getattr(self.loader,k)
 def __len__(self):return len(self.loader)
 def __iter__(self):
  for batch in self.loader:
   x,y,domain,meta=batch
   if set(meta['rx_i'].tolist())-set(d.FIT) or (self.role=='U' and bool((y>=0).any())):raise ValueError('Full source membership/hidden U truth required')
   keys=('eq_i','rx_i','day_i','sig_i','base_index');ids=list(zip(*(meta[k].tolist() for k in keys)))
   self.stream.write(json.dumps(dict(role=self.role,ids=ids))+'\n');yield batch

class Observer:
 def __init__(self,c):self.c=c;self.optimizer=None
 def data(self,native,ctx,out):
  Path(out).mkdir(parents=True,exist_ok=True)
  if ctx['num_domains']!=15:raise ValueError('All five source RX and three days required')
  self.sequence=gzip.open(Path(out)/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  for key,role in (('train_loader','L'),('unlabeled_loader','U')):ctx[key]=RecordingLoader(ctx[key],role,self.sequence)
 def initialize(self,m,c,out):
  self.model=m;self.out=Path(out)
  d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual')})
  if m._full_channel is not None:d.write(self.out/'actual_channel.json',m._full_channel.audit())
 @contextmanager
 def native_scope(self,native):
  original=native.deepcopy
  def copied(v,*a,**kw):
   result=original(v,*a,**kw)
   if v is getattr(self,'model',None):
    self.teacher=result;e=result.id_backbone.encoder;self.teacher_offset=dict(train=e._full_train_packets,eval=e._full_eval_packets)
   return result
  native.deepcopy=copied
  try:
   with native_scope(self,'CENTER_UNITS'):yield
  finally:
   native.deepcopy=original
   if hasattr(self,'sequence'):self.sequence.close()
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222;nf=self.c['training_mechanism']['L_forwards_per_sample']
  if m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*128*nf or m._supcon_calls!=n or m._supcon_applied!=max(0,ep-5)*222 or self.optimizer._sinc_physical_clip_calls!=n or self.optimizer._sinc_unit_steps!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('F/SupCon/actual forwards/optimizer budget differs')
  if m._full_channel is not None and m._full_channel.packets!=n*128:raise ValueError('All physical satellite L required')
  d.write(out/'reference_progress.json',dict(epoch=ep,steps=n,L_packets=m._full_L_packets,As_packets=0 if m._full_channel is None else m._full_channel.packets,source_RX=list(d.FIT),source_RX_holdout=None,U_truth_read=False,target_access=False))
 def finish(self):
  done=d.read(self.out/'completion.json');m=self.model;c=self.c;cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt'] or cp['args'].get('source_fold_held_rx') is not None or cp['checkpoint_selection']!='final_only' or cp['a1_ema_successful_updates']!=4440 or cp.get('ema_model') is None or done['final_source_metrics']['source_val_count']!=27000:raise ValueError('Full-source own scratch20 defaulthead/U provenance required')
  initial=torch.load(self.out/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];W='id_backbone.encoder.core.id_backbone.cls_head.weight'
  if cp['model'][W].shape!=(6,160) or torch.equal(initial[W],cp['model'][W]) or len(m._full_records)!=4440:raise ValueError('Default head must train jointly for complete budget')
  for name,records in (('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records)):
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  result=dict(status='FULL_SOURCE_TRAINED',config=c,actual_steps=4440,L_forward_packets=m._full_L_packets,student_train_forward_packets=e._full_train_packets,student_eval_forward_packets=e._full_eval_packets,teacher_train_forward_packets=t._full_train_packets-self.teacher_offset['train'],teacher_eval_forward_packets=t._full_eval_packets-self.teacher_offset['eval'],teacher_inherited_counter_offset=self.teacher_offset,SupCon_applied=m._supcon_applied,SupCon_layer_forward_terms=sum(len(r['SupCon_details']) for r in m._full_records),As_training_observations=0 if m._full_channel is None else m._full_channel.packets,channel=None if m._full_channel is None else m._full_channel.summary(),coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(m._full_coverage.items())],seen_V=done['final_source_metrics'],source_RX_holdout=None,head_fit_updates=0,U_truth_read=False,target_access=False,elapsed_seconds=done['elapsed_seconds'])
  native=sys.modules['SSDG.train_ssdg'];a=type('ActualArgs',(),cp['args'])()
  d.write(self.out/'ordinary_augmentation_actual.json',dict(use_aug=cp['args']['use_aug'],base_config=native.build_aug_base_cfg(a),args={k:v for k,v in cp['args'].items() if k.startswith('aug_')},same_generated_IQ_each_RD_pair=True,ordinary_redraws_in_repeat=0,practical_input_is_raw=True))
  d.write(self.out/'training_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
