"""Whole-tail records, legal labels and exact six-L/physical accounting."""
from contextlib import contextmanager
import gzip,json,sys,math
from pathlib import Path
from . import design as d,model
from .source import parent_payload,segment_train
from experiments.cvs_satellite_dropout18.observer import Observer as Original

def joint(counter):
 return [dict(view=k[0],TX=k[1],RX=k[2],scene=k[3],state=k[4],lock=k[5],count=v) for k,v in sorted(counter.items())]

class Observer(Original):
 def __init__(self,c):
  self.c=d.validate(c);self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c);self.physics_records=0;self.physics_file=None
 def initialize(self,m,c,out):
  super().initialize(m,c,out)
  self.physics_file=gzip.open(self.out/'prop_contrast_physics.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  model.physics_sink(m,self.write_physics)
  actual=dict(config=c,first_channel=m._full_channel.audit(),training_unit=c['quality_entropy'],modules={k:str(v.__file__) for k,v in sys.modules.items() if k in ('experiments.cvs_quality_entropy77.model','experiments.cvs_quality_entropy77.channel','experiments.cvs_quality_entropy77.entropy','experiments.cvs_full_channel_pair68.model','SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual')})
  d.write(self.out/'prop_contrast_actual.json',actual);print(json.dumps(dict(stage='ACTUAL_QUALITY_ENTROPY77',**actual)),flush=True)
 def write_physics(self,r):
  if r['step']!=self.physics_records+1 or len(r['view0'])!=64 or len(r['view1'])!=64 or len(r['labels_L_As'])!=64 or any(isinstance(y,bool) or not isinstance(y,int) or y not in range(6) for y in r['labels_L_As']) or self.physics_file is None:raise ValueError('One ordered legal L64 group and exact copied second view')
  self.physics_file.write(json.dumps(r,allow_nan=False)+chr(10));self.physics_records+=1
 @contextmanager
 def native_scope(self,native):
  try:
   with super().native_scope(native):yield
  finally:
   if hasattr(self,'model'):model.physics_sink(self.model)
   if self.physics_file is not None:self.physics_file.close();self.physics_file=None
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);n=len(rows)*d.STEPS
  if len(m._prop_records)!=n or m._prop_generated!=64*n or m._prop_slots!=128*n or m._full_channel.packets!=64*n or len(m._pair_records)!=n or m._pair_generated!=64*n or sum(m._prop_coverage.values())!=128*n or self.physics_records!=n or m._pair_sessions!=m._full_channel.sessions:raise ValueError('SixL/native64/copy64/group-contrast accounting differs')
  if self.physics_file is not None:self.physics_file.flush()
  d.write(out/'prop_contrast_progress.json',self.prop_summary())
 def prop_summary(self):
  m=self.model;r=m._prop_records;n=len(r);means={}
  if n:
   for k,v in r[0].items():
    if isinstance(v,(int,float)) and not isinstance(v,bool) and all(isinstance(t.get(k),(int,float)) and not isinstance(t.get(k),bool) and math.isfinite(t[k]) for t in r):means[k+'_mean']=sum(t[k] for t in r)/n
  return dict(arm=self.c['arm'],steps=n,generated_As0=m._prop_generated,generated_As1=0,native_waveforms=m._prop_generated,
   As_forward_NN=m._prop_slots,L_NN=n*768,logical_second_slots=m._pair_generated,
   share_propagation=self.c['share_propagation'],As_contrast=self.c['As_contrast'],
   ordinary_SupCon_coefficient=.1,As_SupCon_coefficient=.05 if self.c['As_contrast'] else 0.,
   total_L_CE_scale=1.25,As_CE_coefficient=.25,per_pass_As_CE_coefficient=.125,As_KL=False,
   original_dropout=True,original_ordinary_raw_objectives=True,new_deployment_parameters=0,extra_NN=0,extra_U_NN=0,U_truth_read=False,target_access=False,
   physics_records=self.physics_records,session_count=len(m._pair_sessions),means=means)
 def finish(self):
  result=super().finish();m=self.model
  if len(m._prop_records)!=2220 or self.physics_records!=2220 or m._prop_generated!=142080 or m._prop_slots!=284160 or result['L_NN']!=1704960 or result['Ag_packets'] or result['extra_U_NN'] or result['head_fit_updates'] or result['U_truth_read']:raise ValueError('Fixed E20+10/fullsource/defaulthead budget differs')
  with gzip.open(self.out/'prop_contrast_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
   for r in m._prop_records:f.write(json.dumps(r,allow_nan=False)+chr(10))
  result.update(new_epochs=10,parent_epochs=20,cumulative_epochs=30,new_updates=2220,parent_updates=4440,cumulative_updates=6660,
   L_As_total=142080,L_As_forward_NN=284160,prop_contrast_summary=self.prop_summary(),quality_entropy_summary=self.prop_summary(),TX_RX_scene_state_lock=joint(m._prop_coverage),source_RX_holdout=None,
   L_As_note='64 native process outputs per step;second64 exact IQ copies,all128 As CE slots supervised',parent_history_difference=self.c['parent_history_difference'])
  result['native_NN']=sum(result[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'))
  d.write(self.out/'result.json',result);return result

def train(c):
 o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
