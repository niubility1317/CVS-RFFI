"""Keep native L/U streams; bind a source-only shared finite nominal As pool."""
from pathlib import Path
import sys,time,gzip,json
import torch
from . import design as d,model,pool
from .source import parent_payload,segment_train
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original

class Observer(Original):
 def __init__(self,c):
  self.c=d.validate(c);self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c)
 def data(self,native,ctx,out):
  super().data(native,ctx,out);self.ctx=ctx
 def initialize(self,m,c,out):
  super().initialize(m,c,out)
  trace=Path(d.old_config()['output_root'])/'training_sequence.jsonl.gz'
  started=time.perf_counter()
  generated=pool.prepare(self.ctx,Path(c['quality_pool']['root']),trace,
   seed=c['quality_pool']['allocation_seed'],updates=d.UPDATES,steps=d.STEPS)
  model.bind_pool(m,c,Path(c['quality_pool']['root']))
  d.write(out/'quality_pool_initialization.json',dict(summary=generated,load_or_prepare_seconds=time.perf_counter()-started,
   source_trace=str(trace),fixed_before_first_update=True,V_read=False,target_read=False,
   allocation_seed=c['quality_pool']['allocation_seed'],method=c['method'],shared_pool=True,
   ordinary_raw_U_loader_unchanged=True,no_extra_neural_forward=True))
  d.write(out/'actual_channel.json',m._full_channel.audit())
  modules=[(name,type(v).__name__) for name,v in m.named_modules() if isinstance(v,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.GroupNorm,torch.nn.LayerNorm))]
  if any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in m.modules()):raise ValueError('BatchNorm coupling not permitted in the claimed fixed per-packet risk experiment')
  d.write(out/'quality_training_actual.json',dict(config=c,normalization_modules=modules,active_BatchNorm=0,
   SupCon='unchanged ORD128 only; no satellite cross-sample term',
   satellite_CE_coefficient=.25,satellite_KL_coefficient=.25,
   satellite_KL='original unweighted per-observation sameIQ RDrop',extra_NN=0,
   modules={k:str(v.__file__) for k,v in sys.modules.items() if k in ('SSDG.train_ssdg','leo_practical.channel','leo_practical.residual','DataAugmentation')}))
 def finish(self):
  result=super().finish();m=self.model
  if (result['optimizer_steps']!=2220 or result['L_NN']!=1704960 or result['L_As']!=142080
   or result['extra_U_NN'] or result['U_As'] or result['U_truth_read'] or result['head_fit_updates'] or result['Ag_packets']):raise ValueError('SixL/nativeU/true-label/default-head budget differs')
  actual=d.read(self.out/'actual_native_args.json');args=actual.get('args',actual)
  if args.get('pseudo_temporal_mode')!='batch_neighbor' or not args.get('pseudo_temporal_gate'):raise ValueError('Original U rules changed')
  with gzip.open(self.out/'quality_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
   for r in m._quality_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  summary=m._full_channel.summary()
  result.update(quality_pool=summary,quality_method=self.c['method'],
   U_steps=d.UPDATES,new_epochs=10,parent_epochs=20,cumulative_epochs=30,new_updates=2220,
   parent_updates=4440,cumulative_updates=6660,extra_neural_forwards=0,extra_deployment_parameters=0,
   own_EMA_parent_restored=True,TX_RX_scene_state_lock=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(m._full_coverage.items())])
  result['native_NN']=sum(result[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'))
  if result['native_NN']!=3162962:raise ValueError('Quality estimation must add no neural forward')
  d.write(self.out/'quality_pool_final.json',summary);d.write(self.out/'result.json',result)
  return result

def train(c):
 observer=Observer(c);segment_train(c,design=d,model_api=model,observer=observer);return observer.finish()
