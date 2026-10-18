"""Full actual traces and one As two-mask CE aggregation intervention."""
from pathlib import Path
import sys,time,gzip,json
from . import design as d,model
from .source import parent_payload,segment_train
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original
class Observer(Original):
 def __init__(self,c):
  self.c=d.validate(c);self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c)
 def initialize(self,m,c,out):
  super().initialize(m,c,out);started=time.perf_counter()
  model.bind_pool(m,c,Path(c['quality_pool']['root']))
  d.write(out/'marginal_initialization.json',dict(method=c['method'],as_marginalization=c['as_marginalization'],reader=m._full_channel.summary(),pool_load_seconds=time.perf_counter()-started,
   nominal_pool_reused=True,original_generation_order=True,new_channel_calls=0,fixed_before_first_update=True,V_read=False,target_read=False,extra_neural_forward=0))
  d.write(out/'actual_channel.json',m._full_channel.audit())
  d.write(out/'marginal_training_actual.json',dict(config=c,modules={k:str(v.__file__) for k,v in sys.modules.items() if k in ('SSDG.train_ssdg','leo_practical.channel','leo_practical.residual','DataAugmentation','experiments.cvs_quality_risk70.model','experiments.cvs_as_marginal73.model','experiments.cvs_as_marginal73.objective','experiments.cvs_as_marginal73.ordered_pool')},
   satellite_CE_coefficient=.25,satellite_KL_coefficient=.25,changed='Only As two-mask CE aggregation; no private mask, no extra loss',Dropout_KL_unchanged=True,ordinary_raw_U_objectives_unchanged=True,extra_NN=0,real_channel_regeneration=0))
 def finish(self):
  result=super().finish();m=self.model
  if result['optimizer_steps']!=2220 or result['L_NN']!=1704960 or result['L_As']!=142080 or result['extra_U_NN'] or result['U_As'] or result['U_truth_read'] or result['head_fit_updates'] or result['Ag_packets'] or len(m._marginal_records)!=2220 or len(m._quality_records)!=2220:raise ValueError('Original6L/nativeU/defaulthead and matched aggregation budget required')
  for name,records in [('marginal_steps.jsonl.gz',m._marginal_records),('ordered_steps.jsonl.gz',m._quality_records)]:
   with gzip.open(self.out/name,'xt',encoding='utf8',compresslevel=3) as stream:
    for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
  summary=m._full_channel.summary();d.write(self.out/'ordered_pool_final.json',summary)
  actual=d.read(self.out/'actual_native_args.json');args=actual.get('args',actual)
  if args.get('pseudo_temporal_mode')!='batch_neighbor' or not args.get('pseudo_temporal_gate'):raise ValueError('Original U rules changed')
  result.update(ordered_pool=summary,as_marginalization=self.c['as_marginalization'],marginal_record_count=len(m._marginal_records),
   U_steps=2220,new_epochs=10,parent_epochs=20,cumulative_epochs=30,new_updates=2220,parent_updates=4440,cumulative_updates=6660,extra_neural_forwards=0,
   extra_deployment_parameters=0,new_channel_calls=0,pool_generated_this_run=0,nominal_pool_reused=True,own_EMA_parent_restored=True,
   TX_RX_scene_state_lock=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(m._full_coverage.items())])
  result['native_NN']=sum(result[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'))
  if result['native_NN']!=3162962:raise ValueError('As aggregation must add zero NN')
  d.write(self.out/'result.json',result);return result
def train(c):
 observer=Observer(c);segment_train(c,design=d,model_api=model,observer=observer);return observer.finish()
