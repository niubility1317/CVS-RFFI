"""Shared default-head prediction lifecycle. No feature extraction or fitting."""
import time
from pathlib import Path

class DefaultPipeline:
 def __init__(self,design,proof,build,baseline,observation=None):
  self.d=design;self.proof=proof;self.build=build;self.baseline=baseline
  if observation is None:
   from experiments.cvs_frequency_context50.predict import observation_metadata
   observation=observation_metadata
  self.observation=observation
 def checkpoint(self,c):
  source=c.get('source_config') or c
  return str(Path(source['output_root'])/'final_ssdg.pth')
 def frozen(self,c):
  return dict(status='DEFAULT_MODEL_FROZEN',config=c,readouts=list(self.d.READOUTS),checkpoint=self.checkpoint(c),head_type='ResidualPhysicalHead',head_shape=[6,160],head_scale=30.,original_default_W_retained=True,joint_source_training=True,separate_head_fit=False,separate_head_updates=0,backbone_updates_after_freeze=0,all_model_parameters_frozen=True,query_fit=False,truth_read=False)
 def provenance(self,c,source):
  reused=c.get('source_config') is not None
  return dict(status='VERIFIED',config=c,checkpoint=self.checkpoint(c),source_root=str(source),source_roles='EXACT_MATCH',ancestors=[],own_final50=True,original_default_head=True,separate_head_fit=False,model_frozen_before_target_IQ=True,readouts=list(self.d.READOUTS),truth_read=False,query_fit=False,Phase2_claim=False,baseline_original_reused=reused,baseline_prediction_root=str(Path(self.d.PROJECT)/'runs'/self.d.BASELINE_EVAL_RUN/c['row_id']/'prediction') if reused else None)
 def complete(self,c):
  return dict(status='PREDICTIONS_COMPLETE',config=c,count=self.d.QUERY_COUNT,views=list(self.d.VIEWS),readouts=list(self.d.READOUTS),truth_read=False,query_fit=False,model_frozen_before_target_IQ=True,separate_head_fit=False,separate_head_updates=0,baseline_original_reused=c.get('source_config') is not None)
 def decisions(self,value):
  if value.shape!=(self.d.QUERY_COUNT,) or value.dtype.kind not in 'iu' or value.min()<0 or value.max()>=6:raise ValueError('Invalid default six-class decisions')
 def preflight(self,row):
  import numpy as np
  d=self.d;c=d.config(next(r for r in d.rows() if r['row_id']==row));source=self.proof(c);output=d.BASE/row/'prediction';done=d.read(output/'complete.json')
  if any(done.get(k)!=v for k,v in self.complete(c).items()):raise ValueError('Truth closed: incomplete default prediction')
  if d.read(output/'provenance.json')!=self.provenance(c,source) or d.read(output/'default_frozen.json')!=self.frozen(c):raise ValueError('Truth closed: default freeze/provenance differs')
  ids,unused=self.observation()
  with np.load(output/'predictions.npz',allow_pickle=False) as saved:
   if set(saved.files)!={'ids','ORIGINAL_HEAD'} or not np.array_equal(saved['ids'],ids):raise ValueError('Truth closed: default-only predictions or IDs differ')
   self.decisions(saved['ORIGINAL_HEAD'])
   if c.get('source_config') is not None and not np.array_equal(saved['ORIGINAL_HEAD'],self.baseline(c,ids)):raise ValueError('Truth closed: reused baseline decisions differ')
  return c,ids
 def predict(self,row):
  import numpy as np,torch
  from experiments.cvs_default_head.predict import logits as default_logits
  from experiments.cvs_equivariant_identity.precision import numerical_context
  from experiments.cvs_report_ssl.evaluate import iq_tensor
  d=self.d;c=d.config(next(r for r in d.rows() if r['row_id']==row));source=self.proof(c);output=d.BASE/row/'prediction'
  if output.exists():raise FileExistsError('Existing prediction must be reconciled, not overwritten')
  output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
  if c.get('source_config') is not None:
   ids,unused=self.observation();values=self.baseline(c,ids);self.decisions(values)
   d.write(output/'default_frozen.json',self.frozen(c));d.write(output/'provenance.json',self.provenance(c,source))
   np.savez_compressed(output/'predictions.npz',ids=ids,ORIGINAL_HEAD=values)
   d.write(output/'complete.json',dict(self.complete(c),inference_seconds=0.,wall_seconds=time.perf_counter()-started));self.preflight(row);return
  with numerical_context(d.FULL_FP32_POLICY):
   model=self.build(c,'cuda:0');before={k:v.clone() for k,v in model.state_dict().items()}
   smoke=default_logits(model,torch.zeros(2,2,256,device='cuda:0'))
   if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Own default checkpoint no-query smoke failed')
   d.write(output/'default_frozen.json',self.frozen(c));d.write(output/'provenance.json',self.provenance(c,source))
   ids,observations=self.observation();array=np.load(observations['clean_ref'],mmap_mode='r',allow_pickle=False)
   if array.shape!=(d.QUERY_COUNT,2,256) or array.dtype.name!='float32':raise ValueError('Fixed clean observation differs')
   values=np.empty(d.QUERY_COUNT,dtype=np.int64);infer_started=time.perf_counter()
   for begin in range(0,d.QUERY_COUNT,256):
    x=iq_tensor(np.ascontiguousarray(array[begin:begin+256]),'cuda:0');score=default_logits(model,x)
    values[begin:begin+len(x)]=score.argmax(1).cpu().tolist()
   torch.cuda.synchronize();self.decisions(values)
   if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Prediction changed frozen model state')
   np.savez_compressed(output/'predictions.npz',ids=ids,ORIGINAL_HEAD=values)
   d.write(output/'complete.json',dict(self.complete(c),inference_seconds=time.perf_counter()-infer_started,wall_seconds=time.perf_counter()-started))
  self.preflight(row)
