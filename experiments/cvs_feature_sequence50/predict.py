"""Default-head clean decisions only; no probe, fitting, or truth access."""
import time
import torch
from . import design as d
from .provenance import source_proof,own_model
COUNT=168000

def predict(row,design=d,proof=None,load=None):
 d=design;source_proof=proof or globals()["source_proof"];own_model=load or globals()["own_model"]
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import iq_tensor,physical_ids
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source=source_proof(c);out=d.BASE/row/'prediction'
 if out.exists():raise FileExistsError('Existing predictions; reconcile rather than overwrite')
 torch.set_num_threads(2);start=time.perf_counter();ids=physical_ids()
 if len(ids)!=COUNT:raise ValueError('Registered clean physical count differs')
 with numerical_context(d.FULL_FP32_POLICY):
  if c['baseline_reuse']:
   from experiments.cvs_feature_multiscale50.predict import preflight as original
   oldrow=c['baseline_reference']['row_id'];original(oldrow)
   with np.load(d.old.BASE/oldrow/'prediction/predictions.npz',allow_pickle=False) as p:
    if not np.array_equal(ids,p['ids']):raise ValueError('Historical default-head physical IDs differ')
    decisions=p['clean'].copy()
  else:
   model=own_model(c,'cuda:0')
   with torch.no_grad():smoke=model.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
   if smoke['tx_logits'].shape!=(2,6) or smoke['id_feat_joint'].shape!=(2,160) or not torch.isfinite(smoke['tx_logits']).all():raise ValueError('Default classifier no-query smoke failed')
   a=np.load(d.CAPSULE+'/clean.npy',mmap_mode='r',allow_pickle=False)
   if a.shape!=(COUNT,2,256) or a.dtype.name!='float32':raise ValueError('Validated clean observation shape differs')
   decisions=np.empty(COUNT,dtype=np.int64)
   with torch.no_grad():
    for begin in range(0,COUNT,256):
     x=iq_tensor(np.ascontiguousarray(a[begin:begin+256]),'cuda:0');v=model.forward_identity_only(x)
     if v['tx_logits'].shape!=(len(x),6) or v['id_feat_joint'].shape!=(len(x),160) or not torch.isfinite(v['tx_logits']).all() or not torch.isfinite(v['id_feat_joint']).all():raise FloatingPointError('Nonfinite default-head decisions')
     decisions[begin:begin+len(x)]=v['tx_logits'].argmax(1).cpu().tolist()
   torch.cuda.synchronize()
  if decisions.shape!=(COUNT,) or decisions.dtype.kind not in 'iu' or decisions.min()<0 or decisions.max()>=6:raise ValueError('Invalid default classifier decisions')
  out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'predictions.npz',ids=ids,NATIVE_HEAD=decisions)
  d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(source/'final_ssdg.pth'),baseline_native_reused=c['baseline_reuse'],default_classifier_only=True,separate_head_training=False,backbone_frozen=True,truth_read=False,query_fit=False,historical_benchmark_exposed=True))
  d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=['clean'],readouts=['NATIVE_HEAD'],truth_read=False,query_fit=False,head_fit_updates=0,wall_seconds=time.perf_counter()-start))
 print('PREDICTIONS_COMPLETE '+row,flush=True)

def preflight(row,design=d,proof=None):
 d=design;source_proof=proof or globals()["source_proof"]
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import physical_ids
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source=source_proof(c);out=d.BASE/row/'prediction';done=d.read(out/'complete.json');p=d.read(out/'provenance.json')
 if done['status']!='PREDICTIONS_COMPLETE' or done['config']!=c or done['count']!=COUNT or done['views']!=['clean'] or done['readouts']!=['NATIVE_HEAD'] or done['truth_read'] or done['query_fit'] or done['head_fit_updates']!=0:raise ValueError('Truth remains closed: complete default-head predictions required')
 expected=dict(status='VERIFIED',config=c,checkpoint=str(source/'final_ssdg.pth'),baseline_native_reused=c['baseline_reuse'],default_classifier_only=True,separate_head_training=False,backbone_frozen=True,truth_read=False,query_fit=False,historical_benchmark_exposed=True)
 if p!=expected:raise ValueError('Default-head prediction provenance differs')
 ids=physical_ids()
 with np.load(out/'predictions.npz',allow_pickle=False) as pred:
  if set(pred.files)!={'ids','NATIVE_HEAD'} or not np.array_equal(ids,pred['ids']):raise ValueError('Physical ID/readout mismatch')
  a=pred['NATIVE_HEAD']
  if a.shape!=(COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid independent default-head decisions')
 return c,ids
