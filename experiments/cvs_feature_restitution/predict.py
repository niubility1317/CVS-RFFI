"""Two predeclared readouts; no target fit, scoring imports or truth access."""
import json,time
import torch
from . import design as d
from .provenance import source_proof,own_model
from .probe import CosineProbe,proof as probe_proof
COUNT=168000

def predict(row):
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import iq_tensor,physical_ids
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source_proof(c);ck,pv=probe_proof(c)
 out=d.BASE/row/'prediction'
 if out.exists():raise FileExistsError('Existing predictions; reconcile rather than replace')
 torch.set_num_threads(2);start=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(c,'cuda:0');head=CosineProbe(scale=30.).to('cuda:0');head.weight.data.copy_(ck['weight'].to('cuda:0'));head.eval().requires_grad_(False)
  with torch.no_grad():
   x=torch.zeros(2,2,256,device='cuda:0');v=model.forward_identity_only(x)
   if not torch.allclose(v['tx_logits'],model(x),atol=1e-6,rtol=1e-5) or v['id_feat_joint'].shape!=(2,160):raise ValueError('Shared deployed embedding/native readout smoke differs')
  ids=physical_ids();native=None
  if c['baseline_reuse']:
   from experiments.cvs_feature_multiscale50.predict import preflight as old_preflight
   oldrow=c['baseline_reference']['row_id'];old_preflight(oldrow)
   with np.load(d.old.BASE/oldrow/'prediction/predictions.npz',allow_pickle=False) as original:
    if not np.array_equal(original['ids'],ids):raise ValueError('Historical baseline prediction IDs differ')
    native=original['clean'].copy()
  a=np.load(d.CAPSULE+'/clean.npy',mmap_mode='r',allow_pickle=False)
  if a.shape!=(COUNT,2,256) or a.dtype.name!='float32':raise ValueError('Validated clean observation array differs')
  out.mkdir(parents=True,exist_ok=False);values={r:np.empty(COUNT,dtype=np.int64) for r in d.READOUTS}
  if native is not None:values['NATIVE_HEAD']=native
  with torch.no_grad():
   for begin in range(0,COUNT,256):
    x=iq_tensor(np.ascontiguousarray(a[begin:begin+256]),'cuda:0');v=model.forward_identity_only(x);z=v['id_feat_joint'];refit=head(z)
    if z.shape!=(len(x),160) or refit.shape!=(len(x),6) or v['tx_logits'].shape!=(len(x),6) or not all(torch.isfinite(t).all() for t in (z,refit,v['tx_logits'])):raise FloatingPointError('Frozen decisions nonfinite')
    values['REFIT_COSINE'][begin:begin+len(x)]=refit.argmax(1).cpu().tolist()
    if native is None:values['NATIVE_HEAD'][begin:begin+len(x)]=v['tx_logits'].argmax(1).cpu().tolist()
  torch.cuda.synchronize();np.savez_compressed(out/'predictions.npz',ids=ids,**values)
  d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(source_proof(c)/'final_ssdg.pth'),baseline_native_reused=c['baseline_reuse'],historical_benchmark_exposed=True,refit_fit_role='L_s',backbone_frozen=True,truth_read=False,query_fit=False))
  d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=['clean'],readouts=list(d.READOUTS),truth_read=False,query_fit=False,wall_seconds=time.perf_counter()-start))
 print('PREDICTIONS_COMPLETE '+row,flush=True)

def preflight(row):
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import physical_ids
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source=source_proof(c);probe_proof(c);out=d.BASE/row/'prediction';done=d.read(out/'complete.json');p=d.read(out/'provenance.json')
 if done['status']!='PREDICTIONS_COMPLETE' or done['config']!=c or done['count']!=COUNT or done['views']!=['clean'] or done['readouts']!=list(d.READOUTS) or done['truth_read'] or done['query_fit']:raise ValueError('Truth remains closed: both readouts must be complete')
 expected=dict(status='VERIFIED',config=c,checkpoint=str(source/'final_ssdg.pth'),baseline_native_reused=c['baseline_reuse'],historical_benchmark_exposed=True,refit_fit_role='L_s',backbone_frozen=True,truth_read=False,query_fit=False)
 if p!=expected:raise ValueError('Prediction/source-only refit provenance differs')
 ids=physical_ids()
 with np.load(out/'predictions.npz',allow_pickle=False) as pred:
  if set(pred.files)!=set(d.READOUTS)|{'ids'} or not np.array_equal(ids,pred['ids']):raise ValueError('Every fixed readout/physical ID is required before truth')
  for readout in d.READOUTS:
   a=pred[readout]
   if a.shape!=(COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid independent decisions')
 return c,ids
