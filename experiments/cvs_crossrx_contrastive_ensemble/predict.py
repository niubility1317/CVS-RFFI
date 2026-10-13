import time
from pathlib import Path
from . import design as d
def input_proof(c):
 d.validate(c)
 from experiments.cvs_crossrx_contrastive_clean.predict import source_proof as parent_proof
 parent=parent_proof(d.parent_config(c));row=next(r for r in d.rows() if r['row_id']==c['row_id'])
 from .members import source_proof
 members=[d.member_config(r) for r in d.member_rows() if r['row_id']==row['row_id']]
 for mc in members:source_proof(mc)
 return parent,members
def own_model(c,device):
 input_proof(c)
 from experiments.cvs_crossrx_contrastive_clean.predict import own_model as parent_model
 from .members import own_model as member_model
 from experiments.cvs_ensemble_final30.model import ProbabilityEnsemble
 models=[parent_model(d.parent_config(c),'FINAL',device)]
 for r in d.member_rows():
  if r['row_id']==c['row_id']:models.append(member_model(d.member_config(r),'FINAL',device))
 return ProbabilityEnsemble(models).to(device).eval().requires_grad_(False)
def predict(row):
 import numpy as np,torch
 from experiments.cvs_report_ssl.evaluate import physical_ids,iq_tensor
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=d.config(next(r for r in d.rows() if r['row_id']==row));input_proof(c);out=d.BASE/row/'prediction'
 if out.exists():raise FileExistsError('Existing ensemble prediction; reconcile')
 torch.set_num_threads(2);start=time.perf_counter();ids=physical_ids();out.mkdir(exist_ok=False)
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(c,'cuda:0');before={k:v.clone() for k,v in model.state_dict().items()}
  a=np.load(Path(d.CAPSULE)/'clean.npy',mmap_mode='r',allow_pickle=False)
  if a.shape!=(d.COUNT,2,256) or a.dtype.name!='float32':raise ValueError('Clean observations differ')
  pr=np.empty(d.COUNT,dtype=np.int64)
  with torch.no_grad():
   smoke=model(torch.zeros(2,2,256,device='cuda:0'))
   if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real ensemble noquery smoke')
   for begin in range(0,d.COUNT,256):
    x=iq_tensor(np.ascontiguousarray(a[begin:begin+256]),'cuda:0');z=model(x)
    if z.shape!=(len(x),6) or not torch.isfinite(z).all():raise FloatingPointError('Frozen ordinarycos member ensemble')
    pr[begin:begin+len(x)]=z.argmax(1).cpu().tolist()
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Query changed frozen members')
  np.savez_compressed(out/'predictions.npz',ids=ids,ENSEMBLE=pr)
  d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=d.COUNT,views=['clean'],readouts=['ENSEMBLE'],truth_read=False,query_fit=False,wall_seconds=time.perf_counter()-start))
def preflight(row):
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import physical_ids
 c=d.config(next(r for r in d.rows() if r['row_id']==row));input_proof(c);out=d.BASE/row/'prediction';v=d.read(out/'complete.json');ids=physical_ids()
 if v['status']!='PREDICTIONS_COMPLETE' or v['config']!=c or v['count']!=d.COUNT or v['views']!=['clean'] or v['readouts']!=['ENSEMBLE'] or v['truth_read'] or v['query_fit']:raise ValueError('Truthclosed: incomplete ensemble predictions')
 with np.load(out/'predictions.npz',allow_pickle=False) as p:
  if set(p.files)!={'ids','ENSEMBLE'} or not np.array_equal(ids,p['ids']):raise ValueError('Fixed ensemble IDs differ')
  a=p['ENSEMBLE']
  if a.shape!=(d.COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid independent allclass predictions')
 return c,ids
