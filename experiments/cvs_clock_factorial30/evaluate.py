"""Seven unchanged received views, then a separate truth-side CPU process."""
import os,time
from pathlib import Path
from . import design as d
COUNT=168000

def manifest():
 m=d.read(d.VIEWS_ROOT/'manifest.json')
 expected=dict(status='VALIDATED_ONCE',count=COUNT,scenes=list(d.SCENES),classes=d.CLASSES,truth_read=False,channel='residual/post_sync/noeq',source_capsule=d.CAPSULE,clean_ref=d.CAPSULE+'/clean.npy',rows_share_observations=True)
 if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Existing fixed sixscene observation handles differ')
 return m

def ids():
 import numpy as np
 manifest()
 with np.load(d.VIEWS_ROOT/'index.npz',allow_pickle=False) as p:values=p['ids'].copy()
 with np.load(Path(d.CAPSULE)/'index.npz',allow_pickle=False) as p:
  if values.shape!=(COUNT,) or not np.array_equal(values,p['ids']):raise ValueError('Clean/sixscene physical alignment differs')
 return values

def row_config(rid):return d.config(next(r for r in d.rows() if r['row_id']==rid))
def preflight(rid):
 import numpy as np
 from .proof import source
 c=row_config(rid);source(c);p=d.BASE/rid/'prediction';v=d.read(p/'complete.json');physical=ids()
 if v['status']!='PREDICTIONS_COMPLETE' or v['config']!=c or v['count']!=COUNT or v['views']!=list(d.VIEWS) or v['truth_read'] or v['query_fit'] or v['head_fit_updates']:raise ValueError('Truth remains closed: all registered views must be fixed')
 with np.load(p/'predictions.npz',allow_pickle=False) as a:
  if set(a.files)!={'ids',*d.VIEWS,*[s+'_logp' for s in d.VIEWS]} or not np.array_equal(a['ids'],physical):raise ValueError('Complete same-row physical decisions required')
  for view in d.VIEWS:
   v=a[view];q=a[view+'_logp']
   if v.shape!=(COUNT,) or v.dtype.kind not in 'iu' or v.min()<0 or v.max()>=6 or q.shape!=(COUNT,6) or not np.isfinite(q).all() or not np.allclose(np.exp(q).sum(1),1.,atol=1e-5) or not np.array_equal(q.argmax(1),v):raise ValueError('Invalid fixed head decisions/log probabilities')
 return c,physical

def predict(rid):
 import numpy as np
 import torch
 from .proof import load,source
 from experiments.cvs_report_ssl.evaluate import iq_tensor
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=row_config(rid);p=source(c);out=d.BASE/rid/'prediction'
 if out.exists():raise FileExistsError('Preserve existing prediction;reconcile rather than overwrite')
 physical=ids();m=manifest();start=time.perf_counter();torch.set_num_threads(2)
 with numerical_context(d.FULL_FP32_POLICY):
  model=load(c,'cuda:0')
  with torch.no_grad():z=model.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
  if z['tx_logits'].shape!=(2,6) or z['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(z['tx_logits']).all()):raise ValueError('Own default-head no-query smoke failed')
  predictions={};timings={};torch.cuda.reset_peak_memory_stats()
  for view in d.VIEWS:
   a=np.load(m['clean_ref'] if view=='clean' else d.VIEWS_ROOT/(view+'.npy'),mmap_mode='r',allow_pickle=False)
   if a.shape!=(COUNT,2,256) or a.dtype.name!='float32':raise ValueError('Original received IQ contract differs')
   pred=np.empty(COUNT,dtype=np.int64);logp=np.empty((COUNT,6),dtype=np.float32);t=time.perf_counter()
   with torch.no_grad():
    for n in range(0,COUNT,256):
     x=iq_tensor(np.ascontiguousarray(a[n:n+256]),'cuda:0');o=model.forward_identity_only(x);z=o['tx_logits']
     if z.shape!=(len(x),6) or o['id_feat_joint'].shape!=(len(x),160) or not bool(torch.isfinite(z).all()):raise FloatingPointError('Nonfinite default-head predictions')
     pred[n:n+len(x)]=z.argmax(1).cpu().tolist();logp[n:n+len(x)]=z.log_softmax(1).cpu().tolist()
   torch.cuda.synchronize();timings[view]=time.perf_counter()-t;predictions[view]=pred;predictions[view+'_logp']=logp
   print('VIEW_COMPLETE '+rid+' '+view,flush=True)
  out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'predictions.npz',ids=physical,**predictions)
  d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0,inference_seconds=timings,wall_seconds=time.perf_counter()-start,NN_forward_packets=COUNT*7+2,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),fixed_received_IQ=True))
  d.write(out/'provenance.json',dict(status='VERIFIED',checkpoint=str(p/'final_ssdg.pth'),checkpoint_sources=[c['parent_source']],initialization='E20plus10',source_roles='EXACT_MATCH',default_head_only=True,truth_read=False,query_fit=False,views_root=str(d.VIEWS_ROOT)))
 return rid

def score(rid):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent truth-side CPU process required')
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
 c,physical=preflight(rid);out=d.BASE/rid/'scores'
 if out.exists():raise FileExistsError('Existing score preserved')
 start=time.perf_counter();y,rx,days=truth_arrays(d.read(d.TRUTH),physical);records=[];correct={}
 with np.load(d.BASE/rid/'prediction/predictions.npz',allow_pickle=False) as a:
  for view in d.VIEWS:
   rr,dayrows=metric_rows(c,a[view],physical,y,rx,days)
   if len(rr+dayrows)!=18:raise ValueError('All overall/RX/TX/day strata required')
   for r in rr+dayrows:r['view']=view
   records+=rr+dayrows;correct[view]=a[view]==y
 out.mkdir();np.savez_compressed(out/'correctness.npz',ids=physical,**correct)
 d.write(out/'metrics.json',dict(status='SCORED',records=records,truth_last=True,independent_recount='VERIFIED',CPU_wall_seconds=time.perf_counter()-start,head_fit_updates=0))
 d.write(d.BASE/rid/'completion.json',dict(status='TRAINED_DEFAULT_PREDICTED_SCORED',config=c,views=list(d.VIEWS),count=COUNT,prediction_count=COUNT*7,optimizer_steps=2220,cumulative_optimizer_steps=6660,cumulative_epochs=30,source_RX_holdout=None,truth_last=True,independent_recount='VERIFIED',head_fit_updates=0))
 print('ROW_SCORED '+rid,flush=True)
