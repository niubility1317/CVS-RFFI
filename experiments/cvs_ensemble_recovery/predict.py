import json,os,sys,time
from pathlib import Path
from . import design as d
COUNT=168000
def source_proof(c):
 d.validate(c)
 from .source import input_proof
 from experiments.cvs_phase1_stack.recover import verify_contract
 original,contract=input_proof(c);out=Path(c['output']);done=d.read(out/'completion.json');frozen=d.read(out/'source_frozen.json')
 if done!=dict(status='SOURCE_REUSED_FROZEN',config=c,source_roles='EXACT_MATCH',new_model_updates=0,target_access=False):raise ValueError('Recovery source reuse differs')
 if frozen!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(original/'identity.pth'),selected_by='originalfixedfinal30sourcequalifiedrecipe',target_access=False):raise ValueError('Recovery freeze differs')
 verify_contract(d.read(out/'source_contract.json'),d.read(d.SOURCE));return out
def own_model(c,device):
 source_proof(c)
 from experiments.cvs_ensemble_final30.predict import own_model as original_model
 return original_model(c['source_config'],device,repair_reference_copy=True)

def observation_metadata():
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import physical_ids
 ids=physical_ids();m=d.read(d.VIEWS_ROOT/'manifest.json')
 expected=dict(status='VALIDATED_ONCE',count=COUNT,scenes=list(d.SCENES),classes=d.CLASSES,truth_read=False,channel='residual/post_sync/noeq',source_capsule=d.CAPSULE,clean_ref=d.CAPSULE+'/clean.npy',rows_share_observations=True)
 if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Existing seven-view observation handles differ')
 with np.load(d.VIEWS_ROOT/'index.npz',allow_pickle=False) as ix:
  if not np.array_equal(ids,ix['ids']):raise ValueError('Existing sixscene physical index differs')
 return ids,m
def predict(row):
 import numpy as np,torch
 from experiments.cvs_report_ssl.evaluate import iq_tensor
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source_proof(c);out=d.BASE/row/'prediction'
 if out.exists():raise FileExistsError('Existing prediction; reconcile rather than overwrite')
 torch.set_num_threads(2);start=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(c,'cuda:0')
  with torch.no_grad():z=model(torch.zeros(2,2,256,device='cuda:0'))
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Own final30 no-query smoke')
  ids,m=observation_metadata();out.mkdir(parents=True,exist_ok=False);values={};times={}
  d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(Path(c['source_output'])/'identity.pth'),own_final30=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'))
  with torch.no_grad():
   for view in d.VIEWS:
    a=np.load(m['clean_ref'] if view=='clean' else d.VIEWS_ROOT/(view+'.npy'),mmap_mode='r',allow_pickle=False)
    if a.shape!=(COUNT,2,256) or a.dtype.name!='float32':raise ValueError('Fixed observation array differs')
    ts=time.perf_counter();p=np.empty(COUNT,dtype=np.int64)
    for begin in range(0,COUNT,256):
     x=iq_tensor(np.ascontiguousarray(a[begin:begin+256]),'cuda:0');logits=model(x)
     if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise FloatingPointError('Seven-view classifier')
     p[begin:begin+len(x)]=logits.argmax(1).cpu().tolist()
    torch.cuda.synchronize();values[view]=p;times[view]=time.perf_counter()-ts;print('PREDICTION_VIEW '+json.dumps(dict(row_id=row,view=view,count=COUNT,truth_read=False)),flush=True)
  np.savez_compressed(out/'predictions.npz',ids=ids,**values)
  d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,inference_seconds=times,wall_seconds=time.perf_counter()-start))
def preflight(row):
 import numpy as np
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source_proof(c);out=d.BASE/row/'prediction';done=d.read(out/'complete.json');proof=d.read(out/'provenance.json');ids,m=observation_metadata()
 if done['status']!='PREDICTIONS_COMPLETE' or done['config']!=c or done['count']!=COUNT or done['views']!=list(d.VIEWS) or done['truth_read'] or done['query_fit'] or proof!=dict(status='VERIFIED',config=c,checkpoint=str(Path(c['source_output'])/'identity.pth'),own_final30=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'):raise ValueError('Truth closed: incomplete prediction/provenance')
 with np.load(out/'predictions.npz',allow_pickle=False) as p:
  if set(p.files)!=set(d.VIEWS)|{'ids'} or not np.array_equal(ids,p['ids']):raise ValueError('Fixed predictions/IDs differ')
  for view in d.VIEWS:
   a=p[view]
   if a.shape!=(COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid six-class decisions')
 return c,ids
