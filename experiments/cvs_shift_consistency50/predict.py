import json,os,sys,time
from pathlib import Path
from . import design as d
COUNT=168000
def source_proof(c):
 d.validate(c);p=Path(c['output']);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');frozen=d.read(p/'source_frozen.json')
 from .source import validate_counts
 validate_counts(c,done['counts'],done['distinct_U'])
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=d.EPOCHS or done['updates']!=d.UPDATES or done['logged_steps']!=d.UPDATES or done['target_access'] or done['U_truth_read'] or done['V_forward'] or done['ancestors']:raise ValueError('Own E50 source provenance differs')
 if frozen!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'identity.pth'),epoch=d.EPOCHS,updates=d.UPDATES,selection='fixed_final_epoch50',target_access=False) or resolved['config']!=c or resolved['checkpoint_loaded'] or resolved['ancestors'] or resolved['EMA'] or resolved['satellite_generator'] or resolved['identity_input_policy']['explicit_CFO_identity_input'] or resolved['V_forward'] or resolved['source_performance_gate'] or resolved['U_iterator']!=c['source_U'] or resolved['actual_statistic_layers']!=(list(d.LEVELS) if c['shift_statistics']['enabled'] else []) or resolved['teacher']!=('owncurrentstudent twoevalweakviews' if c['source_U'] else 'none'):raise ValueError('Own E50 freeze/runtime differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p
def own_model(c,device):
 import torch
 from .model import build
 source=source_proof(c);ck=torch.load(source/'identity.pth',map_location='cpu',weights_only=False)
 if ck['config']!=c or ck['epoch']!=d.EPOCHS or ck['updates']!=d.UPDATES or ck['ancestors'] or ck['initialization']!='scratch' or ck['target_access'] or ck.get('U_truth_read') is not False or ck['checkpoint_selection']!='fixed_final_epoch50':raise ValueError('Own final30 checkpoint differs')
 model=build(c,device);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False);return model
def observation_metadata():
 from experiments.cvs_report_ssl.evaluate import physical_ids
 ids=physical_ids()
 if len(ids)!=COUNT:raise ValueError('Existing clean physical index differs')
 return ids,dict(clean_ref=d.CAPSULE+'/clean.npy')
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
  d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(Path(c['output'])/'identity.pth'),own_final30=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'))
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
 if done['status']!='PREDICTIONS_COMPLETE' or done['config']!=c or done['count']!=COUNT or done['views']!=list(d.VIEWS) or done['truth_read'] or done['query_fit'] or proof!=dict(status='VERIFIED',config=c,checkpoint=str(Path(c['output'])/'identity.pth'),own_final30=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'):raise ValueError('Truth closed: incomplete prediction/provenance')
 with np.load(out/'predictions.npz',allow_pickle=False) as p:
  if set(p.files)!=set(d.VIEWS)|{'ids'} or not np.array_equal(ids,p['ids']):raise ValueError('Fixed predictions/IDs differ')
  for view in d.VIEWS:
   a=p[view]
   if a.shape!=(COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid six-class decisions')
 return c,ids
