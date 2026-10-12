import json,os,sys,time
from pathlib import Path
from . import design as d
COUNT=168000
def baseline_proof(c):
 d.validate(c)
 from experiments.cvs_scratch30_baseline.predict import source_proof as original
 return original(d.baseline_config(c['model_seed']))

def source_proof(c,repair_reference_copy=False):
 d.validate(c);baseline_proof(c);source=Path(c['output']);done=d.read(source/'completion.json');frozen=d.read(source/'source_frozen.json')
 expected=dict(status='SOURCE_TRAINED',config=c,members=3,new_members=2,epochs_per_member=30,new_updates=13320,reused_updates=6660,total_updates=19980,ensemble_fit_updates=0,target_access=False,U_truth_read=False,allmembersfrozen=True,trainable_parameters=531075)
 if done!=expected or frozen!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(source/'identity.pth'),selection=c['checkpoint_selection'],target_access=False):raise ValueError('Ensemble final source freeze differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 reference=d.read(d.SOURCE);aggregate=d.read(source/'source_contract.json')
 if repair_reference_copy:
  # The r01 writer copied exactly the reference recipe into the aggregate.
  # Repair only that observed metadata defect from actual member evidence.
  if aggregate!=reference:raise ValueError('Recovery source differs from known reference-copy defect')
  from .contracts import verified_component_contract
  rr=next(r for r in d.rows() if r['row_id']==c['row_id'])
  aggregate=verified_component_contract([d.read(Path(d.member_config(rr,i)['output'])/'source_contract.json') for i in (1,2)],reference)
 verify_contract(aggregate,reference)
 row=next(r for r in d.rows() if r['row_id']==c['row_id'])
 for member in (1,2):
  mc=d.member_config(row,member);p=Path(mc['output']);v=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');sf=d.read(p/'source_frozen.json')
  if v['config']!=mc or v['status']!='SOURCE_TRAINED' or v['epoch']!=30 or v['updates']!=6660 or v['logged_steps']!=6660 or v['ancestors'] or v['target_access'] or v['U_truth_read'] or v['counts']['identity_updates']!=6660 or v['counts']['L_packets']!=852480 or v['counts']['U_packets'] or v['counts']['teacher_forwards'] or v['distinct_U']:raise ValueError('Own fresh ensemble member incomplete or illegal')
  if resolved['config']!=mc or resolved['checkpoint_loaded'] or resolved['ancestors'] or resolved['U_iterator'] or resolved['EMA'] or resolved['satellite_generator'] or resolved['actual_initialization_seed']!=mc['member_init_seed'] or resolved['identity_input_policy']['explicit_CFO_identity_input']:raise ValueError('Actual new member implementation differs')
  if sf!=dict(status='SOURCE_FROZEN',config=mc,checkpoint=str(p/'identity.pth'),epoch=30,updates=6660,selection='fixed_final_epoch30',target_access=False):raise ValueError('Own member final freeze differs')
  verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE))
 return source

def own_model(c,device,repair_reference_copy=False):
 import torch
 from .model import build_member,ProbabilityEnsemble
 from experiments.cvs_scratch30_baseline.predict import own_model as baseline_model
 source=source_proof(c,repair_reference_copy=repair_reference_copy);bundle=torch.load(source/'identity.pth',map_location='cpu',weights_only=False);row=next(r for r in d.rows() if r['row_id']==c['row_id'])
 members=[dict(member=0,config=d.baseline_config(c['model_seed']),checkpoint=d.baseline_config(c['model_seed'])['output']+'/identity.pth'),*[dict(member=i,config=d.member_config(row,i),checkpoint=d.member_config(row,i)['output']+'/identity.pth') for i in (1,2)]]
 if bundle!=dict(config=c,members=members,formula=c['ensemble']['formula'],epochs_per_member=30,new_updates=13320,reused_updates=6660,total_updates=19980,target_access=False,ensemble_fit_updates=0):raise ValueError('Fixed inference bundle differs')
 models=[baseline_model(d.baseline_config(c['model_seed']),device)]
 for member in (1,2):
  mc=d.member_config(row,member);ck=torch.load(Path(mc['output'])/'identity.pth',map_location='cpu',weights_only=False)
  if ck['config']!=mc or ck['epoch']!=30 or ck['updates']!=6660 or ck['ancestors'] or ck['target_access'] or ck['initialization']!='scratch' or ck['checkpoint_selection']!='fixed_final_epoch30':raise ValueError('Fresh member checkpoint provenance differs')
  m=build_member(mc,device);m.load_state_dict(ck['model'],strict=True);models.append(m)
 return ProbabilityEnsemble(models).to(device).eval().requires_grad_(False)

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
