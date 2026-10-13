import time
from pathlib import Path
from . import design as d
COUNT=168000
def source_proof(c):
 d.validate(c);p=Path(c['output']);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');frozen=d.read(p/'source_frozen.json');expect=710400 if c['source_U'] else 0
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=30 or done['updates']!=6660 or done['logged_steps']!=6660 or done['target_access'] or done['U_truth_read'] or done['ancestors'] or done['V_forward'] or done['source_performance_gate'] or done['counts']['L_packets']!=852480 or done['counts']['U_packets']!=expect or done['counts']['identity_updates']!=6660 or done['distinct_U']!=(56700 if c['source_U'] else 0) or done['readouts']!=list(d.readouts(c)):raise ValueError('Reportmechanism actual training provenance differs')
 if frozen!=dict(status='SOURCE_FROZEN',config=c,readouts=list(d.readouts(c)),epoch=30,updates=6660,selection='fixed_final_epoch30',target_access=False) or resolved['config']!=c or resolved['checkpoint_loaded'] or resolved['ancestors'] or resolved['target_access'] or resolved['U_truth_read'] or resolved['V_forward'] or resolved['source_performance_gate'] or resolved['identity_input_policy']['explicit_CFO_identity_input'] or resolved['classifier']!=c['classifier']:raise ValueError('Reportmechanism freeze differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE))
 if done['counts']['statistic_refreshes']!=30 or done['counts']['statistic_L_packets']!=189000 or resolved['drift_query_state'] or resolved['drift_fit_role']!='L_only':raise ValueError('Crossfitted statistic role/cost differs')
 return p
def own_model(c,name,device):
 import torch
 from .model import build
 p=source_proof(c)
 if name not in d.readouts(c):raise ValueError('Unregistered readout')
 ck=torch.load(p/(name+'.pth'),map_location='cpu',weights_only=False);contract=d.read(p/'source_contract.json')
 if ck['config']!=c or ck['readout']!=name or ck['epoch']!=30 or ck['updates']!=6660 or ck['ancestors'] or ck['initialization']!='scratch' or ck['target_access'] or ck['U_truth_read'] or len(ck['fit_ids'])!=6300 or sorted(ck['fit_ids'])!=sorted(contract['role_ids']['L_s']) or sorted(ck['fit_U_ids'])!=(sorted(contract['role_ids']['U_s']) if c['source_U'] else []):raise ValueError('Own final30 actualcheckpoint/data differs')
 model=build(c,device);model.load_state_dict(ck['model'],strict=True);return model.eval().requires_grad_(False)
def predict(row):
 import numpy as np,torch
 from experiments.cvs_report_ssl.evaluate import physical_ids,iq_tensor
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source_proof(c);out=d.BASE/row/'prediction'
 if out.exists():raise FileExistsError('Existingprediction; reconcilebefore retry')
 torch.set_num_threads(2);start=time.perf_counter();ids=physical_ids();out.mkdir(exist_ok=False);values={};times={}
 with numerical_context(d.FULL_FP32_POLICY):
  models={name:own_model(c,name,'cuda:0') for name in d.readouts(c)};states={name:{k:v.clone() for k,v in m.state_dict().items()} for name,m in models.items()};a=np.load(Path(d.CAPSULE)/'clean.npy',mmap_mode='r',allow_pickle=False)
  if a.shape!=(COUNT,2,256) or a.dtype.name!='float32':raise ValueError('Cleanobservations differ')
  with torch.no_grad():
   for name,model in models.items():
    smoke=model(torch.zeros(2,2,256,device='cuda:0'))
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Own realcheckpoint noquery smoke')
    t=time.perf_counter();pr=np.empty(COUNT,dtype=np.int64)
    for begin in range(0,COUNT,256):
     x=iq_tensor(np.ascontiguousarray(a[begin:begin+256]),'cuda:0');logits=model(x)
     if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise FloatingPointError('Clean ordinarycos decisions')
     pr[begin:begin+len(x)]=logits.argmax(1).cpu().tolist()
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in states[name].items()):raise ValueError('Query mutatedcheckpoint')
    values[name]=pr;times[name]=time.perf_counter()-t;print('PREDICTION_COMPLETE '+row+' '+name,flush=True)
  np.savez_compressed(out/'predictions.npz',ids=ids,**values);d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=['clean'],readouts=list(d.readouts(c)),truth_read=False,query_fit=False,inference_seconds=times,wall_seconds=time.perf_counter()-start))
def preflight(row):
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import physical_ids
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source_proof(c);out=d.BASE/row/'prediction';v=d.read(out/'complete.json');ids=physical_ids()
 if v['status']!='PREDICTIONS_COMPLETE' or v['config']!=c or v['count']!=COUNT or v['views']!=['clean'] or v['readouts']!=list(d.readouts(c)) or v['truth_read'] or v['query_fit']:raise ValueError('Truthclosed:allreadouts incomplete')
 with np.load(out/'predictions.npz',allow_pickle=False) as p:
  if set(p.files)!=set(d.readouts(c))|{'ids'} or not np.array_equal(ids,p['ids']):raise ValueError('Fixedclean IDs differ')
  for name in d.readouts(c):
   a=p[name]
   if a.shape!=(COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid allclasscos decisions')
 return c,ids
