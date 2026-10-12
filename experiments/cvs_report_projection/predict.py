import time
from pathlib import Path
import torch
from . import design as d
from .statistics import classify
from experiments.cvs_scratch30_baseline.predict import own_model,source_proof as original_source_proof
from experiments.cvs_report_ssl.evaluate import physical_ids,iq_tensor
from experiments.cvs_equivariant_identity.precision import numerical_context
def states(seed):
 original_source_proof(d.parent(seed));root=d.BASE/('s'+str(seed));v=d.read(root/'source_frozen.json')
 if v['status']!='SOURCE_FROZEN' or v['seed']!=seed or v['methods']!=list(d.METHODS) or v['fit_L']!=6300 or v['parent_config']!=d.parent(seed) or v['gradient_updates'] or v['parent_epochs']!=30 or v['target_access']:raise ValueError('Projectionnotfrozen')
 result={}
 for name in d.METHODS:
  c=d.config(next(r for r in d.rows() if r['model_seed']==seed and r['arm']==name));p=Path(c['output']);ck=torch.load(p/'model.pth',map_location='cpu',weights_only=False)
  if ck['config']!=c or ck['fit_ids']!=v['fit_ids'] or ck['parent_config']!=d.parent(seed) or ck['epoch']!=30 or ck['new_gradient_updates'] or ck['ancestors']!=[d.parent(seed)['run_id']] or ck['target_access']:raise ValueError('Fittedprojectionstate provenance differs')
  state=ck['state']
  if set(state)!=set(('A','bias','weight','classifier','source_gradient_updates','phase2_eligibility')) or state['A'].shape!=(160,160) or state['bias'].shape!=(160,) or state['weight'].shape!=(6,160) or any(not torch.isfinite(state[k]).all() for k in ('A','bias','weight')):raise ValueError('Ordinarycos projection checkpoint shape/value differs')
  result[name]=state
 return result
def run(seed,device='cuda:0'):
 import numpy as np
 ss=states(seed);ids=physical_ids();out=d.BASE/('s'+str(seed))/'prediction';out.mkdir(exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(d.parent(seed),device);before={k:v.clone() for k,v in model.state_dict().items()};ss={n:{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in state.items()} for n,state in ss.items()};a=np.load(Path(d.CAPSULE)/'clean.npy',mmap_mode='r',allow_pickle=False)
  if a.shape!=(168000,2,256) or a.dtype.name!='float32':raise ValueError('Cleanobservations differ')
  preds={n:np.empty(d.COUNT,dtype=np.int64) for n in d.METHODS}
  with torch.no_grad():
   for begin in range(0,d.COUNT,256):
    x=iq_tensor(np.ascontiguousarray(a[begin:begin+256]),device);outmodel=model.forward_identity_only(x);h=torch.nn.functional.normalize(outmodel['id_feat_joint'],dim=1,eps=1e-4)
    for name,state in ss.items():
     logits=classify(h,state)
     if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise FloatingPointError('Cleanprojectioncos logits')
     if name=='NATIVE' and not torch.allclose(logits,outmodel['tx_logits'],atol=1e-5,rtol=1e-5):raise ValueError('Nativeprojectioncontrol changedbaselineformula')
     decision=outmodel['tx_logits'] if name=='NATIVE' else logits
     preds[name][begin:begin+len(x)]=decision.argmax(1).cpu().tolist()
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Querychangedfrozenparent')
  np.savez_compressed(out/'predictions.npz',ids=ids,**preds);d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',seed=seed,methods=list(d.METHODS),views=['clean'],count=168000,truth_read=False,query_fit=False,wall_seconds=time.perf_counter()-started))
 print('PREDICTIONS_COMPLETE '+str(seed),flush=True)
def preflight(seed):
 import numpy as np
 states(seed);ids=physical_ids();p=d.BASE/('s'+str(seed))/'prediction';v=d.read(p/'complete.json')
 if v['status']!='PREDICTIONS_COMPLETE' or v['seed']!=seed or v['methods']!=list(d.METHODS) or v['views']!=['clean'] or v['count']!=168000 or v['truth_read'] or v['query_fit']:raise ValueError('Truthclosed:projectionprediction incomplete')
 with np.load(p/'predictions.npz',allow_pickle=False) as pr:
  if set(pr.files)!=set(d.METHODS)|{'ids'} or not np.array_equal(ids,pr['ids']):raise ValueError('CleanphysicalIDs differ')
  for name in d.METHODS:
   a=pr[name]
   if a.shape!=(168000,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalidallclass decisions')
 return ids
