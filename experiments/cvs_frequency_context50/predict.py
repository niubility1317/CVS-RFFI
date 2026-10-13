import json,os,sys,time
from pathlib import Path
from . import design as d
COUNT=d.QUERY_COUNT

def source_proof(c):
 d.validate(c);p=Path(c['output_root']);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');init=d.read(p/'initialization.json');frozen=d.read(p/'source_frozen.json');args=d.read(p/'resolved_native_args.json')
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=c['epochs'] or done['logged_steps']!=c['updates'] or done['optimizer_steps']!=c['updates'] or done['target_access'] or done['target_evaluated'] or done['checkpoint_sources']:raise ValueError('Source budget or role differs')
 if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['seed']!=c['model_seed'] or init['source_roles']!='EXACT_MATCH':raise ValueError('Scratch provenance differs')
 if any(resolved.get(k)!=v for k,v in c.items()) or resolved['identity_input_policy']['explicit_CFO_identity_input'] or resolved['actual_frequency_context']!=dict(c['frequency_context'],head='unchanged cosine6x160',old_pseudo_gating=True,EMA=True):raise ValueError('Resolved feature recipe differs')
 if not args['from_scratch'] or args['baseline_ckpt'] or args['teacher_ckpt'] or args['label_epochs']!=32 or args['pseudo_epochs']!=18 or any(args.get(k)!=v for k,v in d.LEGACY_FIELDS.items()):raise ValueError('Historical pseudo/EMA/cos execution differs')
 if frozen!=dict(status='SOURCE_FROZEN',config=c,epoch=c['epochs'],updates=c['updates'],checkpoint=str(p/'final_ssdg.pth'),selection='final_only',target_access=False):raise ValueError('Final source freeze differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p

def own_model(c,device):
 import torch
 from types import SimpleNamespace
 from .model import installed
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=c['epochs'] or ck['a1_ema_successful_updates']!=c['updates'] or ck['run_id']!=d.RUN or ck['candidate_id']!=c['row_id'] or ck['checkpoint_selection']!='final_only' or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or not ck['args']['from_scratch']:raise ValueError('Own final checkpoint differs')
 with installed(c) as n:model=n.build_baseline_model(SimpleNamespace(**ck['baseline_args']),torch.device(device))
 model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False);return model

def observation_metadata():
 from experiments.cvs_report_ssl.evaluate import physical_ids
 ids=physical_ids();m=d.read(__import__('pathlib').Path(d.CAPSULE)/'manifest.json')
 if m.get('status')!='VALIDATED_ONCE' or m.get('count')!=COUNT or not m.get('source_target_disjoint') or not m.get('unique_ids'):raise ValueError('Existing clean capsule handles differ')
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
  with torch.no_grad():z=model.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Own final50 no-query smoke')
  ids,m=observation_metadata();out.mkdir(parents=True,exist_ok=False);values={};times={}
  d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(Path(c['output_root'])/'final_ssdg.pth'),own_final50=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'))
  with torch.no_grad():
   for view in d.VIEWS:
    a=np.load(m['clean_ref'] if view=='clean' else d.VIEWS_ROOT/(view+'.npy'),mmap_mode='r',allow_pickle=False)
    if a.shape!=(COUNT,2,256) or a.dtype.name!='float32':raise ValueError('Fixed observation array differs')
    ts=time.perf_counter();p=np.empty(COUNT,dtype=np.int64)
    for begin in range(0,COUNT,256):
     x=iq_tensor(np.ascontiguousarray(a[begin:begin+256]),'cuda:0');logits=model.forward_identity_only(x)['tx_logits']
     if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise FloatingPointError('Seven-view classifier')
     p[begin:begin+len(x)]=logits.argmax(1).cpu().tolist()
    torch.cuda.synchronize();values[view]=p;times[view]=time.perf_counter()-ts;print('PREDICTION_VIEW '+json.dumps(dict(row_id=row,view=view,count=COUNT,truth_read=False)),flush=True)
  np.savez_compressed(out/'predictions.npz',ids=ids,**values)
  d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,inference_seconds=times,wall_seconds=time.perf_counter()-start))
def preflight(row):
 import numpy as np
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source_proof(c);out=d.BASE/row/'prediction';done=d.read(out/'complete.json');proof=d.read(out/'provenance.json');ids,m=observation_metadata()
 if done['status']!='PREDICTIONS_COMPLETE' or done['config']!=c or done['count']!=COUNT or done['views']!=list(d.VIEWS) or done['truth_read'] or done['query_fit'] or proof!=dict(status='VERIFIED',config=c,checkpoint=str(Path(c['output_root'])/'final_ssdg.pth'),own_final50=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'):raise ValueError('Truth closed: incomplete prediction/provenance')
 with np.load(out/'predictions.npz',allow_pickle=False) as p:
  if set(p.files)!=set(d.VIEWS)|{'ids'} or not np.array_equal(ids,p['ids']):raise ValueError('Fixed predictions/IDs differ')
  for view in d.VIEWS:
   a=p[view]
   if a.shape!=(COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid six-class decisions')
 return c,ids
