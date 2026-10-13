import json,os,sys,time
from pathlib import Path
from . import design as d
COUNT=168000

def source_proof(c):
 d.validate(c);p=Path(c['output_root']);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');frozen=d.read(p/'source_frozen.json');initial=d.read(p/'initialization.json');args=d.read(p/'actual_native_args.json')
 from .source import require_budget
 require_budget(done['logged_steps'],done['optimizer_steps'])
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=50 or done['target_access'] or done['target_evaluated'] or done['checkpoint_sources'] or done['ancestors'] or done['initialization']!='scratch' or done['label_epochs']!=33 or done['pseudo_epochs']!=17 or done['baseline_recipe']!='historicaloldgate+EMA+cos':raise ValueError('Own oldgate feature source provenance differs')
 if initial!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']):raise ValueError('Unknown source initialization')
 if frozen!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=50,selection='final_only',target_access=False) or resolved['config']!=c or resolved['identity_input_policy']['explicit_CFO_identity_input'] or done['identity_input_policy']['explicit_CFO_identity_input']:raise ValueError('Own oldgate source freeze/runtime differs')
 validate_recipe_args(args,c)
 expected=expected_feature_report(c)
 if done['feature_extractor']!=expected or resolved['actual_feature_extractor']!=expected:raise ValueError('Feature extractor/classifier differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p

def expected_feature_report(c):
 changed=(list(d.TIME_LAYERS) if c['feature_extractor']['time'] else [])+(list(d.PA_LAYERS) if c['feature_extractor']['PA'] else [])
 settings={n:c['feature_extractor']['time_kernel_dilation'] for n in changed if n in d.TIME_LAYERS}
 settings.update({n:[[k,rate] for k in (3,5,7)] for n,rate in zip(d.PA_LAYERS,(1,2,4)) if n in changed})
 return dict(changed_layers=changed,classifier_type='ResidualPhysicalHead',classifier_weight_shape=[6,160],classifier_scale=30.,output_embedding=160,structural_classifier_changes=False,multiscale=settings)

def validate_recipe_args(args,c):
 # Native resolved_config is a grouped report; final payload.args is flat.
 if 'args' in args:args=args['args']
 expected=d.make_args(c,'cpu')
 keys=('run_id','candidate_id','epochs','label_epochs','pseudo_epochs','use_unlabeled','use_ema_teacher','ema_decay','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_temporal_window','pseudo_temporal_min_conf','pseudo_strong_agreement','pseudo_domain_gate','pseudo_threshold_mode','pseudo_quantile','label_smoothing','lambda_u','lambda_ent','phase1_lr_schedule','phase1_lr_min','use_concat_sat_channel_aug','use_sat_consistency','lambda_sat_cls','lambda_sat_cons','sat_training_mode','from_scratch','baseline_ckpt','teacher_ckpt','output_dir')
 if any(args.get(k)!=getattr(expected,k) for k in keys):raise ValueError('Historical oldgate EMA cosine recipe differs')
 return True

def own_model(c,device):
 import torch
 from .model import build
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['candidate_id']!=c['row_id'] or ck['run_id']!=c['run_id'] or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=11100:raise ValueError('Own fixedfinal50 checkpoint differs')
 validate_recipe_args(ck['args'],c);model=build(c,device);model.load_state_dict(ck['model'],strict=True)
 if model._feature_extractor!=expected_feature_report(c):raise ValueError('Own architecture differs')
 model.eval().requires_grad_(False);return model
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
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Own final50 no-query smoke')
  ids,m=observation_metadata();out.mkdir(parents=True,exist_ok=False);values={};times={}
  d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(Path(c['output_root'])/'final_ssdg.pth'),own_final50=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'))
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
 if done['status']!='PREDICTIONS_COMPLETE' or done['config']!=c or done['count']!=COUNT or done['views']!=list(d.VIEWS) or done['truth_read'] or done['query_fit'] or proof!=dict(status='VERIFIED',config=c,checkpoint=str(Path(c['output_root'])/'final_ssdg.pth'),own_final50=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'):raise ValueError('Truth closed: incomplete prediction/provenance')
 with np.load(out/'predictions.npz',allow_pickle=False) as p:
  if set(p.files)!=set(d.VIEWS)|{'ids'} or not np.array_equal(ids,p['ids']):raise ValueError('Fixed predictions/IDs differ')
  for view in d.VIEWS:
   a=p[view]
   if a.shape!=(COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid six-class decisions')
 return c,ids
