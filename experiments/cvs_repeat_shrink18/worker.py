"""Two original full-source fixed20 readouts;V only,zero optimizer/teacher/target."""
import os,sys,time,gc,random
from collections import Counter
from types import SimpleNamespace
import torch
import numpy as np
from . import design as d,dsp

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES') in ('',None):raise ValueError('Reserved one GPU required')
 d.validate(d.config());out=d.resolve(d.BASE);started=time.perf_counter();torch.set_num_threads(2)
 from experiments.cvs_fullsource_concat20 import design as ref,proof,model as api
 from experiments.cvs_equivariant_identity.precision import numerical_context
 contract=d.read(d.resolve(d.SOURCE));models={};states={};args={};NN=0
 with numerical_context(ref.FULL_FP32_POLICY),torch.no_grad():
  # The launcher's first model action is the real-checkpoint no-query smoke.
  for arm in d.MODELS:
   c=ref.config(next(r for r in ref.rows() if r['row_id']==arm+'-s2026092701'));p=proof.source(c);sc=d.read(p/'source_contract.json')
   if sc['role_ids']!=contract['role_ids'] or sc.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
   cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
   if (cp['epoch'],cp['run_id'],cp['candidate_id'],cp['a1_ema_successful_updates'],cp['checkpoint_selection'])!=(20,d.REFERENCE_RUN,arm+'-s2026092701',4440,'final_only') or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt'] or cp['args'].get('source_fold_held_rx') is not None or cp['args']['freeze_backbone']:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
   args[arm]=cp['args'];m=api.build(c,'cuda:0');m.load_state_dict(cp['model'],strict=True);m.eval().requires_grad_(False);del cp
   head=m.id_backbone.encoder.core.id_backbone.cls_head
   if head.weight.shape!=(6,160) or head.scale!=30. or m.id_backbone.encoder.sequence_pool.modes!=('SIG_TIME',)*3 or m.id_backbone.encoder.response.contract()['explicit_CFO_identity_input'] is not False:raise ValueError('Existing F/defaulthead/NoExplicitCFO required')
   s=m.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'));NN+=2
   if s['tx_logits'].shape!=(2,6) or s['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(s['tx_logits']).all()):raise ValueError('Real-checkpoint smoke failed')
   models[arm]=m;states[arm]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',CP=2,NN=4,IQ=0,query_read=False,updates=0));print('SMOKE PASS real2CP/defaulthead/noquery;continue sourceV',flush=True)
  prep=d.resolve(d.PREPARATION);ready=d.read(prep/'preparation_complete.json');packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False)
  if ready['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or ready['target_access'] or packed['contract']['role_ids']!=contract['role_ids'] or ready['source_contract']['role_ids']!=contract['role_ids']:raise ValueError('Existing unchanged source handles differ')
  V={k:packed['validation'][k] for k in ('x','y','rx','day','ids')};del packed
  y,rx,day=(V[k].tolist() for k in ('y','rx','day'));ids=V['ids']
  if len(ids)!=27000 or set(ids)!=set(contract['role_ids']['V']) or V['x'].shape!=(27000,2,256) or Counter(zip(rx,day,y))!=Counter({(r,z,t):300 for r in d.FIT for z in (1,2,3) for t in range(6)}):raise ValueError('Original V27000/all5RX required')
  selected=[]
  for t in range(6):
   for r in d.FIT:
    for z in (1,2,3):selected+=sorted((i for i in range(27000) if (y[i],rx[i],day[i])==(t,r,z)),key=lambda i:ids[i])[:d.PER_CELL]
  if len(selected)!=1800 or len(set(selected))!=1800:raise ValueError('Fixed V1800 grid required')
  x=V['x'][selected].float().to('cuda:0');labels=V['y'][selected].long().to('cuda:0');selected_ids=[ids[i] for i in selected];raw=x.clone()
  # Reproduce the actual common ordinary policy, not its unscaled defaults.
  if any(args[d.MODELS[0]][k]!=args[d.MODELS[1]][k] for k in args[d.MODELS[0]] if k.startswith('aug_')):raise ValueError('Common ordinary args differ across original parents')
  n=sys.modules['SSDG.train_ssdg'];a=SimpleNamespace(**args[d.MODELS[0]]);base_cfg=n.build_aug_base_cfg(a)
  if base_cfg['enable_class_signature']:raise ValueError('No class signature injection')
  aug=n.make_augmentor(base_cfg);stage=n.configure_augmentor_for_epoch(aug,base_cfg,13,a);ordinary=[]
  python_rng=random.getstate()
  try:
   random.seed(d.ORDINARY_SEED)
   with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(d.ORDINARY_SEED)
    for start in range(0,1800,128):ordinary.append(aug(x[start:start+128],labels=labels[start:start+128],no_pa=(not bool(a.aug_enable_pa_normal))))
  finally:random.setstate(python_rng)
  ordinary=torch.cat(ordinary)
  if not bool(torch.isfinite(ordinary).all()) or not torch.equal(x,raw):raise ValueError('Common ordinary changed raw/nonfinite input')
  observations=dict(raw=raw,ordinary=ordinary);views={};numerics=[]
  for name,b in observations.items():
   # Safe value bridge for the pinned Torch/NumPy ABI;no .numpy/from_numpy.
   base_array=np.asarray(b.detach().cpu().tolist(),dtype=np.float32)
   for strength in (0.,.1):
    arr=dsp.apply(base_array,strength);numerics.append(dict(base=name,**dsp.numerics(base_array,arr,strength)))
    z=torch.frombuffer(bytearray(arr.tobytes(order='C')),dtype=torch.float32).reshape(arr.shape).clone().to(b.device)
    if not bool(torch.isfinite(z).all()) or (strength==0. and not torch.equal(z,b)):raise ValueError('Identity/finite bridge failed')
    views[(name,str(strength))]=z
  d.write(out/'resolved_config.json',dict(config=d.config(),source_role_counts={k:len(v) for k,v in contract['role_ids'].items()},selected_V_ids=selected_ids,ordinary_actual_stage=stage,ordinary_base_cfg=base_cfg,ordinary_PA_normal=bool(a.aug_enable_pa_normal),classifier='original6x160cosine/scale30',feature='F/SIG_TIME_X3/CENTER_25K',numerical_policy=ref.FULL_FP32_POLICY,model_module=api.__file__,augmentation_module=n.__file__,device='oneGPU',target_access=False,updates=0))
  pred={}
  for arm,m in models.items():
   pred[arm]={}
   for (base,strength),z in views.items():
    logits=[]
    for start in range(0,1800,128):
     v=m.forward_identity_only(z[start:start+128]);NN+=len(v['tx_logits'])
     if v['tx_logits'].shape!=(len(z[start:start+128]),6) or v['id_feat_joint'].shape!=(len(z[start:start+128]),160) or not bool(torch.isfinite(v['tx_logits']).all()):raise ValueError('Default head invalid')
     logits+=v['tx_logits'].cpu().tolist()
    pred[arm][base+'/'+strength]=logits;print('VIEW_COMPLETE '+arm+' '+base+' '+strength,flush=True)
   if any(not torch.equal(v.detach().cpu(),states[arm][k]) for k,v in m.state_dict().items()):raise ValueError('Frozen state changed')
  if NN!=14404:raise ValueError('Fixed source forward budget differs')
  d.write(out/'predictions.json',dict(ids=selected_ids,y=labels.cpu().tolist(),rx=[rx[i] for i in selected],day=[day[i] for i in selected],logits=pred))
  d.write(out/'prediction_complete.json',dict(status='SOURCE_PREDICTIONS_COMPLETE',config=d.config(),NN=NN,source_V_selected=1800,source_cache_loads=1,CP=2,ordinary_generations=1800,operator_applications=7200,nonzero_operator_applications=3600,model_state_unchanged=True,numerics=numerics,ordinary_actual_stage=stage,source_RX_holdout=None,teacher_NN=0,updates=0,target_access=False,head_fit_updates=0,wall_seconds=time.perf_counter()-started))
 os.environ['CUDA_VISIBLE_DEVICES']='';os.execv(sys.executable,[sys.executable,'-u',str(d.ROOT/'experiments/cvs_repeat_shrink18/runner.py'),'--analyze'])
