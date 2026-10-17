"""All original L6300,existing default head;no fitted probes or target access."""
import os,sys,time
from collections import Counter
import numpy as np
import torch
from . import design as d
from .coupled import observations

def metric(y,p):
 cm=np.zeros((6,6),dtype=np.int64)
 for a,b in zip(y,p):cm[int(a),int(b)]+=1
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(count=len(y),accuracy=float(np.trace(cm)/len(y)),macro_F1=float(f.mean()),confusion=cm.tolist())

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU source comparison only')
 torch.set_num_threads(2);out=d.resolve(d.BASE);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cost=0
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  from experiments.cvs_fullsource_concat20 import design as base,proof
  if base.RUN!=d.REFERENCE_RUN or tuple(base.FIT)!=d.FIT:raise ValueError('Full-source reference profile differs')
  c=base.config(next(r for r in base.rows() if r['row_id']==d.REFERENCE_ROW));src=proof.source(c)
  contract=d.read(d.resolve(d.SOURCE));trained=d.read(src/'source_contract.json')
  for k,v in contract.items():
   if trained.get(k)!=v:raise ValueError('Checkpoint source data contract differs')
  if trained.get('native_role_comparison')!='EXACT_MATCH' or {k:len(v) for k,v in trained['role_ids'].items()}!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Fullsource roles required before loading checkpoint')
  prep=d.resolve(d.PREPARATION);ready=d.read(prep/'preparation_complete.json')
  if ready['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or ready['target_access'] or ready['identity_updates']!=0:raise ValueError('Existing source preparation differs')
  packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False)
  for actual in (packed['contract'],ready['source_contract']):
   if any(actual.get(k)!=v for k,v in contract.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Source cache roles differ')
  L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
  ids=L['ids'];y,rx,day=(L[k].tolist() for k in ('y','rx','day'))
  if len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(contract['role_ids']['L_s']) or set(ids)&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])):raise ValueError('All and only original L6300 required')
  if L['x'].shape!=(6300,2,256) or L['x'].dtype!=torch.float32 or not bool(torch.isfinite(L['x']).all()) or Counter(zip(rx,day,y))!=Counter({(r,z,t):70 for r in d.FIT for z in (1,2,3) for t in range(6)}):raise ValueError('Full 90sourcecells/no RX holdout required')
  model=proof.load(c,'cpu');encoder=model.id_backbone.encoder;head=encoder.core.id_backbone.cls_head
  if encoder.sequence_pool.modes!=('SIG_TIME',)*3 or encoder.response.contract()['explicit_CFO_identity_input'] is not False or head.weight.shape!=(6,160) or head.scale!=30.:raise ValueError('Unchanged F/defaulthead/NoExplicitCFO required')
  before={k:v.detach().clone() for k,v in model.state_dict().items()}
  def forward(x):
   nonlocal cost
   with torch.no_grad():o=model.forward_identity_only(x);z=o['tx_logits']
   if z.shape!=(len(x),6) or o['id_feat_joint'].shape!=(len(x),160) or not bool(torch.isfinite(z).all()):raise ValueError('Finite defaulthead/joint160 required')
   cost+=len(x);return z.detach().cpu().tolist()
  forward(torch.zeros(2,2,256));d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,real_IQ_forwarded=0,query_read=False,head_fit_updates=0));print('SMOKE PASS real fullsource defaulthead;continue',flush=True)
  d.write(out/'resolved_config.json',dict(config=d.config(),execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),reference_commit=d.read(d.ROOT/'runtime.json')['reference_commit'],reference_checkpoint=str(src/'final_ssdg.pth'),reference_scratch_final20=True,reference_ancestor_sources=[],data_role_counts={k:len(v) for k,v in trained['role_ids'].items()},source_L=6300,source_RX_holdout=None,source_cache_other_roles_features_deserialized_but_unused=True,U_truth_used=False,V_statistically_used=False,target_access=False,head_fit_updates=0,model_parameters=sum(p.numel() for p in model.parameters()),checkpoint_loads=1,source_cache_loads=1,channel_module=sys.modules['leo_practical.channel'].__file__,residual_module=sys.modules['leo_practical.residual'].__file__))
  logits={'raw':[]};records={}
  for st in range(0,6300,128):logits['raw']+=forward(L['x'][st:st+128].clone())
  for scene in d.SCENES:
   records[scene]=[]
   for k in d.VARIANTS:logits[scene+'/'+k]=[]
   for st in range(0,6300,128):
    a,meta=observations(np.asarray(L['x'][st:st+128].tolist(),dtype=np.float64),ids[st:st+128],rx[st:st+128],day[st:st+128],scene)
    for k,v in a.items():
     xx=torch.frombuffer(bytearray(v.tobytes()),dtype=torch.float32).reshape(v.shape);logits[scene+'/'+k]+=forward(xx)
    records[scene]+=meta;d.write(out/'progress.json',dict(scene=scene,source_completed=st+len(meta),NN=cost,updates=0,target_access=False))
   print('SCENE_COMPLETE '+scene,flush=True)
  from .decide import summarize,decide
  scores=summarize(logits,y,rx,day,records);conditions=decide(logits,y,records,scores)
  passed=all(conditions.values());decision='PASS_SOURCE_N0_OBSERVATION_LAW_ONLY' if passed else 'FAIL_FIXED_N0_PRIOR_SOURCE_CANDIDATE_CLOSED'
  if cost!=100802 or encoder._full_eval_packets!=cost or encoder._full_train_packets or model.training or any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('Frozen model or actual budget changed')
  np.savez_compressed(out/'source_predictions.npz',ids=np.asarray(ids),labels=np.asarray(y),RX=np.asarray(rx),day=np.asarray(day),**{k:np.asarray(v) for k,v in logits.items()})
  d.write(out/'coupled_draws.json',records)
  coverage={scene:dict(total=6300,variants={key:dict(lock_counts=dict(Counter(r['variants'][key]['lock'] for r in rr)),quality_changed=sum(abs(r['variants'][key]['quality']-r['nominal_quality'])>1e-10 for r in rr),lock_changed=sum(r['variants'][key]['lock']!=r['nominal_lock'] for r in rr),AGC_limited=sum(r['variants'][key]['AGC_limited'] for r in rr),mean_N0_factor=d.FACTORS[key],max_fixed_signal_error=max((r['variants'][key]['fixed_quality_signal_error'] or 0) for r in rr),max_fixed_noise_error=max((r['variants'][key]['fixed_quality_noise_error'] or 0) for r in rr)) for key in d.VARIANTS},all_H_and_hardware_unchanged=all(r['same_H'] and r['same_hardware'] for r in rr),expected_reference_N0_factor=1.) for scene,rr in records.items()}
  result=dict(status='ANALYZED_SOURCE_ONLY',decision=decision,conditions=conditions,scores=scores,coverage=coverage,config=d.config(),actual_NN=cost,real_IQ_unique=6300,source_decisions=100800,physical_realizations=18900,processing_calls=94500,CP=1,source_cache_loads=1,teacher_forwards=0,head_fit_updates=0,training_updates=0,source_RX_holdout=None,all_model_state_unchanged=True,target_access=False,Ag_efficacy_proved=False,As_training_gain_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-started)
  d.write(out/'result.json',result);d.write(out/'completion.json',{k:result[k] for k in ('status','decision','actual_NN','real_IQ_unique','training_updates','target_access','automatic_training')});print('COMPLETE '+decision,flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),automatic_retry=False));raise
