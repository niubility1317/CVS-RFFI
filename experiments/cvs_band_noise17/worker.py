"""All original L6300,existing default head;no fitted probes or target access."""
import os,sys,time
from collections import Counter
import numpy as np
import torch
from . import design as d
from .noise import paired,spectral_support

def metric(y,p):
 cm=np.zeros((6,6),dtype=np.int64)
 for a,b in zip(y,p):cm[int(a),int(b)]+=1
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(count=len(y),accuracy=float(np.trace(cm)/len(y)),macro_F1=float(f.mean()),confusion=cm.tolist())

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU source guard only')
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
  d.write(out/'resolved_config.json',dict(config=d.config(),execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),reference_commit=d.read(d.ROOT/'runtime.json')['reference_commit'],reference_checkpoint=str(src/'final_ssdg.pth'),reference_scratch_final20=True,reference_ancestor_sources=[],data_role_counts={k:len(v) for k,v in trained['role_ids'].items()},source_L=6300,source_RX_holdout=None,source_cache_other_roles_features_deserialized_but_unused=True,U_truth_used=False,V_statistically_used=False,target_access=False,head_fit_updates=0,model_parameters=sum(p.numel() for p in model.parameters()),checkpoint_loads=1,source_cache_loads=1))
  x=np.asarray(L['x'].tolist(),dtype=np.float64);support=np.asarray(spectral_support(x));support_stats={str(r):dict(count=rx.count(r),median=float(np.median(support[np.asarray(rx)==r])),q10=float(np.quantile(support[np.asarray(rx)==r],.1))) for r in d.FIT}
  logits={};info={};sourcepred={}
  raw=[]
  for st in range(0,6300,128):raw+=forward(L['x'][st:st+128].clone())
  logits['raw']=raw
  for draw in (0,1):
   wlog=[];clog=[];records=[]
   for st in range(0,6300,128):
    w,col,v=paired(x[st:st+128],ids[st:st+128],draw)
    def tensor(a):return torch.frombuffer(bytearray(np.ascontiguousarray(a).tobytes()),dtype=torch.float32).reshape(a.shape)
    wlog+=forward(tensor(w));clog+=forward(tensor(col));records.append(dict(start=st,count=len(w),**v));d.write(out/'progress.json',dict(draw=draw,source_completed=st+len(w),NN=cost,updates=0,target_access=False))
   logits['white_draw'+str(draw)]=wlog;logits['color_draw'+str(draw)]=clog;info[str(draw)]=records;print('DRAW_COMPLETE '+str(draw),flush=True)
  scores={};rawpred=np.asarray(logits['raw']).argmax(1);truth=np.asarray(y);rawcorrect=rawpred==truth
  for view,z in logits.items():
   zz=np.asarray(z);pred=zz.argmax(1);q=metric(y,pred);q['raw_correct_retention']=float(np.mean((pred==truth)[rawcorrect]))
   q['by_RX']={str(r):metric(truth[np.asarray(rx)==r],pred[np.asarray(rx)==r]) for r in d.FIT}
   q['by_TX']={str(t):dict(count=y.count(t),accuracy=float(np.mean(pred[truth==t]==t))) for t in range(6)}
   q['by_day']={str(t):metric(truth[np.asarray(day)==t],pred[np.asarray(day)==t]) for t in (1,2,3)}
   alt=zz.copy();alt[np.arange(6300),truth]=-np.inf;q['mean_true_margin_over30']=float(np.mean((zz[np.arange(6300),truth]-alt.max(1))/30));scores[view]=q;sourcepred[view]=pred
  criteria=d.config()['criteria'];conditions={'all_RX_source_spectral_support':all(v['median']>=.90 for v in support_stats.values())}
  for draw in (0,1):
   q=scores['color_draw'+str(draw)];ref=scores['raw'];delta=lambda k:100*(q[k]-ref[k]);records=info[str(draw)];changes=[n for r in records for n in r['color_relative_IQ_change']]
   conditions['draw'+str(draw)]=dict(accuracy=delta('accuracy')>=-.5,macro_F1=delta('macro_F1')>=-.5,retention=q['raw_correct_retention']>=.99,all_groups=all(100*(q['by_'+dimension][key]['accuracy']-ref['by_'+dimension][key]['accuracy'])>=-1. for dimension in ('RX','TX','day') for key in q['by_'+dimension]),effective_nonexcessive_noise=float(np.median(changes))>=.01 and max(changes)<=.15)
  passed=conditions['all_RX_source_spectral_support'] and all(all(conditions['draw'+str(k)].values()) for k in (0,1))
  if cost!=31502 or encoder._full_eval_packets!=cost or encoder._full_train_packets or model.training or any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('Frozen state/actual forward cost differs')
  np.savez_compressed(out/'source_predictions.npz',ids=np.asarray(ids),labels=truth,RX=np.asarray(rx),day=np.asarray(day),source_inband=support,**{k:np.asarray(v) for k,v in logits.items()})
  d.write(out/'noise_draws.json',info);result=dict(status='ANALYZED_SOURCE_ONLY',decision='PASS_SOURCE_IDENTITY_AND_COARSE_SPECTRAL_PRIOR_ONLY' if passed else 'FAIL_FIXED_BAND_NOISE_SOURCE_GUARD_CLOSED',conditions=conditions,scores=scores,source_support=support_stats,config=d.config(),actual_NN=cost,real_IQ_unique=6300,source_decisions=31500,CP=1,source_cache_loads=1,teacher_forwards=0,head_fit_updates=0,training_updates=0,source_RX_holdout=None,all_model_state_unchanged=True,target_access=False,Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-started)
  d.write(out/'result.json',result);d.write(out/'completion.json',{k:result[k] for k in ('status','decision','actual_NN','real_IQ_unique','training_updates','target_access','automatic_training')});print('COMPLETE '+result['decision'],flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),automatic_retry=False));raise
