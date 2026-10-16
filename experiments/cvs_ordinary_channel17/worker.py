"""Frozen defaulthead ordinary-channel total-effect factorial;sourcefit only."""
import inspect,os,random,sys,time
from types import SimpleNamespace
import torch
from . import design as d
from .checkpoint import model,fit_data
from .metadata import validate_base,native_args_for_log
from .paired import paired_views
from .decision import qualify

def score(y,p):
 if not y or len(y)!=len(p) or any(v not in range(6) for v in y+p):raise ValueError('Complete sixclass source decisions required')
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 fs=[]
 for i in range(6):
  den=sum(cm[i])+sum(z[i] for z in cm);fs.append(2*cm[i][i]/den if den else 0.)
 return dict(count=len(y),accuracy=sum(cm[i][i] for i in range(6))/len(y),macro_F1=sum(fs)/6,confusion=cm)

def run(seed):
 if seed not in d.SEEDS or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU sourcefactorial only')
 torch.set_num_threads(2);d.validate(d.config());out=d.resolve(d.BASE)/('s'+str(seed));out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cost=0
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,model_seed=seed))
 try:
  m,trained,provenance,native_args=model(seed);before={k:v.detach().clone() for k,v in m.state_dict().items()}
  if seed==d.SEEDS[0]:
   with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(2026101720)))['tx_logits']
   if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Realcheckpoint noquerysmoke failed')
   cost+=2;d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',synthetic_NN_packets=2,source_IQ_reads=0,optimizer_updates=0,target_access=False));print('SMOKE PASS real defaulthead;2synthetic IQ;continue',flush=True)
  from experiments.cvs_phase1_overlay.model import native_modules
  native=native_modules();args=SimpleNamespace(**native_args);logged_args,nonfinite_defaults=native_args_for_log(native_args)
  if args.seed!=seed or args.epochs!=20 or args.label_epochs!=13 or not args.use_aug or not args.aug_enable_pa_normal or args.aug_enable_class_signature or args.aug_p_dac!=0:raise ValueError('Originalnative augmentation controls differ')
  base=native.build_aug_base_cfg(args);validate_base(base,seed);aug=native.make_augmentor(base);schedule=native.configure_augmentor_for_epoch(aug,base,13,args)
  synthetic=torch.randn(32,2,256,generator=torch.Generator().manual_seed(202610172040));sy=torch.arange(32)%6;random.seed(202610172041);torch.manual_seed(202610172041);py=random.getstate();ts=torch.random.get_rng_state();canonical,canonical_sp=aug(synthetic.clone(),labels=sy,no_pa=False,return_pa_strength=True);pyend=random.getstate();tsend=torch.random.get_rng_state();random.setstate(py);torch.random.set_rng_state(ts);sv,sp,srecord=paired_views(aug,synthetic,sy)
  if not torch.equal(canonical,sv['full']) or not torch.equal(canonical_sp,sp) or pyend!=random.getstate() or not torch.equal(tsend,torch.random.get_rng_state()):raise ValueError('Instrumentation changed canonicalnative output/RNG')
  d.write(out/'resolved_config.json',dict(config=d.config(),executed_commit=(d.ROOT/'release_commit.txt').read_text().strip(),model_seed=seed,checkpoint_provenance=provenance,native_args=logged_args,native_args_nonfinite_defaults=nonfinite_defaults,native_args_log_only_conversion=True,native_args_log_dataclass_fields=True,actual_native_files=dict(native=native.__file__,augmentor=inspect.getfile(type(aug)),build=inspect.getfile(native.build_aug_base_cfg),schedule=inspect.getfile(native.configure_augmentor_for_epoch)),base_augmentation=base,effective_schedule=schedule,effective_epoch=13,canonical_native_output_RNG=True,synthetic_canonical_ordinary_generations=160,synthetic_kernel_replay_random_scalars=srecord['kernel_replay_random_scalars'],source_cache_L_deserialized=6300,unused_source_V_features_deserialized=True,U_roleID_metadata_only=True,U_IQ_or_labels_in_cache=False,fit_L_unique=5040,held_RX1_used=False,python=sys.executable,torch_version=torch.__version__,source_only=True,target_access=False))
  L=fit_data(trained);labels,rx,day=(L[k].tolist() for k in ('y','rx','day'));d.write(out/'source_metadata.json',dict(physical_ids=L['ids'],labels=labels,RX=rx,day=day))
  for rr in d.FIT:
   for dd in (1,2,3):
    for yy in range(6):
     if sum((a,b,c)==(rr,dd,yy) for a,b,c in zip(rx,day,labels))!=70:raise ValueError('Exactly72 sourcecells/70packets required')
  (out/'predictions').mkdir()
  def forward(x):
   nonlocal cost
   with torch.no_grad():z=m.forward_identity_only(x)['tx_logits'].double()
   if z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite defaulthead outputs required')
   cost+=len(x);return z.tolist()
  def metrics(logits,anchors,full=False):
   truth=[labels[i] for i in anchors];pred=[max(range(6),key=z.__getitem__) for z in logits];s=score(truth,pred);s['mean_CE']=float(torch.nn.functional.cross_entropy(torch.tensor(logits,dtype=torch.float64),torch.tensor(truth)))
   if full:
    for key,values,allowed in [('RX',rx,d.FIT),('TX',labels,range(6)),('day',day,(1,2,3))]:s['by_'+key]={str(v):score([truth[j] for j,i in enumerate(anchors) if values[i]==v],[pred[j] for j,i in enumerate(anchors) if values[i]==v]) for v in allowed}
    s['cells']=[]
    for rr in d.FIT:
     for dd in (1,2,3):
      for yy in range(6):
       ii=[j for j,i in enumerate(anchors) if (rx[i],day[i],labels[i])==(rr,dd,yy)];s['cells'].append(dict(RX=rr,day=dd,TX=yy,**score([truth[j] for j in ii],[pred[j] for j in ii])))
   return pred,s
  rawlog=[]
  for start in range(0,5040,128):rawlog+=forward(L['x'][start:start+128].clone())
  rawpred,rawscore=metrics(rawlog,list(range(5040)),True);d.write(out/'predictions/raw.json',dict(anchor_indices=list(range(5040)),labels=labels,predictions=rawpred,logits=rawlog));draws=[];allanchors=[];alllog={arm:[] for arm in d.ARMS};counts=dict(ordinary_generations=0,kernel_replay_random_scalars=0,MP_primitive_packets=0,AWGN_primitive_packets=0,MP_applied_packets=0,AWGN_applied_packets=0,PA_observations_per_arm=0)
  for draw,augseed in enumerate(d.AUG_SEEDS):
   order=torch.randperm(5040,generator=torch.Generator().manual_seed(202610171710+draw)).tolist();random.seed(augseed);torch.manual_seed(augseed);logs={arm:[] for arm in d.ARMS};batches=[]
   for start in range(0,5040,128):
    ai=order[start:start+128];x=L['x'][ai].clone();views,sp,record=paired_views(aug,x,L['y'][ai]);py=random.getstate();ts=torch.random.get_rng_state();stats={};px=x.double().square().sum((1,2));pref=x[:,:,80:160].double().square().sum((1,2)).clamp_min(1e-30)
    for arm in d.ARMS:
     value=views[arm];logs[arm]+=forward(value);stats[arm]=dict(relative_IQ_RMS_change=torch.sqrt((value.double()-x.double()).square().sum((1,2))/px).tolist(),relative_power_change=(value.double().square().sum((1,2))/px-1).tolist(),reference_relative_IQ_RMS_change=torch.sqrt((value[:,:,80:160].double()-x[:,:,80:160].double()).square().sum((1,2))/pref).tolist())
     for e in record['events'][arm]:
      counts[e['kind']+'_primitive_packets']+=len(x)
      if not e['removed']:counts[e['kind']+'_applied_packets']+=len(x)
    if py!=random.getstate() or not torch.equal(ts,torch.random.get_rng_state()):raise ValueError('Frozen eval consumed augmentationRNG')
    counts['ordinary_generations']+=record['ordinary_generations'];counts['kernel_replay_random_scalars']+=record['kernel_replay_random_scalars'];counts['PA_observations_per_arm']+=int((sp>0).sum());batches.append(dict(start=start,count=len(ai),anchor_indices=ai,pa_strength=sp.tolist(),input_stats=stats,**record));d.write(out/'progress.json',dict(model_seed=seed,draw=draw+1,draw_completed=start+len(ai),NN_packets=cost,source_total=85680,optimizer_updates=0,target_access=False))
   predictions={};scores={}
   for arm in d.ARMS:predictions[arm],scores[arm]=metrics(logs[arm],order);alllog[arm]+=logs[arm]
   gains=100*(scores['no_both']['accuracy']-scores['full']['accuracy']);interaction={key:100*(scores['no_both'][key]-scores['no_AWGN'][key]-scores['no_MP'][key]+scores['full'][key]) for key in ('accuracy','macro_F1')};draws.append(dict(draw=draw+1,augmentation_seed=augseed,shuffle_seed=202610171710+draw,scores=scores,primary_accuracy_gain_pp=gains,interaction_pp=interaction));d.write(out/'predictions'/('draw%02d.json'%(draw+1)),dict(draw=draw+1,augmentation_seed=augseed,shuffle_seed=202610171710+draw,anchor_indices=order,labels=[labels[i] for i in order],predictions=predictions,logits=logs,batches=batches));allanchors+=order;print('DRAW '+str(draw+1)+' full='+str(scores['full']['accuracy'])+' no_both='+str(scores['no_both']['accuracy']),flush=True)
  scores={};preds={};correct=[j for j,i in enumerate(allanchors) if rawpred[i]==labels[i]]
  for arm in d.ARMS:preds[arm],scores[arm]=metrics(alllog[arm],allanchors,True);scores[arm]['raw_correct_count']=len(correct);scores[arm]['raw_correct_retention']=sum(preds[arm][j]==labels[allanchors[j]] for j in correct)/len(correct)
  decision=qualify(rawscore,scores['full'],scores['no_both'],[x['primary_accuracy_gain_pp'] for x in draws]);interaction={key:100*(scores['no_both'][key]-scores['no_AWGN'][key]-scores['no_MP'][key]+scores['full'][key]) for key in ('accuracy','macro_F1')};expected=85680+(2 if seed==d.SEEDS[0] else 0)
  if cost!=expected or m.id_backbone.encoder._view_eval_packets!=cost or m.id_backbone.encoder._view_train_packets or m.training or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m._joint_calls or m._supcon_calls or m._joint_As_packets or counts['ordinary_generations']!=80640:raise ValueError('Frozenstate or actualcost differs')
  result=dict(status='ANALYZED_SOURCE_ORDINARY_CHANNEL_ONLY',decision=decision['decision'],qualification=decision,model_seed=seed,raw_score=rawscore,scores=scores,draws=draws,interaction_pp=interaction,actual_NN_packets=cost,source_decisions=85680,unique_source_L=5040,counts=counts,synthetic_canonical_ordinary_generations=160,checkpoint_loads=1,optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,model_state_unchanged=True,default_head_only=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-started)
  d.write(out/'result.json',result);d.write(out/'completion.json',{k:result[k] for k in ('status','decision','model_seed','actual_NN_packets','source_decisions','optimizer_updates','target_access','automatic_training')});print('COMPLETE '+decision['decision'],flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),no_automatic_retry=True));raise
