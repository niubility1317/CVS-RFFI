"""Original ordinary IQ and same-IQ RMS restoration under frozen default heads."""
import inspect,os,random,sys,time
from types import SimpleNamespace
import torch
from . import design as d
from .checkpoint import model,fit_data
from .calibration import restore,qualify
from .metadata import validate_base,native_args_for_log

def score(y,p):
 if not y or len(y)!=len(p) or any(v not in range(6) for v in y+p):raise ValueError('Complete sixclass predictions required')
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 f=[]
 for i in range(6):
  den=sum(cm[i])+sum(z[i] for z in cm);f.append(2*cm[i][i]/den if den else 0.)
 return dict(count=len(y),accuracy=sum(cm[i][i] for i in range(6))/len(y),macro_F1=sum(f)/6,confusion=cm)

def run(seed):
 if seed not in d.SEEDS or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU source guard only')
 torch.set_num_threads(2);d.validate(d.config());out=d.resolve(d.BASE)/('s'+str(seed));out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cost=0
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,model_seed=seed))
 try:
  m,trained,provenance,native_args=model(seed);before={k:v.detach().clone() for k,v in m.state_dict().items()}
  if seed==d.SEEDS[0]:
   with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(2026101717)))['tx_logits']
   if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Realcheckpoint noquerysmoke failed')
   cost+=2;d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',synthetic_NN_packets=2,source_IQ_reads=0,optimizer_updates=0,target_access=False));print('SMOKE PASS real defaulthead;2synthetic IQ;continue',flush=True)
  from experiments.cvs_phase1_overlay.model import native_modules
  native=native_modules();args=SimpleNamespace(**native_args);logged_args,nonfinite_defaults=native_args_for_log(native_args)
  if args.seed!=seed or args.epochs!=20 or args.label_epochs!=13 or not args.use_aug or not args.aug_enable_pa_normal or args.aug_enable_class_signature or args.aug_p_dac!=0:raise ValueError('Original actual augmentation controls differ')
  base=native.build_aug_base_cfg(args)
  validate_base(base,seed)
  aug=native.make_augmentor(base);schedule=native.configure_augmentor_for_epoch(aug,base,13,args)
  synthetic=torch.randn(32,2,256,generator=torch.Generator().manual_seed(202610171740));sy=torch.arange(32)%6;random.seed(202610171741);torch.manual_seed(202610171741);pybegin=random.getstate();begin=torch.random.get_rng_state();canonical=aug(synthetic.clone(),labels=sy,no_pa=False);pyend=random.getstate();end=torch.random.get_rng_state();random.setstate(pybegin);torch.random.set_rng_state(begin);returned,sp=aug(synthetic.clone(),labels=sy,no_pa=False,return_pa_strength=True)
  if not torch.equal(canonical,returned) or pyend!=random.getstate() or not torch.equal(end,torch.random.get_rng_state()):raise ValueError('Strength-return changed native ordinary output or RNG')
  L=fit_data(trained);labels,rx,day=(L[k].tolist() for k in ('y','rx','day'));d.write(out/'source_metadata.json',dict(physical_ids=L['ids'],labels=labels,RX=rx,day=day))
  for rr in d.FIT:
   for dd in (1,2,3):
    for yy in range(6):
     if sum((a,b,c)==(rr,dd,yy) for a,b,c in zip(rx,day,labels))!=70:raise ValueError('Exactly72cells/70packets required')
  d.write(out/'resolved_config.json',dict(config=d.config(),executed_commit=(d.ROOT/'release_commit.txt').read_text().strip(),model_seed=seed,checkpoint_provenance=provenance,native_args=logged_args,native_args_nonfinite_defaults=nonfinite_defaults,native_args_log_only_conversion=True,native_args_log_dataclass_fields=True,actual_native_files=dict(native=native.__file__,augmentor=inspect.getfile(type(aug)),build=inspect.getfile(native.build_aug_base_cfg),schedule=inspect.getfile(native.configure_augmentor_for_epoch)),base_augmentation=base,effective_schedule=schedule,effective_epoch=13,actual_p_pa=aug.p_pa,canonical_native_output_RNG=True,synthetic_canonical_generated_packets=64,source_cache_L_deserialized=6300,unused_source_V_features_deserialized=True,U_roleID_metadata_only=True,U_IQ_or_labels_in_cache=False,fit_L_unique=5040,held_RX1_used=False,python=sys.executable,torch_version=torch.__version__,source_only=True,target_access=False))
  (out/'predictions').mkdir()
  def forward(x):
   nonlocal cost
   with torch.no_grad():z=m.forward_identity_only(x)['tx_logits'].double()
   if z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite originaldefaulthead output required')
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
  rawpred,rawscore=metrics(rawlog,list(range(5040)),True);d.write(out/'predictions/raw.json',dict(anchor_indices=list(range(5040)),labels=labels,predictions=rawpred,logits=rawlog))
  draws=[];allanchors=[];ordinary_log=[];restored_log=[];stats={k:[] for k in ('gain','relative_power_error','positive_scalar_shape_error')};PA=0
  for draw,augseed in enumerate(d.AUG_SEEDS):
   order=torch.randperm(5040,generator=torch.Generator().manual_seed(202610171710+draw)).tolist();random.seed(augseed);torch.manual_seed(augseed);olog=[];nlog=[];records=[]
   for start in range(0,5040,128):
    ai=order[start:start+128];x=L['x'][ai].clone();saved=x.clone();o,sp=aug(x,labels=L['y'][ai],no_pa=False,return_pa_strength=True);saved_o=o.clone();before_rng=torch.random.get_rng_state();before_py=random.getstate();n,info=restore(x,o)
    if not torch.equal(saved,x) or not torch.equal(saved_o,o) or not torch.equal(before_rng,torch.random.get_rng_state()) or before_py!=random.getstate():raise ValueError('RMS operation changed input/ordinary or RNG')
    olog+=forward(o);nlog+=forward(n)
    if not torch.equal(before_rng,torch.random.get_rng_state()) or before_py!=random.getstate():raise ValueError('Frozen eval consumed ordinary RNG')
    for k in stats:stats[k]+=info[k]
    PA+=int((sp>0).sum());records.append(dict(start=start,count=len(ai),anchor_indices=ai,pa_strength=sp.tolist(),**info));d.write(out/'progress.json',dict(model_seed=seed,draw=draw+1,effective_epoch=13,draw_completed=start+len(ai),NN_packets=cost,source_total=45360,optimizer_updates=0,target_access=False))
   op,oscore=metrics(olog,order);np,nscore=metrics(nlog,order);gain=100*(nscore['accuracy']-oscore['accuracy']);draws.append(dict(draw=draw+1,augmentation_seed=augseed,shuffle_seed=202610171710+draw,ordinary_score=oscore,RMS_score=nscore,accuracy_gain_pp=gain))
   d.write(out/'predictions'/('draw%02d.json'%(draw+1)),dict(draw=draw+1,augmentation_seed=augseed,shuffle_seed=202610171710+draw,anchor_indices=order,labels=[labels[i] for i in order],ordinary_predictions=op,RMS_predictions=np,ordinary_logits=olog,RMS_logits=nlog,batches=records));allanchors+=order;ordinary_log+=olog;restored_log+=nlog;print('DRAW '+str(draw+1)+' ordinary='+str(oscore['accuracy'])+' RMS='+str(nscore['accuracy']),flush=True)
  op,ordinaryscore=metrics(ordinary_log,allanchors,True);np,restoredscore=metrics(restored_log,allanchors,True);correct=[j for j,i in enumerate(allanchors) if rawpred[i]==labels[i]];restoredscore['raw_correct_count']=len(correct);restoredscore['raw_correct_retention']=sum(np[j]==labels[allanchors[j]] for j in correct)/len(correct);decision=qualify(rawscore,ordinaryscore,restoredscore,[z['accuracy_gain_pp'] for z in draws],stats)
  expected=45360+(2 if seed==d.SEEDS[0] else 0)
  if cost!=expected or m.id_backbone.encoder._view_eval_packets!=cost or m.id_backbone.encoder._view_train_packets or m.training or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m._joint_calls or m._supcon_calls or m._joint_As_packets:raise ValueError('Frozenstate oractual NN cost differs')
  result=dict(status='ANALYZED_SOURCE_SCALE_IDENTITY_ONLY',decision=decision['decision'],qualification=decision,model_seed=seed,raw_score=rawscore,ordinary_score=ordinaryscore,RMS_score=restoredscore,draws=draws,actual_NN_packets=cost,source_decisions=45360,unique_source_L=5040,ordinary_generated_packets=20160,RMS_generated_views=20160,synthetic_ordinary_canonical_packets=64,canonical_native_output_RNG=True,PA_observations=PA,checkpoint_loads=1,optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,model_state_unchanged=True,default_head_only=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-started)
  d.write(out/'result.json',result);d.write(out/'completion.json',{k:result[k] for k in ('status','decision','model_seed','actual_NN_packets','source_decisions','optimizer_updates','target_access','automatic_training')});print('COMPLETE '+decision['decision'],flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),no_automatic_retry=True));raise
