"""Bounded nonaffine phase on legal fitL;two frozen original default heads."""
import gzip,inspect,json,os,random,sys,time
from types import SimpleNamespace
import torch
from . import design as d
from .checkpoint import model,fit_data
from .metadata import validate_base,native_args_for_log
from .phase import generate
from .decision import qualify

def score(y,p):
 if not y or len(y)!=len(p) or any(v not in range(6) for v in y+p):raise ValueError('Complete source sixclass decisions required')
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 f=[]
 for i in range(6):
  den=sum(cm[i])+sum(z[i] for z in cm);f.append(2*cm[i][i]/den if den else 0.)
 return dict(count=len(y),accuracy=sum(cm[i][i] for i in range(6))/len(y),macro_F1=sum(f)/6,confusion=cm)

def run(seed):
 if seed not in d.SEEDS or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU sourceguard only')
 torch.set_num_threads(2);d.validate(d.config());out=d.resolve(d.BASE)/('s'+str(seed));out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cost=0
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,model_seed=seed))
 try:
  m,trained,provenance,native_args=model(seed);before={k:v.detach().clone() for k,v in m.state_dict().items()}
  if seed==d.SEEDS[0]:
   with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(2026101732)))['tx_logits']
   if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Realcheckpoint noquerysmoke failed')
   cost+=2;d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',synthetic_NN_packets=2,source_IQ_reads=0,optimizer_updates=0,target_access=False));print('SMOKE PASS real defaulthead;2synthetic IQ;continue',flush=True)
  from experiments.cvs_phase1_overlay.model import native_modules
  native=native_modules();args=SimpleNamespace(**native_args);logged,nonfinite=native_args_for_log(native_args)
  if args.seed!=seed or args.epochs!=20 or args.label_epochs!=13 or not args.use_aug or not args.aug_enable_pa_normal or args.aug_enable_class_signature or args.aug_p_dac!=0:raise ValueError('Original baseline augmentation controls differ')
  base=native.build_aug_base_cfg(args);validate_base(base,seed);aug=native.make_augmentor(base);module=inspect.getmodule(type(aug));synthetic=torch.randn(32,2,256,generator=torch.Generator().manual_seed(202610173240));torch.manual_seed(202610173241);value,info=generate(synthetic)
  if value.shape!=synthetic.shape or len(info['phase_rad'])!=32:raise ValueError('Synthetic phase smoke differs')
  d.write(out/'resolved_config.json',dict(config=d.config(),executed_commit=(d.ROOT/'release_commit.txt').read_text().strip(),model_seed=seed,checkpoint_provenance=provenance,native_args=logged,native_args_nonfinite_defaults=nonfinite,native_args_log_only_conversion=True,native_args_log_dataclass_fields=True,base_augmentation_of_existing_model=base,old_ordinary_generated=0,actual_native_files=dict(native=native.__file__,augmentor=inspect.getfile(type(aug)),phase=inspect.getfile(generate)),synthetic_phase_views_generated=32,source_cache_L_deserialized=6300,unused_source_V_features_deserialized=True,U_roleID_metadata_only=True,U_IQ_or_labels_in_cache=False,fit_L_unique=5040,held_RX1_used=False,python=sys.executable,torch_version=torch.__version__,source_only=True,target_access=False))
  L=fit_data(trained);labels,rx,day=(L[k].tolist() for k in ('y','rx','day'));d.write(out/'source_metadata.json',dict(physical_ids=L['ids'],labels=labels,RX=rx,day=day))
  for rr in d.FIT:
   for dd in (1,2,3):
    for yy in range(6):
     if sum((a,b,c)==(rr,dd,yy) for a,b,c in zip(rx,day,labels))!=70:raise ValueError('Exactly72sourcecells/70packets required')
  (out/'predictions').mkdir()
  def forward(x):
   nonlocal cost
   with torch.no_grad():z=m.forward_identity_only(x)['tx_logits'].double()
   if z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite defaulthead logits required')
   cost+=len(x);return z.tolist()
  def metrics(logits,anchors):
   truth=[labels[i] for i in anchors];pred=[max(range(6),key=z.__getitem__) for z in logits];s=score(truth,pred);s['mean_CE']=float(torch.nn.functional.cross_entropy(torch.tensor(logits,dtype=torch.float64),torch.tensor(truth)))
   for key,values,allowed in [('RX',rx,d.FIT),('TX',labels,range(6)),('day',day,(1,2,3))]:s['by_'+key]={str(v):score([truth[j] for j,i in enumerate(anchors) if values[i]==v],[pred[j] for j,i in enumerate(anchors) if values[i]==v]) for v in allowed}
   s['cells']=[]
   for rr in d.FIT:
    for dd in (1,2,3):
     for yy in range(6):
      ii=[j for j,i in enumerate(anchors) if (rx[i],day[i],labels[i])==(rr,dd,yy)];s['cells'].append(dict(RX=rr,day=dd,TX=yy,**score([truth[j] for j in ii],[pred[j] for j in ii])))
   return pred,s
  def retain(pred,anchors,s):
   rc=[j for j,i in enumerate(anchors) if rawpred[i]==labels[i]];s['raw_correct_count']=len(rc);s['raw_correct_retention']=sum(pred[j]==labels[anchors[j]] for j in rc)/len(rc)
  rawlog=[]
  for start in range(0,5040,128):rawlog+=forward(L['x'][start:start+128].clone())
  rawpred,rawscore=metrics(rawlog,list(range(5040)));d.write(out/'predictions/raw.json',dict(anchor_indices=list(range(5040)),labels=labels,predictions=rawpred,logits=rawlog));draws=[];allanchors=[];alllog=[];all_iq=[]
  for draw,augseed in enumerate(d.AUG_SEEDS):
   order=torch.randperm(5040,generator=torch.Generator().manual_seed(d.SHUFFLE_SEEDS[draw])).tolist();random.seed(augseed);torch.manual_seed(augseed);logits=[];batches=[];phase_records=[]
   for start in range(0,5040,128):
    ai=order[start:start+128];x=L['x'][ai].clone();saved=x.clone();value,info=generate(x);phase_records+=info.pop('phase_rad');py=random.getstate();ts=torch.random.get_rng_state();logits+=forward(value)
    if not torch.equal(x,saved) or py!=random.getstate() or not torch.equal(ts,torch.random.get_rng_state()):raise ValueError('Input changed orfrozen eval consumednoiseRNG')
    all_iq+=info['relative_IQ_RMS_change'];batches.append(dict(start=start,count=len(ai),anchor_indices=ai,**info));d.write(out/'progress.json',dict(model_seed=seed,draw=draw+1,draw_completed=start+len(ai),NN_packets=cost,source_total=25200,optimizer_updates=0,target_access=False))
   with gzip.open(out/'predictions'/('phase_paths_draw%02d.json.gz'%(draw+1)),'wt',encoding='utf8') as stream:json.dump(dict(anchor_indices=order,phase_rad=phase_records),stream,allow_nan=False,separators=(',',':'))
   pred,s=metrics(logits,order);retain(pred,order,s);draws.append(dict(draw=draw+1,augmentation_seed=augseed,shuffle_seed=d.SHUFFLE_SEEDS[draw],score=s));d.write(out/'predictions'/('draw%02d.json'%(draw+1)),dict(draw=draw+1,augmentation_seed=augseed,shuffle_seed=d.SHUFFLE_SEEDS[draw],anchor_indices=order,labels=[labels[i] for i in order],predictions=pred,logits=logits,batches=batches));allanchors+=order;alllog+=logits;print('DRAW '+str(draw+1)+' phase='+str(s['accuracy']),flush=True)
  pred,phase_score=metrics(alllog,allanchors);retain(pred,allanchors,phase_score);decision=qualify(rawscore,phase_score,[z['score'] for z in draws],dict(relative_IQ_RMS_change=all_iq));expected=25200+(2 if seed==d.SEEDS[0] else 0)
  if cost!=expected or m.id_backbone.encoder._view_eval_packets!=cost or m.id_backbone.encoder._view_train_packets or m.training or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m._joint_calls or m._supcon_calls or m._joint_As_packets:raise ValueError('Frozenmodelstate or NN count differs')
  result=dict(status='ANALYZED_SOURCE_BOUNDED_NONAFFINE_PHASE_ONLY',decision=decision['decision'],qualification=decision,model_seed=seed,raw_score=rawscore,phase_score=phase_score,draws=draws,actual_NN_packets=cost,source_decisions=25200,unique_source_L=5040,phase_views_generated=20160,synthetic_phase_views_generated=32,old_ordinary_generated=0,PA_generated=0,As_generated=0,checkpoint_loads=1,optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,model_state_unchanged=True,default_head_only=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-started)
  d.write(out/'result.json',result);d.write(out/'completion.json',{k:result[k] for k in ('status','decision','model_seed','actual_NN_packets','source_decisions','optimizer_updates','target_access','automatic_training')});print('COMPLETE '+decision['decision'],flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),no_automatic_retry=True));raise
