"""Real CP smoke first,then full fixed source-only shift counterfactual matrix."""
import inspect,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .core import paired,summary,qualify
from experiments.cvs_commonphase_source.core import select_fit,score
from experiments.cvs_commonphase_source.run import checkpoint_model

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU diagnostic only')
 torch.set_num_threads(2);c=d.config();d.validate(c);out=d.BASE;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  m,trained=checkpoint_model();before={k:v.detach().clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(20261016131)))['tx_logits']
  if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Real no-query CP smoke failed')
  d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',synthetic_NN_packets=2,source_IQ_reads=0,target_access=False,optimizer_updates=0));print('SMOKE PASS real matching checkpoint;2synthetic packets;continue',flush=True)
  from experiments.cvs_phase1_overlay.model import native_modules
  native=native_modules();payload=torch.load(Path(d.MODEL_SOURCE)/'final_ssdg.pth',map_location='cpu',weights_only=False);args=SimpleNamespace(**payload['args']);del payload
  if not args.use_aug or not args.aug_enable_pa_normal or args.aug_enable_class_signature or args.aug_p_dac!=0 or args.epochs!=20 or args.seed!=2026092701:raise ValueError('Actual original ordinary recipe differs')
  base=native.build_aug_base_cfg(args);original=d.read(Path(d.MODEL_SOURCE)/'ordinary_augmentation_actual.json')
  if base!=original['base_config'] or not original['no_second_pair_redraw'] or not original['practical_input_is_raw']:raise ValueError('Original ordinary effective config differs')
  aug=native.make_augmentor(base)
  carrier=d.read(d.SOURCE);prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
  if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']!=0:raise ValueError('Source preparation mismatch')
  packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
  for actual in (packed['contract'],prepared['source_contract']):
   if any(actual.get(k)!=v for k,v in carrier.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Reused source handle mismatch')
  L=select_fit({k:packed['labeled'][k] for k in ('x','y','rx','day','ids')},carrier,trained);del packed
  d.write(out/'fit_L_ids.json',L['ids']);d.write(out/'resolved_config.json',dict(config=c,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_commit=d.BACKEND_COMMIT,python=sys.executable,torch_version=torch.__version__,base_augmentation=base,actual_aug_args={k:v for k,v in vars(args).items() if k.startswith('aug_')},native_callable_files=dict(native=native.__file__,augmentation=inspect.getfile(type(aug)),build_aug_base_cfg=inspect.getfile(native.build_aug_base_cfg),configure_augmentor=inspect.getfile(native.configure_augmentor_for_epoch)),fit_unique_L=5040,other_source_roles_deserialized_but_unused=True,held_RX1_used=False,target_access=False))
  print('INPUT PASS fitL5040;exact original augmentor/args;held/U/V/target unused',flush=True)
  (out/'predictions').mkdir();labels=L['y'].tolist();rx=L['rx'].tolist();day=L['day'].tolist();cost=2
  def forward(x):
   nonlocal cost
   with torch.no_grad():value=m.forward_identity_only(x)['tx_logits'].double()
   if value.shape!=(len(x),6) or not bool(torch.isfinite(value).all()):raise ValueError('Complete finite default-head decisions required')
   cost+=len(x);return value.tolist()
  def metrics(logits):
   pred=[max(range(6),key=z.__getitem__) for z in logits];s=score(labels,pred)
   s['mean_CE']=float(torch.nn.functional.cross_entropy(torch.tensor(logits,dtype=torch.float64),L['y']))
   return pred,s
  rawlog=[]
  for start in range(0,5040,128):rawlog+=forward(L['x'][start:start+128].clone())
  rawpred,rawscore=metrics(rawlog);d.write(out/'predictions/raw.json',dict(labels=labels,predictions=rawpred,logits=rawlog,physical_ids=L['ids'],RX=rx,day=day))
  epochs=[];full_all=[];bypass_all=[];changed_all=[];nonzero_all=[];inspection_draws=0;shift_calls=0;canonical_checks=0
  for epoch in d.EPOCHS:
   state=native.configure_augmentor_for_epoch(aug,base,epoch,args);
   if epoch>args.label_epochs:raise ValueError("Native L schedule exceeds label epoch cap")
   random.seed(d.SEED+epoch);torch.manual_seed(d.SEED+epoch)
   full_log=[];bypass_log=[];records=[];nonzero=[];changed=[];shifts=[]
   for start in range(0,5040,128):
    x=L['x'][start:start+128].clone();y=L['y'][start:start+128];saved=x.clone();full,bypass,info=paired(aug,x,y,canonical=start==0)
    if not torch.equal(x,saved):raise ValueError('Source raw input mutated')
    rng=torch.random.get_rng_state();py_rng=random.getstate();full_log+=forward(full);bypass_log+=forward(bypass)
    if not torch.equal(rng,torch.random.get_rng_state()) or py_rng!=random.getstate():raise ValueError('Frozen eval consumed augmentation RNG')
    info.update(start=start,count=len(x));records.append(info);nonzero+=info['nonzero_shift'];changed+=info['changed'];shifts+=info['shifts'];inspection_draws+=info['shift_inspection_scalar_draws'];shift_calls+=info['shift_calls'];canonical_checks+=int(info['canonical_checked'])
    d.write(out/'progress.json',dict(epoch=epoch,batch_end=start+len(x),source_decisions=cost-2,total_source_decisions=136080,optimizer_updates=0,target_access=False))
   fullpred,fullscore=metrics(full_log);bypasspred,bypassscore=metrics(bypass_log);s=summary(labels,fullpred,bypasspred,nonzero,rx,day)
   if any(a!=b for a,b,ch in zip(fullpred,bypasspred,changed) if not ch):raise ValueError('Identical input must have identical frozen predictions')
   d.write(out/'predictions'/('epoch%02d.json'%epoch),dict(epoch=epoch,augmentation_seed=d.SEED+epoch,labels=labels,physical_ids=L['ids'],RX=rx,day=day,full_predictions=fullpred,bypass_predictions=bypasspred,full_logits=full_log,bypass_logits=bypass_log,shifts=shifts,nonzero_shift=nonzero,changed=changed,batches=records))
   record=dict(epoch=epoch,state=state,actual_time_parameters=dict(p_time_shift=aug.p_time_shift,max_time_shift=aug.max_time_shift),nonzero_shift=sum(nonzero),changed=sum(changed),full_score=fullscore,bypass_score=bypassscore,paired=s)
   epochs.append(record);full_all+=fullpred;bypass_all+=bypasspred;changed_all+=changed;nonzero_all+=nonzero
   d.write(out/'epoch_scores.json',epochs);print('EPOCH '+str(epoch)+' nonzero_shift='+str(sum(nonzero))+' changed='+str(sum(changed))+' full='+str(fullscore['accuracy'])+' bypass='+str(bypassscore['accuracy']),flush=True)
  total=summary(labels*len(d.EPOCHS),full_all,bypass_all,nonzero_all,rx*len(d.EPOCHS),day*len(d.EPOCHS));decision=qualify(total)
  if canonical_checks!=13 or cost!=136082 or m.id_backbone.encoder._view_eval_packets!=cost or m.id_backbone.encoder._view_train_packets or m.training or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m._joint_calls or m._supcon_calls or m._joint_As_packets:raise ValueError('Frozen state or actual NN cost mismatch')
  result=dict(status='SOURCE_SHIFT_COUNTERFACTUAL_COMPLETE',raw_score=rawscore,epochs=epochs,pooled=total,qualification=decision,nonzero_shift_total=sum(nonzero_all),changed_total=sum(changed_all),shift_calls=shift_calls,shift_inspection_scalar_draws=inspection_draws,canonical_whole_checks=canonical_checks,physical_L_packets=5040,ordinary_generated_packets=131040,canonical_check_generated_packets=1664,total_augmentation_generated_packets=132704,source_decisions=136080,actual_NN_packets=cost,training_updates=0,head_fit_updates=0,teacher_forwards=0,As_generations=0,model_state_unchanged=True,default_head_only=True,NoExplicitCFO=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,elapsed_seconds=time.perf_counter()-started)
  d.write(out/'result.json',result);d.write(out/'completion.json',dict(status=result['status'],decision=decision['decision'],epochs=13,source_decisions=136080,actual_NN_packets=cost,optimizer_updates=0,target_access=False,automatic_training=False));print('COMPLETE '+str(decision),flush=True)
 except Exception as error:d.write(out/'failure.json',dict(status='FAILED',error=repr(error),no_automatic_retry=True));raise
