"""P60 four-row CPU recount; no Torch/model/IQ access or new scoring launch."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse,csv,gzip,json,math,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
RUN='20261018-phase1-FcrossRX-PAwaveform-tail-e30-s2026092701-m2-r01'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
PANEL_RUN='20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01'
CONTRACT_RUN='phase1_daot_rc4_pure_game_m3_20260917_r2'
SEED=2026092701;COUNT=168000;SCALE=30.;STEPS=222
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
STATES=('LOS','PARTIAL','BLOCKED');LOCKS=('locked','degraded','unlocked')
RX=(1,3,4,6,8);DAYS=(1,2,3)
TARGET_RX=('1-1','14-7','2-1','20-1','7-14','7-7','8-8')
TARGET_DAYS=('2021_03_01','2021_03_08','2021_03_15','2021_03_23')
ARMS=('NOAS30','AS30','NOAS_PAOFF','AS_PAOFF')
PAIRS=(('AS_PAOFF','NOAS_PAOFF'),('AS_PAOFF','AS30'),('NOAS_PAOFF','NOAS30'),('AS30','NOAS30'))
COUNTERS=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def json_scalar(value):
 if isinstance(value,np.generic):return value.item()
 raise TypeError('Unsupported JSON value type: '+type(value).__name__)
def write(path,obj):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False,default=json_scalar);f.write('\n')
def csv_write(path,rows):
 rows=list(rows)
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with Path(path).open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def lines(path):
 with (gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as f:
  for line in f:
   if line.strip():yield json.loads(line)
def require(ok,message):
 if not ok:raise ValueError(message)
def finite(value,name):
 require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),'Nonfinite active field '+name);return float(value)
def row_specs(project):
 p=Path(project)/'runs';result=[]
 for key,arm in [('NOAS30','B00'),('AS30','B01'),('NOAS_PAOFF','B00'),('AS_PAOFF','B01')]:
  reused=key.endswith('30');run=REF if reused else RUN;row=(arm if reused else key)+'-s'+str(SEED)
  result.append(dict(key=key,run=run,row_id=row,arm=arm if reused else key,reference_arm=arm,root=p/run/row,reused=reused))
 return result

def training_complete(spec,project):
 root=spec['root'];roles=read(Path(project)/'runs'/CONTRACT_RUN/'source_contract.json')['role_ids']
 parent=Path(project)/'runs'/REF/(spec['reference_arm']+'-s'+str(SEED))/'stage20'
 stages=[(parent,20,13,7,4440),(root/'source',10,0,10,2220)]
 for folder,e,l,u,steps in stages:
  done=read(folder/'completion.json');c=done['config'];frozen=read(folder/'source_frozen.json');init=read(folder/'initialization.json');result=read(folder/'result.json')
  is_parent=folder==parent;expected_run=REF if is_parent else spec['run'];expected_row=spec['reference_arm']+'-s'+str(SEED) if is_parent else spec['row_id']
  require((done.get('epoch'),done.get('optimizer_steps'),c.get('epochs'),c.get('label_epochs'),c.get('pseudo_epochs'),c.get('updates'))==(e,steps,e,l,u,steps),'Exact own20+10 training calendar required')
  require(c.get('run_id')==expected_run and c.get('row_id')==expected_row and c.get('model_seed')==SEED,'Source row/run/seed differs')
  require(not done.get('target_access') and done.get('head_fit_updates')==0 and done.get('native_fitted_probes_disabled') is True,'Source default head/no target flags differ')
  require(frozen.get('status')=='SOURCE_FROZEN' and frozen.get('config')==c and frozen.get('selection')=='final_only' and not frozen.get('target_access'),'Final-only source freeze required')
  expected=[] if is_parent else [str(parent/'final_ssdg.pth')]
  require(init.get('checkpoint_sources')==expected and init.get('scratch_only')==is_parent and not init.get('target_contact') and init.get('source_roles')=='EXACT_MATCH','Own legal parent provenance differs')
  require(read(folder/'source_contract.json')['role_ids']==roles,'Actual L/U/V roles differ')
  require(result.get('config')==c and result.get('head_fit_updates')==0 and result.get('U_truth_read') is False and result.get('target_access') is False,'Joint defaulthead/hidden U/source result differs')
  require(c.get('Ag_mode')=='NONE' and c.get('new_Ag') is False and c.get('L_NN_per_step')==768 and c.get('total_L_loss_scale')==1.25,'Fixed no-Ag sixL loss contract required')
  require(result.get('source_RX')==list(RX) and result.get('source_RX_holdout') is None,'All source RX required')
  require(done['final_source_metrics'].get('source_val_count')==27000,'Full source V27000 required')
  if not is_parent:
   require(c.get('parent_source')==str(parent) and c.get('epoch_offset')==20 and c.get('step_offset')==4440,'Exact own20 parent and cumulative clock required')
   if not spec['reused']:
    pa=c.get('PA_waveform',{});prior=c.get('scene_prior',{})
    require(c.get('arm')==spec['key'] and c.get('As')==(spec['key']=='AS_PAOFF') and pa.get('injected_PA_packets')==0 and pa.get('parent_E1_E20_PA') is True and pa.get('native_p_pa_unchanged') is True and pa.get('original_simulate_pa_runs_on_clone') is True and pa.get('U_changed') is False and pa.get('As_changed') is False and prior.get('probabilities')==[1/3]*3,'Only fixed PA waveform return change and nominal As allowed')
 return read(root/'source/completion.json')

def frozen_predictions(spec,done,index_ids,count=COUNT):
 """No truth access: validate every view before returning any prediction array."""
 root=Path(spec['root']);folder=root/'prediction';p=read(folder/'complete.json');s=read(root/'scoring/complete.json');resolved=read(folder/'resolved_config.json');rowdone=read(root/'complete.json');c=done['config']
 require(p.get('status')=='PREDICTIONS_COMPLETE' and p.get('row_id')==root.name and p.get('count')==count and p.get('views')==list(VIEWS) and p.get('config')==c,'Truth closed: incomplete prediction identity/config')
 require(p.get('truth_read') is False and p.get('query_fit') is False and p.get('all_parameters_frozen') is True and p.get('head_fit_updates')==0,'Truth closed: prediction not frozen/defaulthead')
 require(resolved.get('config')==c and resolved.get('views')==list(VIEWS) and resolved.get('model_frozen') is True and resolved.get('separate_head_fit') is False and resolved.get('truth_read') is False and resolved.get('query_fit') is False,'Truth closed: resolved predictor flags differ')
 require(resolved.get('checkpoint')==str(root/'source/final_ssdg.pth'),'Truth closed: own final checkpoint metadata differs')
 require(s.get('status')=='SCORED_COMPLETE' and s.get('row_id')==root.name and s.get('views')==list(VIEWS) and s.get('truth_last') is True,'Truth closed: independent scoring incomplete')
 require(rowdone.get('status')=='TRAINED_AND_SCORED' and rowdone.get('row_id')==root.name,'Truth closed: row closure incomplete')
 require(p.get('actual_NN')==count*7+2,'Frozen defaulthead prediction NN differs')
 ids=np.asarray(index_ids);require(ids.shape==(count,) and ids.dtype.kind in 'US' and len(set(ids.tolist()))==count,'Truth closed: unique formal IDs required')
 with np.load(folder/'predictions.npz',allow_pickle=False) as f:
  require(set(f.files)=={'ids',*VIEWS} and np.array_equal(f['ids'],ids),'Truth closed: prediction IDs/views differ');pred={v:f[v].copy() for v in VIEWS}
 logits={}
 for view in VIEWS:
  a=pred[view];require(a.shape==(count,) and a.dtype.kind in 'iu' and np.all((a>=0)&(a<6)),'Truth closed: bad decisions')
  with np.load(folder/(view+'_scores.npz'),allow_pickle=False) as f:
   z=f['logits'];require(np.array_equal(f['ids'],ids) and z.shape==(count,6) and np.isfinite(z).all() and np.array_equal(z.argmax(1),a),'Truth closed: bad logits/IDs/argmax');logits[view]=z.copy()
 return dict(ids=ids.copy(),pred=pred,logits=logits,prediction=p,scoring=s)

def truth_last_arrays(specs,index_ids,truth_path,project,count=COUNT,checker=None):
 """All logical rows and all7 scores are checked before the sole truth read."""
 checker=checker or training_complete;data={}
 for spec in specs:data[spec['key']]=frozen_predictions(spec,checker(spec,project),index_ids,count)
 truth=read(truth_path) # Deliberately after the entire four-row gate.
 ids=np.asarray(index_ids);require(set(truth)==set(ids.tolist()),'Truth physical IDs differ')
 y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids]);day=np.asarray([str(truth[s]['day']) for s in ids])
 require(y.dtype.kind in 'iu' and y.shape==(count,) and np.all((y>=0)&(y<6)),'Truth class map differs')
 if count==COUNT:
  require(set(rx)==set(TARGET_RX) and set(day)==set(TARGET_DAYS),'Fixed heldout RX/day differ')
  for label in range(6):
   for receiver in TARGET_RX:
    for date in TARGET_DAYS:require(int(((y==label)&(rx==receiver)&(day==date)).sum())==1000,'Fixed formal cell count differs')
 return data,dict(ids=ids,y=y,rx=rx,day=day)

def interaction_value(values):
 require(set(values)==set(ARMS),'All four matched logical cells required')
 return values['AS_PAOFF']-values['NOAS_PAOFF']-values['AS30']+values['NOAS30']

def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);require(y.ndim==1 and p.shape==y.shape and y.dtype.kind in 'iu' and p.dtype.kind in 'iu' and np.all((y>=0)&(y<6)&(p>=0)&(p<6)),'Invalid class decisions')
 n=len(y);cm=np.bincount((y*6+p).astype(int),minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(count=n,correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/n) if n else None,macro_f1=float(f.mean()) if n else None,confusion=cm.tolist(),small_cell=n<30,empty_cell=n==0)
def paired(y,old,new,cohort=None):
 y=np.asarray(y);a=np.asarray(old)==y;b=np.asarray(new)==y;require(a.shape==b.shape==y.shape,'Paired lengths differ');c=np.ones(len(y),bool) if cohort is None else np.asarray(cohort,bool);require(c.shape==y.shape,'Cohort length differs');n=len(y);nc=int(c.sum());repair=int((~a&b).sum());damage=int((a&~b).sum())
 return dict(count=n,control_correct=int(a.sum()),new_correct=int(b.sum()),repairs=repair,new_errors=damage,net_repairs=repair-damage,accuracy_delta_pp=100*(repair-damage)/n if n else None,fixed_NOAS30_clean_correct_count=nc,cohort_repairs=int((~a&b&c).sum()),cohort_new_errors=int((a&~b&c).sum()),control_retention=float(a[c].mean()) if nc else None,new_retention=float(b[c].mean()) if nc else None)
def logit_metric(y,z,cohort):
 m=metric(y,z.argmax(1));n=len(y);other=z.copy()
 if n:
  other[np.arange(n),y]=-np.inf;margin=(z[np.arange(n),y]-other.max(1))/SCALE;mean=float(margin.mean());median=float(np.median(margin))
 else:mean=median=None
 nc=int(cohort.sum());retained=int(((z.argmax(1)==y)&cohort).sum());m.update(mean_margin_over30=mean,median_margin_over30=median,fixed_NOAS30_clean_correct_count=nc,fixed_NOAS30_clean_correct_retained=retained,satellite_retention=retained/nc if nc else None);return m

def reconcile_partitions(overall,parts):
 cm=np.asarray(overall['confusion']);require(parts,'Partition cannot be empty');require(np.array_equal(sum((np.asarray(x['confusion']) for x in parts),np.zeros((6,6),int)),cm),'Partition confusion/count does not reconcile');return True

def source_recovery_state(decision,failure,recovery):
 """Reconcile the existing zero-NN report repair; never recalculate criteria."""
 r=recovery or {};d=decision or {}
 checks=dict(status_verified=r.get('status')=='VERIFIED',criteria_unchanged=r.get('criteria_unchanged') is True,source_decision_exact_match=decision is not None and r.get('source_decision')==decision,original_failure_preserved=r.get('original_failure_preserved') is True,zero_cost=all(type(r.get(k)) is int and r[k]==0 for k in ('NN','IQ_reads','checkpoint_reads','updates')),formal_scores_not_read=r.get('formal_scores_read') is False,no_rerun=all(r.get(k) is False for k in ('source_qualification_rerun','training_rerun','formal_rerun')),source_only_decision=d.get('status')=='SOURCE_READOUT_DECIDED' and d.get('formal_scores_read') is False and d.get('target_feedback') is False and d.get('NN')==0 and d.get('model_seed_count')==1)
 verified=failure is not None and all(checks.values())
 return dict(verified=verified,checks=checks,historical_failure=failure,unresolved=failure is not None and not verified)

def stored_source_status(decision,unresolved=False):
 if decision is None:return 'SOURCE_ANALYZED_WITH_STORED_DECISION_PARTIAL'
 require(decision.get('status') in ('SOURCE_READOUT_DECIDED','SOURCE_READOUT_PARTIAL') and decision.get('formal_scores_read') is False and decision.get('target_feedback') is False and decision.get('NN')==0,'Stored source-only decision flags differ')
 if decision['status']=='SOURCE_READOUT_DECIDED':require(decision.get('model_seed_count')==1,'Fixed source model seed count differs')
 return 'SOURCE_ANALYZED' if decision['status']=='SOURCE_READOUT_DECIDED' and not unresolved else 'SOURCE_ANALYZED_WITH_STORED_DECISION_PARTIAL'

def source_analysis(project,specs,out):
 panel=Path(project)/'runs'/PANEL_RUN/'diagnostic';role=read(panel/'data_roles.json');ids=np.asarray(role['physical_ids']);n=1800
 require(role.get('role')=='V' and role.get('counts')=={'L_s':6300,'U_s':56700,'V':27000} and role.get('held_RX') is None and role.get('target_access') is False and role.get('U_truth_used') is False and role.get('updates')==0 and role.get('V_panel')==n and ids.shape==(n,) and len(set(ids.tolist()))==n,'Exact legal source V panel required')
 pc=read(panel/'observations/completion.json');require(pc.get('status')=='SOURCE_OBSERVATIONS_COMPLETE' and pc.get('unique_V')==n and pc.get('draws')==2 and tuple(pc.get('scenes',()))==SCENES and pc.get('conditions',[None])[0]=='full' and pc.get('target_access') is False and pc.get('training_updates')==0,'Original full source condition/draw order required')
 with np.load(panel/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
 for k,x in physics.items():require(x.shape==(2,3,len(pc['conditions']),n),'Source physics shape differs '+k)
 data={};health={};qproof={}
 for spec in specs:
  done=training_complete(spec,project);q=read(spec['root']/'source_qualification/complete.json');require(q.get('status')=='SOURCE_READOUT_COMPLETE' and q.get('config')==done['config'] and q.get('NN')==12600 and all(q.get(k)==0 for k in ('training_updates','head_fit_updates')) and q.get('target_access') is False and q.get('formal_scores_read') is False,'Frozen source readout incomplete')
  with np.load(spec['root']/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in ('ids','TX','RX','day','clean','satellite')}
  require(np.array_equal(x['ids'],ids) and x['clean'].shape==(n,6) and x['satellite'].shape==(2,3,n,6) and np.isfinite(x['clean']).all() and np.isfinite(x['satellite']).all(),'Source panel IDs/logits differ')
  for k,allowed in [('TX',range(6)),('RX',RX),('day',DAYS)]:require(x[k].shape==(n,) and x[k].dtype.kind in 'iu' and set(x[k])==set(allowed),'Source role metadata differs '+k)
  require(abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])<1e-12 and np.allclose((x['satellite'].argmax(-1)==x['TX']).mean(-1),q['source_accuracies'],atol=1e-12,rtol=0),'Source readout does not recount')
  data[spec['key']]=x;health[spec['key']]=done['final_source_metrics'];qproof[spec['key']]=q
 ref=data['NOAS30'];cohort=ref['clean'].argmax(1)==ref['TX'];records=[];effects=[];interactions=[]
 for x in data.values():require(all(np.array_equal(x[k],ref[k]) for k in ('ids','TX','RX','day')),'Matched source physical roles differ')
 for scene in ('clean',*SCENES):
  si=SCENES.index(scene) if scene!='clean' else None
  for draw in ((-1,) if scene=='clean' else (-1,0,1)):
   ds=(0,1) if draw==-1 else (draw,);copies=1 if scene=='clean' else len(ds);y=np.tile(ref['TX'],copies);cc=np.tile(cohort,copies);rx=np.tile(ref['RX'],copies);day=np.tile(ref['day'],copies)
   zz={a:x['clean'] if scene=='clean' else np.concatenate([x['satellite'][d,si] for d in ds]) for a,x in data.items()};masks=[('overall','ALL',np.ones(len(y),bool))]
   for dim,values,allowed in [('RX',rx,RX),('day',day,DAYS),('TX',y,range(6))]:masks.extend((dim,str(v),values==v) for v in allowed)
   if scene!='clean':
    state=np.concatenate([physics['state'][d,si,0] for d in ds]);lock=np.concatenate([physics['lock'][d,si,0] for d in ds]);quality=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in ds])
    require(np.isfinite(state).all() and np.isfinite(lock).all() and np.all((state>=0)&(state<3)&(state==state.astype(int))) and np.all((lock>=0)&(lock<3)&(lock==lock.astype(int))),'State/lock indices differ')
    masks.extend(('state',s,state==i) for i,s in enumerate(STATES));masks.extend(('lock',s,lock==i) for i,s in enumerate(LOCKS));masks.extend(('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(STATES) for j,l in enumerate(LOCKS))
    masks.extend(('signal_to_added_noise_db',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]);masks.append(('signal_to_added_noise_db','MISSING',np.isnan(quality)))
   for dim,label,mask in masks:
    context=dict(scene=scene,draw=draw,dimension=dim,stratum=label);current={}
    for arm,z in zz.items():current[arm]=logit_metric(y[mask],z[mask],cc[mask]);records.append(dict(row_id=arm,model_seed=SEED,**context,**current[arm]))
    for new,control in PAIRS:
     p=paired(y[mask],zz[control][mask].argmax(1),zz[new][mask].argmax(1),cc[mask]);p.update(new=new,control=control,**context,mean_margin_over30_delta=current[new]['mean_margin_over30']-current[control]['mean_margin_over30'] if mask.any() else None);effects.append(p)
    if mask.any():
     interaction=100*interaction_value({arm:current[arm]['accuracy'] for arm in ARMS})
    else:interaction=None
    interactions.append(dict(**context,count=int(mask.sum()),PA_x_As_accuracy_pp=interaction,PA_x_As_macro_f1_pp=100*(current['AS_PAOFF']['macro_f1']-current['NOAS_PAOFF']['macro_f1']-current['AS30']['macro_f1']+current['NOAS30']['macro_f1']) if mask.any() else None))
 decision_path=Path(project)/'runs'/RUN/'source_matrix_decision.json';decision=read(decision_path) if decision_path.is_file() else None
 failure_paths=[s['root']/'source_qualification_failure.json' for s in specs if not s['reused']];stored_failure={p.parent.name:read(p) for p in failure_paths if p.is_file()} or None
 recovery_path=Path(project)/'runs'/RUN/'source_report_recovery.json';stored_recovery=read(recovery_path) if recovery_path.is_file() else None;recovery=source_recovery_state(decision,stored_failure,stored_recovery)
 decision_status=stored_source_status(decision,recovery['unresolved'])
 csv_write(out/'source_strata.csv',({k:v for k,v in r.items() if k!='confusion'} for r in records));write(out/'source_confusions.json',records);csv_write(out/'source_paired_effects.csv',effects);csv_write(out/'source_interaction.csv',interactions)
 summary=dict(status=decision_status,stored_source_qualification_failure=stored_failure,source_qualification_failure_is_historical=recovery['verified'],source_report_recovery_verified=recovery['verified'],source_report_recovery_checks=recovery['checks'],stored_source_report_recovery=stored_recovery,health_V27000=health,readout={k:{a:v for a,v in q.items() if a!='config'} for k,q in qproof.items()},stored_source_decision=decision,source_decision_recomputed=False,source_unique_physical=1800,channel_draws=2,model_seed_count=1,fixed_NOAS30_clean_correct_count=int(cohort.sum()),source_high_urban_available=False,total_SNR_known=False,physical_causality_identified=False,formal_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0)
 write(out/'source_summary.json',summary);return summary,records

def parse_tail(folder,As=True):
 """Read all 2220 actual updates; compare physical loss decomposition, not proxies."""
 folder=Path(folder);result=read(folder/'result.json');done=read(folder/'completion.json');require(result.get('L_NN')==1704960 and result.get('L_As')==(142080 if As else 0) and result.get('optimizer_steps')==2220 and result.get('Ag_packets')==0 and result.get('Ag_control_NN')==568320 and result.get('extra_U_NN')==0 and result.get('U_As')==0,'Actual sixL/As/originalU counters differ')
 sums=defaultdict(Counter);counts=Counter();ordinary=[];max_error=0.;selected=total_u=0
 streams=[lines(folder/f) for f in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  require(all(x is not None for x in items),'Incomplete reference/shape/native/Sinc logs');r,s,n,u=items;local=(i-1)//222+1;ep=local+20;m=n['metrics']
  require(all(x.get('step')==i for x in items) and all(x.get('cumulative_step')==4440+i for x in (r,s,u)) and u.get('physical_clip_calls')==i and n.get('epoch')==local,'Contiguous E21-30 native update clocks differ')
  require(r.get('L_forwards')==6 and r.get('As_packets')==(64 if As else 0) and r.get('Ag_packets')==0 and r.get('Ag_mode')=='NONE' and r.get('extra_control_packets')==128 and r.get('ordinary_redraws')==0 and r.get('same_IQ_each_pair') is True and r.get('all_satellite_L_supervised') is True,'Actual sixL/sameIQ/ungated labeled As differs')
  require(s.get('mode')=='NONE' and s.get('NN')==256 and s.get('packets')==128 and s.get('applied')==0 and s.get('weight')==.25 and s.get('same_IQ_each_pair') is True and not s.get('new_boundary_samples') and not s.get('relative_noise_RMS'),'Unchanged raw control slot differs')
  for k in ('common_CE','common_KL','extra_CE','extra_KL','SupCon','SupCon_weight','total_L','base_L_before_shape','ordinary_raw_MSE'):finite(r[k],k)
  for k in ('shape_CE','shape_KL','shape_RD'):finite(s[k],k)
  loss=.5*(r['common_CE']+r['common_KL'])+.5*(r['extra_CE']+r['extra_KL'])+.1*r['SupCon']+.25*s['shape_RD'];error=abs(loss-r['total_L']);max_error=max(max_error,error)
  require(r['SupCon_weight']==.1 and abs(s['shape_RD']-s['shape_CE']-s['shape_KL'])<3e-5 and error<3e-5 and abs(r['total_L']-r['base_L_before_shape']-.25*s['shape_RD'])<1e-6,'Original normalized total1.25/CE-KL-SupCon loss differs')
  finite(m['train/loss'],'native training loss');require(not any(m.get(k) or 0 for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad','train/pseudo_truth_available')),'Invalid or U-truth-exposed update')
  tu=int(m.get('train/pseudo_total') or 0);su=int(m.get('train/pseudo_selected') or 0);require(0<=su<=tu<=256 and tu>0,'Old U sample counts differ');total_u+=tu;selected+=su
  details=r['SupCon_details'];require(r.get('crossview_mode')=='ORD128' and not r.get('new_losses') and not r.get('extra_U_NN') and len(details)==4,'Unchanged ordinary SupCon evidence required')
  for q in details:require(q.get('mode')=='ORD128' and q.get('total_anchors')==128 and q.get('ordinary_anchors')==128 and q.get('additional_anchors')==0 and q.get('same_RX_positive_pairs')==0 and q.get('true_satellite_anchors')==0,'Original crossRX SupCon pairing differs')
  require({(q['phase'],q['level']) for q in details}=={(p,l) for p in ('first','second') for l in ('time_down','t2.pool')},'Original SupCon layers/passes differ')
  for k in ('common_CE','common_KL','extra_CE','extra_KL','SupCon','total_L'):sums[local][k]+=r[k]
  for k in ('shape_CE','shape_KL','shape_RD'):sums[local][k]+=s[k]
  sums[local]['native_loss']+=m['train/loss'];counts[local]+=1;ordinary.append(dict(step=4440+i,epoch=ep,MSE=r['ordinary_raw_MSE']))
 require(sum(counts.values())==2220 and set(counts)==set(range(1,11)) and set(counts.values())=={222} and total_u==567000,'Complete 2220-update/tenU-epoch tail required')
 raw=list(lines(folder/'metrics_epoch.jsonl'));require([x['epoch'] for x in raw]==list(range(1,11)),'Full native epoch telemetry required')
 epochs=list(lines(folder/'epoch_metrics.jsonl'));require([x['epoch'] for x in epochs]==list(range(1,11)),'All actual epoch summaries required')
 curves=[]
 for e in epochs:
  local=e['epoch'];q=dict(epoch=local+20,local_epoch=local,**{k:v/222 for k,v in sums[local].items()})
  for key,outkey in [('lr','lr'),('val_tx_acc','source_val_accuracy_pct'),('train_grad_backbone','grad_backbone'),('epoch_time_s','epoch_wall_seconds')]:q[outkey]=finite(e[key],key)
  curves.append(q)
 lr=[x['lr'] for x in curves];require(abs(lr[0]-.0002)<1e-15 and abs(lr[-1]-.000001)<1e-15 and np.all(np.diff(lr)<=1e-15),'Actual reset cosine10 calendar differs')
 return dict(status='FULL_TAIL_LOGS_VERIFIED',updates=2220,epochs=10,actual_L_NN=1704960,actual_As=142080 if As else 0,old_U_total=total_u,old_U_selected=selected,maximum_L_decomposition_error=max_error,teacher_offset_not_added=True,curve=curves,ordinary=ordinary,cost=native_cost(folder))
def native_cost(folder):
 r=read(Path(folder)/'result.json');c={k:r[k] for k in COUNTERS};require(all(type(v) is int and v>=0 for v in c.values()),'Actual native counters required');return dict(**c,native_NN=sum(c.values()),wall_seconds=float(r['wall_seconds']))
def epoch_curve(folder,offset):
 folder=Path(folder);sums=Counter();counts=Counter()
 for step in lines(folder/'step_metrics.jsonl.gz'):
  e=step['epoch'];sums[e]+=finite(step['metrics']['train/loss'],'native loss');counts[e]+=1
 records=list(lines(folder/'epoch_metrics.jsonl'));require([x['epoch'] for x in records]==list(range(1,len(records)+1)) and all(counts[x['epoch']]==222 for x in records),'Complete reference epoch/step curves required')
 return [dict(epoch=x['epoch']+offset,source_val_accuracy_pct=x['val_tx_acc'],native_loss=sums[x['epoch']]/222,epoch_wall_seconds=x['epoch_time_s'],lr=x['lr'],grad_backbone=x['train_grad_backbone']) for x in records]
def PA_shadow_audit(folder,result):
 records=list(lines(Path(folder)/'PA_shadow_steps.jsonl.gz'));require(len(records)==2220,'All ordinary L PA shadow records required');totals=Counter()
 for i,r in enumerate(records,1):
  require(r.get('step')==i and r.get('cumulative_step')==4440+i and r.get('count')==128 and r.get('actual_PA_injected_packets')==0 and r.get('exact_native_nonPA_prefix') is True and r.get('same_native_draw') is True and r.get('scope')=='ordinary_L_only' and r.get('U_changed') is False and r.get('As_changed') is False,'Actual ordinary-only shadow bypass differs')
  require(r.get('shadow_PA_calls') in (0,1) and 0<=r['shadow_PA_changed_packets']<=r['shadow_PA_selected_packets']<=128 and r.get('effective_p_pa',0)>0 and r.get('effective_p_dac')==0 and r.get('native_no_pa') is False and r.get('native_class_signature') is False,'Native shadow selection or PA schedule differs')
  for k in ('shadow_PA_MSE_sum','shadow_PA_strength_sum','prefix_raw_MSE'):finite(r[k],k)
  for k in ('shadow_PA_calls','shadow_PA_selected_packets','shadow_PA_changed_packets','shadow_PA_MSE_sum','actual_PA_injected_packets'):totals[k]+=r[k]
 summary=result['PA_shadow'];require(summary.get('ordinary_L_calls')==2220 and summary.get('ordinary_L_packets')==284160 and summary.get('full_native_draws') is True and summary.get('parent_E1_E20_PA') is True,'Actual PA shadow summary scope differs')
 for k,v in totals.items():require(math.isclose(v,summary[k],abs_tol=1e-10,rel_tol=1e-12),'PA per-step/summary recount differs '+k)
 return dict(summary=summary,recount=dict(totals),steps=2220,neural_forwards_added=0),records

def training_analysis(project,specs,out):
 details={};curves=[];coverage=[];shadow_rows=[];mse_rows=[]
 for spec in specs:
  if spec['reused']:
   curves.extend(dict(row_id=spec['key'],reused=True,model_seed=SEED,**x) for x in epoch_curve(spec['root']/'stage20',0)+epoch_curve(spec['root']/'source',20));continue
  As=spec['key']=='AS_PAOFF';folder=spec['root']/'source';result=read(folder/'result.json');tail=parse_tail(folder,As);original=Path(project)/'runs'/REF/(spec['reference_arm']+'-s'+str(SEED));oldtail=parse_tail(original/'source',As);seq_n=seq_bad=0
  for a,b in zip_longest(lines(folder/'training_sequence.jsonl.gz'),lines(original/'source/training_sequence.jsonl.gz')):seq_n+=1;seq_bad+=a!=b
  require(seq_n>0 and seq_bad==0,'Corresponding original tail physical source L/U sequence differs')
  ordinary=tail.pop('ordinary');previous=oldtail.pop('ordinary');require(len(ordinary)==len(previous)==2220,'Both complete ordinary MSE series required')
  deltas=[]
  for a,b in zip(ordinary,previous):
   require(a['step']==b['step'] and a['epoch']==b['epoch'],'Same ordinary MSE clocks required');delta=a['MSE']-b['MSE'];deltas.append(delta);mse_rows.append(dict(row_id=spec['key'],**a,original_MSE=b['MSE'],MSE_delta=delta))
  pa,pr=PA_shadow_audit(folder,result);shadow_rows.extend(dict(row_id=spec['key'],**r) for r in pr)
  for r in result['TX_RX_scene_state_lock']:coverage.append(dict(row_id=spec['key'],actual_channel_generated=As,coverage_kind='actual_As' if As else 'identity_control',**r))
  if As:
   channel=result['L_channel'];expected_sessions={f'rx{rx}:day{day}:eq1' for rx in RX for day in DAYS};require(channel.get('session_count')==15 and set(channel.get('sessions',{}))==expected_sessions,'Actual As hardware must cover all15 RX/day sessions without TX-defined sessions')
   observed=result['TX_RX_scene_state_lock'];require(sum(r['count'] for r in observed)==142080,'Actual labeled As generation count differs')
   require(all(r['TX'] in range(6) and r['RX'] in RX and r['scene'] in SCENES and r['state'] in STATES and r['lock'] in LOCKS and r['count']>0 for r in observed),'Physical coverage schema differs')
   for tx in range(6):
    for rx in RX:
     for scene in SCENES:require(any(r['TX']==tx and r['RX']==rx and r['scene']==scene for r in observed),'Missing TX x source RX x scene support')
   require(all(any(r['state']==s for r in observed) for s in STATES) and all(any(r['lock']==l for r in observed) for l in LOCKS),'All original state/lock support including unlocked required')
  tail.update(PA_shadow=pa,actual_As_hardware_session_count=result['L_channel'].get('session_count') if As else None,identity_control_is_not_channel=not As,physical_sequence_records=seq_n,physical_sequence_mismatch=seq_bad,ordinary_MSE_comparison=dict(records=2220,changed_steps=int(np.count_nonzero(deltas)),mean_delta=float(np.mean(deltas)),max_abs_delta=float(np.max(np.abs(deltas))),equality_expected=False),actual_L_channel=result['L_channel'])
  details[spec['key']]=tail;curves.extend(dict(row_id=spec['key'],reused=False,model_seed=SEED,**r) for r in tail['curve'])
 csv_write(out/'training_curves.csv',curves);csv_write(out/'training_coverage.csv',coverage);csv_write(out/'PA_shadow_steps.csv',shadow_rows);csv_write(out/'ordinary_MSE_comparison.csv',mse_rows);write(out/'new_tail_training.json',details);return details,curves

def formal_analysis(project,specs,out,truth):
 viewroot=Path(project)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views';manifest=read(viewroot/'manifest.json');require(manifest.get('status')=='VALIDATED_ONCE' and manifest.get('count')==COUNT and manifest.get('scenes')==list(VIEWS[1:]) and manifest.get('channel')=='residual/post_sync/noeq' and manifest.get('truth_read') is False and manifest.get('source_target_disjoint') is True and manifest.get('rows_share_observations') is True,'Fixed formal received observations differ')
 with np.load(viewroot/'index.npz',allow_pickle=False) as f:ids=f['ids'].copy()
 data,meta=truth_last_arrays(specs,ids,truth,project);y=meta['y'];rx=meta['rx'];day=meta['day'];cohort=data['NOAS30']['pred']['clean']==y;records=[];effects=[];interactions=[];summary={};partitions=0
 for spec in specs:
  arm=spec['key'];saved=read(spec['root']/'scoring/scores.json');require(saved.get('truth_last') is True and saved.get('target_feedback_forbidden') is True,'Saved independent scorer flags differ');summary[arm]={}
  for view in VIEWS:
   p=data[arm]['pred'][view];z=data[arm]['logits'][view];current=[]
   masks=[('overall','ALL',np.ones(COUNT,bool))]+[('receiver',r,rx==r) for r in sorted(set(rx))]+[('day',d,day==d) for d in sorted(set(day))]+[('transmitter',str(t),y==t) for t in range(6)]
   for dim,label,mask in masks:
    m=logit_metric(y[mask],z[mask],cohort[mask]);r=dict(row_id=arm,model_seed=SEED,reused=spec['reused'],view=view,dimension=dim,stratum=str(label),**m);current.append(r);records.append(r)
    old=next(x for x in saved['results'] if x['view']==view and x['dimension']==dim and x['stratum']==str(label));require(m['confusion']==old['confusion'] and m['count']==old['query_count'] and abs(m['accuracy']-old['accuracy'])<1e-12 and abs(m['macro_f1']-old['macro_f1'])<1e-12,'Independent formal CM/metric recount differs')
   overall=current[0]
   for dim in ('receiver','day','transmitter'):reconcile_partitions(overall,[r for r in current if r['dimension']==dim]);partitions+=1
   summary[arm][view]=dict(accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_RX=min(r['accuracy'] for r in current if r['dimension']=='receiver'),mean_margin_over30=overall['mean_margin_over30'],satellite_retention=overall['satellite_retention'])
 for view in VIEWS:
  for new,control in PAIRS:
   for dim,label,mask in [('overall','ALL',np.ones(COUNT,bool))]+[('receiver',r,rx==r) for r in sorted(set(rx))]+[('day',d,day==d) for d in sorted(set(day))]+[('transmitter',str(t),y==t) for t in range(6)]:effects.append(dict(new=new,control=control,view=view,dimension=dim,stratum=str(label),**paired(y[mask],data[control]['pred'][view][mask],data[new]['pred'][view][mask],cohort[mask])))
 for v in VIEWS:
  contexts=[r for r in records if r['row_id']=='NOAS30' and r['view']==v]
  for ref in contexts:
   match={arm:next(r for r in records if r['row_id']==arm and r['view']==v and r['dimension']==ref['dimension'] and r['stratum']==ref['stratum']) for arm in ARMS}
   value=lambda k:interaction_value({arm:match[arm][k] for arm in ARMS})
   interactions.append(dict(view=v,dimension=ref['dimension'],stratum=ref['stratum'],count=ref['count'],PA_x_As_net_correct=value('correct'),PA_x_As_accuracy_pp=100*value('accuracy'),PA_x_As_macro_f1_pp=100*value('macro_f1'),PA_x_As_mean_margin_over30=value('mean_margin_over30')))
 csv_write(out/'formal_interaction.csv',interactions)
 csv_write(out/'formal_metrics.csv',({k:v for k,v in r.items() if k!='confusion'} for r in records));write(out/'formal_confusions.json',records);csv_write(out/'formal_paired_effects.csv',effects)
 clean=100*(summary['AS_PAOFF']['clean']['accuracy']-summary['NOAS_PAOFF']['clean']['accuracy']);urban=100*np.mean([summary['AS_PAOFF'][v]['accuracy']-summary['NOAS_PAOFF'][v]['accuracy'] for v in VIEWS[-3:]])
 result=dict(status='FORMAL_ANALYZED',rows=summary,independent_confusion_recounts=len(records),partition_reconciliations=partitions,all4_rows_all7_verified_before_truth=True,formal_state_labels_available=False,fixed_NOAS30_clean_correct_count=int(cohort.sum()),stage_clean_delta_pp=clean,stage_clean_max_drop_pp=.1,stage_clean_within_0p1pp=clean>=-.1-1e-10,urban_mean_vs_matched_PAOFF_NoAs_pp=float(urban),stage_clean_reference='NOAS_PAOFF',urban_mean_vs_As30_pp=float(100*np.mean([summary['AS_PAOFF'][v]['accuracy']-summary['AS30'][v]['accuracy'] for v in VIEWS[-3:]])),single_seed_observed_dual_gain=clean>0 and urban>0,model_seed_count=1,reproducible_dual_gain=False,target_feedback=False,NN=0,IQ_reads=0,checkpoint_reads=0)
 write(out/'formal_summary.json',result);return result

def cost_analysis(project,specs,source):
 refs={};new={};runs=Path(project)/'runs'
 for spec in specs:
  if spec['reused']:
   refs[spec['key']]={stage:native_cost(spec['root']/stage) for stage in ('stage20','source')};refs[spec['key']]['source_readout_NN']=read(spec['root']/'source_qualification/complete.json')['NN'];refs[spec['key']]['formal_prediction_NN']=read(spec['root']/'prediction/complete.json')['actual_NN'];continue
  native=native_cost(spec['root']/'source');pred=read(spec['root']/'prediction/complete.json');qp=spec['root']/'source_qualification/complete.json';q=read(qp) if qp.is_file() else None
  qnn=q['NN'] if q and q.get('status')=='SOURCE_READOUT_COMPLETE' else None;parent=native_cost(runs/REF/(spec['reference_arm']+'-s'+str(SEED))/'stage20')
  new[spec['key']]=dict(new10=native,inherited20=parent,source_readout_NN=qnn,formal_prediction_NN=pred['actual_NN'],formal_prediction_wall_seconds=pred['wall_seconds'],new_remote_NN_known=native['native_NN']+(qnn or 0)+pred['actual_NN'],new_remote_NN_complete=qnn is not None,candidate_lifetime_NN_known=parent['native_NN']+native['native_NN']+(qnn or 0)+pred['actual_NN'])
 smoke=read(runs/RUN/'real_checkpoint_smoke.json');require(smoke.get('status')=='PASS' and smoke.get('NN')==4 and len(smoke.get('rows',[]))==2,'Actual compound parent smoke differs')
 historical=sum(row[stage]['native_NN'] for row in refs.values() for stage in ('stage20','source'))+sum(row['source_readout_NN']+row['formal_prediction_NN'] for row in refs.values());new_total=sum(row['new_remote_NN_known'] for row in new.values())+smoke['NN']
 return dict(new_rows=new,new_model_rows=2,new_model_epochs=20,new_updates=4440,new_L_NN=3409920,new_As_generated=142080,parent_epochs_per_row=20,parent_updates_per_row=4440,candidate_cumulative_epochs_per_row=30,candidate_cumulative_updates_per_row=6660,new_source_readout_NN=sum(row['source_readout_NN'] or 0 for row in new.values()),new_formal_prediction_NN=sum(row['formal_prediction_NN'] for row in new.values()),new_remote_smoke_NN=smoke['NN'],new_total_remote_NN_known=new_total,new_total_remote_NN_complete=all(row['new_remote_NN_complete'] for row in new.values()),historical_references=refs,historical_reference_NN_known=historical,historical_reference_model_epochs=60,historical_reference_updates=13320,unique_total_model_epochs_including_reused=80,unique_total_updates_including_reused=17760,unique_total_remote_NN_known=new_total+historical,shared_own_As20_and_NoAs20_parents_counted_once=True,historical_shared_sixrow_smoke_NN_unallocated=True,local_synthetic_NN_known=5376,local_synthetic_successful_NN=4608,local_synthetic_failed_NN=768,local_synthetic_scope='Implementation plus one independent reviewer; separate from remote run; failed toy assertion768 retained',reference_reprediction=0,reference_rescoring=0,analysis_NN=0,budget_exhaustion_is_not_convergence=True)

def plots(out,training,source_rows,formal):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'NOAS30':'#94a3b8','AS30':'#d97706','NOAS_PAOFF':'#475569','AS_PAOFF':'#2563eb'};files=[]
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):p=out/(name+'.'+ext);fig.savefig(p,dpi=180);files.append(str(p))
  plt.close(fig)
 if training:
  fig,axes=plt.subplots(1,3,figsize=(15,4.7))
  for arm,color in colors.items():
   c=[r for r in training if r['row_id']==arm]
   for ax,key,title in zip(axes,('native_loss','source_val_accuracy_pct','grad_backbone'),('Logged native training loss','Logged source V TX accuracy (%)','Backbone gradient norm')):
    ax.plot([r['epoch'] for r in c],[r[key] for r in c],label=arm,color=color,marker='.',markersize=3);ax.set(title=title,xlabel='Cumulative epoch');ax.grid(alpha=.2);ax.axvline(20.5,color='grey',ls='--',alpha=.4)
  axes[0].legend();fig.suptitle('All recorded epochs; new PAOFF tails only E21-30; no imputed connecting points');save(fig,'training_curves')
 if source_rows:
  for scene in SCENES[1:]:
   fig,axes=plt.subplots(1,2,figsize=(15,5))
   for ax,dimension in zip(axes,('state_lock','signal_to_added_noise_db')):
    rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dimension];labels=list(dict.fromkeys(r['stratum'] for r in rr));xx=np.arange(len(labels));ns=[]
    for i,(arm,color) in enumerate(colors.items()):
     seq=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels];vals=[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in seq];ax.bar(xx+(i-1.5)*.20,vals,.20,label=arm,color=color)
     if not ns:ns=[r['count'] for r in seq]
    ax.set(xticks=xx,xticklabels=[l+'\nn='+str(n) for l,n in zip(labels,ns)],ylim=(0,104),ylabel='Source accuracy (%)',title=dimension);ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2)
   axes[0].legend();fig.suptitle(scene+'; two repeated source draws, one model seed; empty cells retained');save(fig,scene+'_source_states')
 if formal:
  fig,ax=plt.subplots(figsize=(12,5));xx=np.arange(7)
  for i,(arm,color) in enumerate(colors.items()):ax.bar(xx+(i-1.5)*.20,[100*formal['rows'][arm][v]['accuracy'] for v in VIEWS],.20,label=arm,color=color)
  ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urban','mid urban','low urban'],ylim=(0,103),ylabel='Formal accuracy (%)',title='Fixed received IQ / joint default cosine head / matched E30 / single seed');ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'formal_accuracy')
 if formal:
  fig,axes=plt.subplots(1,2,figsize=(15,4.8));xx=np.arange(7);rows=formal['rows']
  for new,control in PAIRS[:3]:axes[0].plot(xx,[100*(rows[new][v]['accuracy']-rows[control][v]['accuracy']) for v in VIEWS],marker='o',label=new+' - '+control)
  axes[1].bar(xx,[100*(rows['AS_PAOFF'][v]['accuracy']-rows['NOAS_PAOFF'][v]['accuracy']-rows['AS30'][v]['accuracy']+rows['NOAS30'][v]['accuracy']) for v in VIEWS],color='#2563eb')
  for ax,title in zip(axes,('Paired accuracy changes','PA-return x As accuracy interaction')):ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],ylabel='Absolute percentage points',title=title);ax.axhline(0,color='grey',linewidth=.7);ax.grid(axis='y',alpha=.2)
  axes[0].legend(fontsize=8);fig.suptitle('Fixed E30/default head; one model seed; no reproducibility claim');save(fig,'formal_pair_changes')
 return files

def report(out,result):
 formal=result['formal'];rows=formal['rows'];cost=result['cost'];text=['# P60: ordinary-L PA waveform return ablation','', 'Only the E21-30 ordinary-L PA waveform is bypassed after complete native shadow sampling. Each new row inherits its corresponding original P43 scratch20 with PA intact. Models, raw slots, nominal practical channel, U, joint default head and loss scale are unchanged.','', '| View | NoAs30 % | As30 % | PAOFF NoAs % | PAOFF As % | Matched As pp | PAOFF As−As30 pp | PAOFF NoAs−NoAs30 pp | PA×As pp |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for v in VIEWS:
  a,b,c,d=[rows[k][v]['accuracy'] for k in ARMS];text.append(f'| {v} | {100*a:.6f} | {100*b:.6f} | {100*c:.6f} | {100*d:.6f} | {100*(d-c):+.6f} | {100*(d-b):+.6f} | {100*(c-a):+.6f} | {100*(d-c-b+a):+.6f} |')
 text += ['', '| View | PAOFF As macro F1 % | PAOFF As worst RX % | Mean default margin/30 | Fixed original NoAs30 clean-correct retention % |','|---|---:|---:|---:|---:|']
 for v in VIEWS:
  r=rows['AS_PAOFF'][v];text.append(f"| {v} | {100*r['macro_f1']:.6f} | {100*r['worst_RX']:.6f} | {r['mean_margin_over30']:.8f} | {100*r['satellite_retention']:.6f} |")
 text += ['', f"Matched clean PAOFF As−PAOFF NoAs: {formal['stage_clean_delta_pp']:+.6f} absolute pp; ≤0.1 pp cost stage: {formal['stage_clean_within_0p1pp']}.",f"Matched urban mean gain: {formal['urban_mean_vs_matched_PAOFF_NoAs_pp']:+.6f} pp; PAOFF As−original As30 urban mean: {formal['urban_mean_vs_As30_pp']:+.6f} pp.",'','The source_matrix_decision.json is read unchanged; scientific failure still receives all seven formal views. One model seed cannot establish repeatable dual improvement. Source V27000 health, V1800 panel, two repeated draws and formal target are distinct scopes. No formal per-packet state labels exist; source state associations do not identify a physical cause.']
 if result.get('source'):text += ['', 'Stored source decision: '+json.dumps(result['source']['stored_source_decision'],ensure_ascii=False)]
 text += ['', 'Costs: two new10 tails /4440 updates; original two E30 models /13320 updates already include both shared own20 parents. Unique training represented is80 model epochs /17760 updates, with no parent double charge.',f"New remote NN known: {cost['new_total_remote_NN_known']}; new L_NN3409920, As generated142080; source readout NN{cost['new_source_readout_NN']}; formal prediction NN{cost['new_formal_prediction_NN']}; remote real-parent smoke NN{cost['new_remote_smoke_NN']}. Local synthetic successful4608 + failed768 =5376 NN separately: implementation3840 and one independent reviewer1536.",'Native student/teacher train/eval counters, actual per-row wall times, inherited costs, PA shadow selected/changed versus injected0, physical sequence and ordinary MSE differences are stored separately. Budget exhaustion is not convergence.','', 'Missing/partial components: '+json.dumps(result['missing'],ensure_ascii=False),'', 'Figures:']
 text += ['- '+str(p) for p in result['figures']];text += ['','All four-row accuracy/F1/CM, RX/day/TX, matched repairs/new errors, interaction, margins/30 and fixed original NoAs30 clean-correct retention are in accompanying CSV/JSON. Empty and small source cells are retained. No model inference, optimizer/head update, IQ/checkpoint read, new prediction or scorer process was performed by this CPU analysis.']
 (out/'final_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def run(project,out,truth,code_commit=None):
 project=Path(project);out=Path(out);require(not out.exists(),'Immutable new analysis output required; reconcile existing directory');out.mkdir(parents=False);start=time.perf_counter();specs=row_specs(project);missing=[];source=None;sr=[];tr=None;curves=[]
 # Source stages run first. Their technical partial state does not skip legal formal closure.
 for name,fn in [('training',lambda:training_analysis(project,specs,out)),('source',lambda:source_analysis(project,specs,out))]:
  try:
   value=fn()
   if name=='training':tr,curves=value
   else:
    source,sr=value
    if source['status']!='SOURCE_ANALYZED':missing.append(dict(component='stored_source_decision',error='Existing source decision/failure partial; not recomputed',stored_failure=source['stored_source_qualification_failure']))
  except Exception as error:
   failure=dict(component=name,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True);missing.append(failure);write(out/(name+'_failure.json'),failure)
 formal=formal_analysis(project,specs,out,truth);cost=cost_analysis(project,specs,source);write(out/'costs.json',cost)
 try:figures=plots(out,curves,sr,formal)
 except Exception as error:figures=[];failure=dict(component='plot',error=repr(error),traceback=traceback.format_exc());missing.append(failure);write(out/'plot_failure.json',failure)
 result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,code_commit=code_commit,source=source,formal=formal,cost=cost,missing=missing,figures=figures,training=tr,model_seed_count=1,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,source_decision_recomputed=False,target_feedback=False,crossseed_confirmed=False,goal_complete=False,wall_seconds=time.perf_counter()-start)
 report(out,result);write(out/'complete.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--code-commit');a=p.parse_args();r=run(a.project,a.out,a.truth,a.code_commit);print(json.dumps(dict(status=r['status'],out=str(a.out),missing_components=[x['component'] for x in r['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)
if __name__=='__main__':main()
