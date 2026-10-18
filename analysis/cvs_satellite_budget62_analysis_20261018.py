"""P62 eight-row CPU recount; no Torch/model/IQ access or new scoring launch."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse,csv,gzip,json,math,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
RUN='20261018-phase1-FcrossRX-satelliteDropout-budget-e50-s2026092701-m2-r01'
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
SAT30='20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
BUDGET50='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
ARMS=('NOAS30','RD30','CE30','DET30','NOAS50','RD50','CE50','DET50')
PAIRS=(('DET50','CE50'),('DET30','CE30'),('DET50','DET30'),('CE50','CE30'),('RD50','RD30'),('NOAS50','NOAS30'),('DET50','NOAS50'),('DET50','RD50'),('CE50','NOAS50'),('CE50','RD50'),('DET30','NOAS30'),('CE30','NOAS30'),('RD50','NOAS50'),('RD30','NOAS30'))
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
 for method,arm,run30 in [('NOAS','B00',REF),('RD','B01',REF),('CE','AS_CE',SAT30),('DET','AS_DET',SAT30)]:
  for budget in (30,50):
   run=run30 if budget==30 else BUDGET50 if method in ('NOAS','RD') else RUN;root=p/run/(arm+'-s'+str(SEED));parent30=p/run30/(arm+'-s'+str(SEED))
   result.append(dict(key=method+str(budget),method=method,budget=budget,run=run,row_id=root.name,arm=arm,root=root,parent30=parent30,reused=run!=RUN))
 return sorted(result,key=lambda r:ARMS.index(r['key']))

def lineage_segments(spec):
 p=spec['parent30'];result=[(p/'stage20',20,13,0,None),(p/'source',10,0,20,p/'stage20')]
 if spec['budget']==50:result.append((spec['root']/'source',20,0,30,p/'source'))
 return result

def training_complete(spec,project):
 roles=read(Path(project)/'runs'/CONTRACT_RUN/'source_contract.json')['role_ids']
 for folder,epochs,labels,offset,parent in lineage_segments(spec):
  done=read(folder/'completion.json');c=done['config'];frozen=read(folder/'source_frozen.json');init=read(folder/'initialization.json');r=read(folder/'result.json');expected=[] if parent is None else [str(parent/'final_ssdg.pth')]
  require((done.get('epoch'),done.get('optimizer_steps'),c.get('epochs'),c.get('label_epochs'),c.get('pseudo_epochs'),c.get('updates'),c.get('epoch_offset'),c.get('step_offset'))==(epochs,epochs*222,epochs,labels,epochs-labels,epochs*222,offset,offset*222),'Actual own20/30/50 calendar differs')
  require(c.get('run_id')==folder.parent.parent.name and c.get('row_id')==folder.parent.name and c.get('model_seed')==SEED,'Actual lineage run/row/seed differs')
  require(done.get('target_access') is False and done.get('head_fit_updates')==0 and done.get('native_fitted_probes_disabled') is True,'Default joint head/source-only completion required')
  require(frozen.get('config')==c and frozen.get('selection')=='final_only' and frozen.get('target_access') is False and r.get('config')==c,'Frozen own final config differs')
  require(init.get('checkpoint_sources')==expected and init.get('scratch_only')==(parent is None) and init.get('target_contact') is False and init.get('source_roles')=='EXACT_MATCH' and c.get('checkpoint_sources')==expected,'Corresponding own student/EMA parent differs')
  require(read(folder/'source_contract.json')['role_ids']==roles,'Actual L/U/V roles differ')
  require(c.get('new_Ag') is False and c.get('Ag_mode')=='NONE' and c.get('L_NN_per_step')==768 and c.get('total_L_loss_scale')==1.25 and c.get('As')==(spec['method']!='NOAS') and r.get('source_RX')==list(RX) and r.get('source_RX_holdout') is None and all(r.get(k)==0 for k in ('head_fit_updates','U_truth_read','target_access','extra_U_NN','U_As')),'All-source fixed sixL/defaultU contract differs')
  require(done['final_source_metrics'].get('source_val_count')==27000,'Full source V27000 required')
  if spec['method'] in ('CE','DET'):require(c.get('satellite_mode')==spec['arm'],'Actual satellite CE/dropout mode differs')
 return read(spec['root']/'source/completion.json')

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
 truth=read(truth_path) # Deliberately after the entire eight-row gate.
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
 require(set(values)=={'DET50','CE50','DET30','CE30'},'All four CE/DET budget cells required')
 return values['DET50']-values['CE50']-values['DET30']+values['CE30']

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
 status=decision.get('status');require(status in ('SOURCE_BUDGET_READOUT_COMPLETE','SOURCE_BUDGET_READOUT_PARTIAL') and decision.get('formal_scores_read') is False and decision.get('target_scores_read') is False and decision.get('target_feedback') is False and type(decision.get('NN')) is int and decision['NN']==0 and decision.get('model_seed_count')==1 and decision.get('automatic_seed_extension') is False and decision.get('scientific_failure_skips_formal') is False,'Actual saved P62 source-only budget decision schema differs')
 if status=='SOURCE_BUDGET_READOUT_COMPLETE':require(set(decision.get('rows',{}))==set(ARMS),'Eight logical source rows required')
 else:require(decision.get('source_qualified') is False,'Partial source cannot be qualified')
 return 'SOURCE_ANALYZED' if status=='SOURCE_BUDGET_READOUT_COMPLETE' and not unresolved else 'SOURCE_ANALYZED_WITH_STORED_DECISION_PARTIAL'

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
     interaction=100*interaction_value({arm:current[arm]['accuracy'] for arm in ('DET50','CE50','DET30','CE30')})
    else:interaction=None
    interactions.append(dict(**context,count=int(mask.sum()),budget_x_DET_accuracy_pp=interaction,budget_x_DET_macro_f1_pp=100*(current['DET50']['macro_f1']-current['CE50']['macro_f1']-current['DET30']['macro_f1']+current['CE30']['macro_f1']) if mask.any() else None))
 decision_path=Path(project)/'runs'/RUN/'source_matrix_decision.json';decision=read(decision_path) if decision_path.is_file() else None
 failure_paths=[s['root']/'source_qualification_failure.json' for s in specs if not s['reused']];stored_failure={p.parent.name:read(p) for p in failure_paths if p.is_file()} or None
 recovery_path=Path(project)/'runs'/RUN/'source_report_recovery.json';stored_recovery=read(recovery_path) if recovery_path.is_file() else None;recovery=source_recovery_state(decision,stored_failure,stored_recovery)
 decision_status=stored_source_status(decision,recovery['unresolved'])
 csv_write(out/'source_strata.csv',({k:v for k,v in r.items() if k!='confusion'} for r in records));write(out/'source_confusions.json',records);csv_write(out/'source_paired_effects.csv',effects);csv_write(out/'source_interaction.csv',interactions)
 summary=dict(status=decision_status,stored_source_qualification_failure=stored_failure,source_qualification_failure_is_historical=recovery['verified'],source_report_recovery_verified=recovery['verified'],source_report_recovery_checks=recovery['checks'],stored_source_report_recovery=stored_recovery,health_V27000=health,readout={k:{a:v for a,v in q.items() if a!='config'} for k,q in qproof.items()},stored_source_decision=decision,source_decision_recomputed=False,source_unique_physical=1800,channel_draws=2,model_seed_count=1,fixed_NOAS30_clean_correct_count=int(cohort.sum()),source_high_urban_available=False,total_SNR_known=False,physical_causality_identified=False,formal_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0)
 write(out/'source_summary.json',summary);return summary,records

def satellite_record(q,mode,i,cumulative,inventory):
 require(q.get('mode')==mode and q.get('step')==i and q.get('cumulative_step')==cumulative and q.get('As_supervised')==64 and q.get('extra_NN')==0 and q.get('parameter_updates_extra')==0,'Actual satellite scope/step/count differs')
 calls=2*sum(v['p']>0 for v in inventory);require(calls>0 and q.get('dropout_calls')==calls,'Existing native dropout mask draw count differs')
 require(type(q.get('bypassed_elements')) is int and (q['bypassed_elements']==0 if mode=='AS_CE' else q['bypassed_elements']>0),'Only AS_DET bypasses real satellite elements')
 for k in ('original_mixed_KL','ordinary_half_KL','satellite_half_KL','used_mixed_KL','removed_satellite_KL','satellite_pair_logit_maxdiff'):finite(q[k],k)
 differences=[q['original_mixed_KL']-.5*(q['ordinary_half_KL']+q['satellite_half_KL']),q['used_mixed_KL']-.5*q['ordinary_half_KL'],q['removed_satellite_KL']-.5*q['satellite_half_KL'],q['original_mixed_KL']-q['used_mixed_KL']-q['removed_satellite_KL']]
 require(max(map(abs,differences))<3e-5,'Actual satellite KL removal decomposition differs')
 require(q['satellite_pair_logit_maxdiff']>=0 and (mode!='AS_DET' or q['satellite_pair_logit_maxdiff']<=1e-5 and abs(q['satellite_half_KL'])<=1e-6),'AS_DET satellite pair remains stochastic')
 return max(map(abs,differences))

def loss_decomposition(r,s):
 expected=.5*(r['common_CE']+r['common_KL'])+.5*(r['extra_CE']+r['extra_KL'])+.1*r['SupCon']+.25*s['shape_RD']
 require(abs(s['shape_RD']-s['shape_CE']-s['shape_KL'])<3e-5 and abs(r['total_L']-r['base_L_before_shape']-.25*s['shape_RD'])<1e-6 and abs(expected-r['total_L'])<3e-5,'Normalized base sixL1.25 CE/used-KL objective differs')
 return abs(expected-r['total_L'])

def segment_logs(folder,mode=None):
 folder=Path(folder);done=read(folder/'completion.json');result=read(folder/'result.json');require(done['epoch']==20 and done['optimizer_steps']==4440 and result['L_NN']==3409920 and result['L_As']==284160 and result['Ag_packets']==0 and result['extra_U_NN']==0 and result['U_As']==0,'Complete full practical E31-50 native budget required')
 streams=[lines(folder/f) for f in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz')]
 inventory=result['satellite_subnetwork']['dropout_inventory'] if mode else []
 if mode:streams.append(lines(folder/'satellite_subnetwork_steps.jsonl.gz'))
 sums=defaultdict(Counter);count=Counter();ordinary=[];sat_records=[];maxerr=0.;maxkl=0.
 for i,items in enumerate(zip_longest(*streams),1):
  require(all(x is not None for x in items),'Full source step streams unequal');r,s,n,u=items[:4];local=(i-1)//222+1;epoch=local+30;cumulative=i+6660;m=n['metrics']
  require(all(x.get('step')==i for x in items) and all(x.get('cumulative_step')==cumulative for x in (r,s,u)) and n.get('epoch')==local and u.get('physical_clip_calls')==i,'Actual full step/global E31-50 clocks differ')
  require(r.get('L_forwards')==6 and r.get('As_packets')==64 and r.get('Ag_packets')==0 and r.get('Ag_mode')=='NONE' and r.get('extra_control_packets')==128 and r.get('ordinary_redraws')==0 and r.get('same_IQ_each_pair') is True and r.get('all_satellite_L_supervised') is True,'SixL/same generated IQ/all true As CE scope differs')
  require(s.get('mode')=='NONE' and s.get('NN')==256 and s.get('packets')==128 and s.get('applied')==0 and s.get('weight')==.25 and s.get('same_IQ_each_pair') is True and not s.get('new_boundary_samples'),'Unchanged raw third-pair slot differs')
  for k in ('common_CE','common_KL','extra_CE','extra_KL','SupCon','SupCon_weight','total_L','base_L_before_shape','ordinary_raw_MSE'):finite(r[k],k)
  require(abs(r['SupCon_weight']-.1)<1e-12,'Original full0.1 SupCon schedule differs');maxerr=max(maxerr,loss_decomposition(r,s));finite(m['train/loss'],'native loss')
  require(not any(m.get(k) or 0 for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad','train/pseudo_truth_available')),'Invalid native update or U truth exposure')
  total=int(m['train/pseudo_total']);selected=int(m['train/pseudo_selected']);require(0<=selected<=total<=256 and total>0,'Existing U counts differ');count['U']+=total;count['selected_U']+=selected;count['steps']+=1
  if mode:
   q=items[4];maxkl=max(maxkl,satellite_record(q,mode,i,cumulative,inventory));require(r.get('satellite_subnetwork')=={k:v for k,v in q.items() if k!='cumulative_step'} and abs(r['extra_KL']-q['used_mixed_KL'])<3e-5,'Actual used extra KL/nested satellite telemetry differs');sat_records.append(q)
   for k in ('original_mixed_KL','ordinary_half_KL','satellite_half_KL','used_mixed_KL','removed_satellite_KL','satellite_pair_logit_maxdiff','dropout_calls','bypassed_elements'):sums[local][k]+=q[k]
  for k in ('common_CE','common_KL','extra_CE','extra_KL','SupCon','total_L'):sums[local][k]+=r[k]
  for k in ('shape_CE','shape_KL','shape_RD'):sums[local][k]+=s[k]
  sums[local]['native_loss']+=m['train/loss'];sums[local]['steps']+=1;ordinary.append(dict(step=cumulative,epoch=epoch,MSE=r['ordinary_raw_MSE']))
 require(count['steps']==4440 and count['U']==1134000 and set(sums)==set(range(1,21)) and all(s['steps']==222 for s in sums.values()),'All20 epochs/4440 actual updates required')
 compact=list(lines(folder/'epoch_metrics.jsonl'));full=list(lines(folder/'metrics_epoch.jsonl'));require([q['epoch'] for q in compact]==[q['epoch'] for q in full]==list(range(1,21)),'Both complete native epoch telemetry streams required')
 curve=[]
 for q in compact:
  e=q['epoch'];c=dict(epoch=e+30,local_epoch=e,**{k:v/222 for k,v in sums[e].items() if k!='steps'})
  for k,target in [('lr','lr'),('val_tx_acc','source_val_accuracy_pct'),('train_grad_backbone','grad_backbone'),('epoch_time_s','epoch_wall_seconds')]:c[target]=finite(q[k],k)
  curve.append(c)
 require(abs(curve[0]['lr']-.0002)<1e-15 and abs(curve[-1]['lr']-.000001)<1e-15 and np.all(np.diff([q['lr'] for q in curve])<=1e-15),'Actual reset cosine20 differs')
 if mode:require(result['satellite_subnetwork']['mode']==mode and result['satellite_subnetwork']['steps']==4440 and result['satellite_subnetwork']['extra_NN']==0 and result['satellite_subnetwork']['bypassed_elements']==sum(q['bypassed_elements'] for q in sat_records),'Actual satellite result/steps differs')
 return dict(status='FULL_E31_E50_LOGS_VERIFIED',epochs=20,updates=4440,L_NN=3409920,As_generated=284160,native_U=count['U'],selected_U=count['selected_U'],maximum_L_error=maxerr,maximum_KL_error=maxkl,curve=curve,ordinary=ordinary,satellite_steps=sat_records,cost=native_cost(folder))

def native_cost(folder):
 r=read(Path(folder)/'result.json');c={k:r[k] for k in COUNTERS};require(all(type(v) is int and v>=0 for v in c.values()),'Actual native four counters required');return dict(**c,native_NN=sum(c.values()),wall_seconds=float(r['wall_seconds']))

def epoch_curve(folder,offset):
 folder=Path(folder);sums=Counter();counts=Counter()
 for step in lines(folder/'step_metrics.jsonl.gz'):sums[step['epoch']]+=finite(step['metrics']['train/loss'],'native loss');counts[step['epoch']]+=1
 records=list(lines(folder/'epoch_metrics.jsonl'));require([x['epoch'] for x in records]==list(range(1,len(records)+1)) and all(counts[x['epoch']]==222 for x in records),'Complete historical epoch/step curve required')
 return [dict(epoch=x['epoch']+offset,source_val_accuracy_pct=x['val_tx_acc'],native_loss=sums[x['epoch']]/222,epoch_wall_seconds=x['epoch_time_s'],lr=x['lr'],grad_backbone=x['train_grad_backbone']) for x in records]

def actual_training_coverage(result,row_id,expected_packets=None):
 actual=result.get('TX_RX_scene_state_lock')
 if actual is None:return dict(available=False,cells=None,packets=None),[]
 packets=sum(x['count'] for x in actual)
 if expected_packets is not None:require(packets==expected_packets,'Actual TX/RX/scene/state coverage differs')
 return dict(available=True,cells=len(actual),packets=packets),[dict(row_id=row_id,**x) for x in actual]

def training_analysis(project,specs,out):
 reference=next(s for s in specs if s['key']=='RD50');rd=segment_logs(reference['root']/'source');details={};curves=[];satellite=[];coverage=[];coverage_status={};channel_strata=[]
 for spec in specs:
  if spec['reused']:
   saved=spec['root']/'source/result.json';status,cells=actual_training_coverage(read(saved) if saved.exists() else {},spec['key']);coverage_status[spec['key']]=status;coverage.extend(cells)
   for folder,epochs,labels,offset,parent in lineage_segments(spec):curves.extend(dict(row_id=spec['key'],reused=True,**q) for q in epoch_curve(folder,offset))
   continue
  mode=spec['arm'];root=spec['root'];folder=root/'source';value=segment_logs(folder,mode);n=bad=0
  for a,b in zip_longest(lines(folder/'training_sequence.jsonl.gz'),lines(reference['root']/'source/training_sequence.jsonl.gz')):n+=1;bad+=a!=b
  require(n>0 and bad==0,'Same E50RD source physical L/U sequence differs');mse_bad=sum(a!=b for a,b in zip_longest(value.pop('ordinary'),rd['ordinary']));require(mse_bad==0,'Original frozen ordinary generated IQ MSE stream differs')
  r=read(folder/'result.json');ch=r['L_channel'];require(ch.get('session_count')==15 and set(ch.get('sessions',{}))=={f'rx{rx}:day{day}:eq1' for rx in RX for day in DAYS},'All15 actual RX/day hardware sessions required')
  channel_strata.extend(dict(row_id=spec['key'],**q) for q in ch.get('strata',[]))
  status,cells=actual_training_coverage(r,spec['key'],284160);coverage_status[spec['key']]=status;coverage.extend(cells)
  value.update(physical_sequence_records=n,physical_sequence_mismatch=bad,ordinary_MSE_mismatch=mse_bad,satellite_CE_only_metric_available=False,mixed_CE_scope='extra_CE is the recorded mixed128 CE;64 true As plus64 ordinary;not an isolated As-only CE',L_channel=ch,TX_RX_scene_state_lock_available=status['available'],TX_RX_scene_state_lock_summary=status)
  satellite.extend(dict(row_id=spec['key'],**q) for q in value.pop('satellite_steps'));curves.extend(dict(row_id=spec['key'],reused=False,**q) for q in value['curve']);details[spec['key']]=value
 csv_write(out/'training_curves.csv',curves);csv_write(out/'satellite_steps.csv',satellite);csv_write(out/'coverage.csv',coverage);write(out/'training_coverage_summary.json',coverage_status);csv_write(out/'training_channel_strata.csv',channel_strata);write(out/'new_training.json',details);return details,curves

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
   value=lambda k:interaction_value({arm:match[arm][k] for arm in ('DET50','CE50','DET30','CE30')})
   interactions.append(dict(view=v,dimension=ref['dimension'],stratum=ref['stratum'],count=ref['count'],budget_x_DET_net_correct=value('correct'),budget_x_DET_accuracy_pp=100*value('accuracy'),budget_x_DET_macro_f1_pp=100*value('macro_f1'),budget_x_DET_mean_margin_over30=value('mean_margin_over30')))
 pair_summary=[]
 for new,control in PAIRS:
  for v in VIEWS:
   a=summary[new][v];b=summary[control][v];p=next(r for r in effects if r['new']==new and r['control']==control and r['view']==v and r['dimension']=='overall');pair_summary.append(dict(new=new,control=control,view=v,accuracy_delta_pp=p['accuracy_delta_pp'],macro_f1_delta_pp=100*(a['macro_f1']-b['macro_f1']),worst_RX_delta_pp=100*(a['worst_RX']-b['worst_RX']),mean_margin_over30_delta=a['mean_margin_over30']-b['mean_margin_over30'],fixed_NoAs30_retention_delta_pp=100*(a['satellite_retention']-b['satellite_retention']),repairs=p['repairs'],new_errors=p['new_errors'],count=p['count']))
 csv_write(out/'formal_pair_summary.csv',pair_summary)
 csv_write(out/'formal_interaction.csv',interactions)
 csv_write(out/'formal_metrics.csv',({k:v for k,v in r.items() if k!='confusion'} for r in records));write(out/'formal_confusions.json',records);csv_write(out/'formal_paired_effects.csv',effects)
 clean=100*(summary['DET50']['clean']['accuracy']-summary['NOAS50']['clean']['accuracy']);urban=100*np.mean([summary['DET50'][v]['accuracy']-summary['NOAS50'][v]['accuracy'] for v in VIEWS[-3:]])
 result=dict(status='FORMAL_ANALYZED',rows=summary,independent_confusion_recounts=len(records),partition_reconciliations=partitions,all8_rows_all7_verified_before_truth=True,formal_state_labels_available=False,fixed_NOAS30_clean_correct_count=int(cohort.sum()),stage_clean_delta_pp=clean,stage_clean_max_drop_pp=.1,stage_clean_within_0p1pp=clean>=-.1-1e-10,urban_mean_vs_NoAs50_pp=float(urban),stage_clean_reference='NOAS50',urban_mean_vs_RD50_pp=float(100*np.mean([summary['DET50'][v]['accuracy']-summary['RD50'][v]['accuracy'] for v in VIEWS[-3:]])),single_seed_observed_dual_gain=clean>0 and urban>0,model_seed_count=1,reproducible_dual_gain=False,target_feedback=False,NN=0,IQ_reads=0,checkpoint_reads=0)
 write(out/'formal_summary.json',result);return result

def cost_analysis(project,specs,source):
 unique={};references={};new={}
 for spec in specs:
  lineage=[]
  for folder,epochs,labels,offset,parent in lineage_segments(spec):
   key=str(folder);unique.setdefault(key,dict(folder=key,epochs=epochs,updates=epochs*222,cost=native_cost(folder),new=folder==spec['root']/'source' and not spec['reused']));lineage.append(key)
  pred=read(spec['root']/'prediction/complete.json');qp=spec['root']/'source_qualification/complete.json';q=read(qp) if qp.is_file() else None;qn=q['NN'] if q and q.get('status')=='SOURCE_READOUT_COMPLETE' else None
  r=dict(lineage=lineage,source_readout_NN=qn,formal_prediction_NN=pred['actual_NN'],formal_prediction_wall_seconds=pred['wall_seconds'],cumulative_training_NN=sum(unique[k]['cost']['native_NN'] for k in lineage),cumulative_model_epochs=spec['budget'],cumulative_updates=spec['budget']*222)
  if spec['reused']:references[spec['key']]=r
  else:
   native=unique[str(spec['root']/'source')]['cost'];require(tuple(native[k] for k in COUNTERS)==(4543920,594002,0,1134000) and native['native_NN']==6271922,'Actual new student/teacher train/eval packet counters differ');r.update(new20=native,new_L_NN=3409920,new_As_generated=284160,new_remote_NN_known=native['native_NN']+(qn or 0)+pred['actual_NN'],new_remote_NN_complete=qn is not None);new[spec['key']]=r
 smoke=read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json');require(smoke.get('status')=='PASS' and smoke.get('NN')==4,'Actual two-parent no-query smoke differs')
 hist=sum(v['cost']['native_NN'] for v in unique.values() if not v['new'])+sum((v['source_readout_NN'] or 0)+v['formal_prediction_NN'] for v in references.values());fresh=sum(v['new_remote_NN_known'] for v in new.values())+smoke['NN']
 return dict(new_rows=new,historical_references=references,unique_training_segments=list(unique.values()),new_model_rows=2,new_model_epochs=40,new_updates=8880,new_L_NN=6819840,new_As_generated=568320,new_source_readout_NN=sum(v['source_readout_NN'] or 0 for v in new.values()),new_formal_prediction_NN=sum(v['formal_prediction_NN'] for v in new.values()),new_remote_smoke_NN=smoke['NN'],new_total_remote_NN_known=fresh,new_total_remote_NN_complete=all(v['new_remote_NN_complete'] for v in new.values()),historical_reference_NN_known=hist,historical_unique_model_epochs=sum(v['epochs'] for v in unique.values() if not v['new']),historical_unique_updates=sum(v['updates'] for v in unique.values() if not v['new']),unique_total_model_epochs=sum(v['epochs'] for v in unique.values()),unique_total_updates=sum(v['updates'] for v in unique.values()),unique_total_remote_NN_known=fresh+hist,shared_parent_segments_counted_once=True,historical_shared_smokes_NN_unallocated=True,reference_reprediction=0,reference_rescoring=0,analysis_NN=0,budget_exhaustion_is_not_convergence=True)

def plots(out,training,source_rows,formal):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'NOAS30':'#cbd5e1','RD30':'#fdba74','CE30':'#a5b4fc','DET30':'#86efac','NOAS50':'#475569','RD50':'#d97706','CE50':'#6366f1','DET50':'#15803d'};files=[]
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):p=out/(name+'.'+ext);fig.savefig(p,dpi=180);files.append(str(p))
  plt.close(fig)
 if training:
  fig,axes=plt.subplots(2,3,figsize=(17,9))
  for arm in ('NOAS50','RD50','CE50','DET50'):
   c=[r for r in training if r['row_id']==arm and r['epoch']>=31];x=[r['epoch'] for r in c]
   for ax,key,title in zip(axes.flat,('native_loss','source_val_accuracy_pct','grad_backbone','lr','satellite_half_KL','satellite_pair_logit_maxdiff'),('Native training loss','Logged source V accuracy (%)','Backbone gradient','Cosine20 learning rate','Satellite RDrop KL (observed)','Satellite pair maximum logit difference')):
    points=[(r['epoch'],r[key]) for r in c if key in r];ax.plot([p[0] for p in points],[p[1] for p in points],label=arm,color=colors[arm],marker='.',markersize=3);ax.set(title=title,xlabel='Cumulative epoch');ax.grid(alpha=.2)
  axes[0,0].legend();fig.suptitle('All stored E31-50 summaries; missing historical satellite telemetry is omitted, never imputed');save(fig,'E31_E50_training_curves')
 if source_rows:
  for scene in SCENES[1:]:
   fig,axes=plt.subplots(1,2,figsize=(18,5.8))
   for ax,dimension in zip(axes,('state_lock','signal_to_added_noise_db')):
    rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dimension];labels=list(dict.fromkeys(r['stratum'] for r in rr));xx=np.arange(len(labels));ns=[]
    for i,arm in enumerate(('NOAS50','RD50','CE50','DET50')):
     seq=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels];vals=[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in seq];ax.bar(xx+(i-1.5)*.2,vals,.2,label=arm,color=colors[arm]);ns=ns or [r['count'] for r in seq]
    ax.set(xticks=xx,xticklabels=[l+'\nn='+str(n) for l,n in zip(labels,ns)],ylim=(0,104),ylabel='Source V accuracy (%)',title=dimension);ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2)
   axes[0].legend(ncol=2);fig.suptitle(scene+'; pooled two source draws / one model seed / full condition only');save(fig,scene+'_source_states')
 if formal:
  rows=formal['rows'];xx=np.arange(7);labels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'];fig,axes=plt.subplots(1,2,figsize=(19,5.5))
  for ax,budget in zip(axes,(30,50)):
   for i,method in enumerate(('NOAS','RD','CE','DET')):arm=method+str(budget);ax.bar(xx+(i-1.5)*.2,[100*rows[arm][v]['accuracy'] for v in VIEWS],.2,label=arm,color=colors[arm])
   ax.set(xticks=xx,xticklabels=labels,ylim=(0,103),ylabel='Formal accuracy (%)',title=f'Fixed E{budget} default-head models');ax.legend(ncol=2);ax.grid(axis='y',alpha=.2)
  fig.suptitle('Same fixed received IQ; all8 rows/all7 views gated before truth');save(fig,'formal_all7_accuracy')
  fig,axes=plt.subplots(1,2,figsize=(17,5.2))
  for method in ('NOAS','RD','CE','DET'):axes[0].plot(xx,[100*(rows[method+'50'][v]['accuracy']-rows[method+'30'][v]['accuracy']) for v in VIEWS],marker='o',label=method,color=colors[method+'50'])
  axes[1].bar(xx,[100*interaction_value({arm:rows[arm][v]['accuracy'] for arm in ('DET50','CE50','DET30','CE30')}) for v in VIEWS],color=colors['DET50'])
  for ax,title in zip(axes,('Each own E50 minus E30: total budget effect','(DET50−CE50)−(DET30−CE30)')):ax.set(xticks=xx,xticklabels=labels,ylabel='Absolute percentage points',title=title);ax.axhline(0,color='grey',linewidth=.7);ax.grid(axis='y',alpha=.2)
  axes[0].legend();fig.suptitle('Single model seed; budget differences are not all new-mechanism gains');save(fig,'paired_budget_effects')
 return files

def report(out,result):
 f=result['formal'];rows=f['rows'];cost=result['cost'];text=['# P62 satellite-subnetwork E50 budget validation','', 'Two new CE/DET continuations each start from their own fixed SAT18 FINAL30 and add20 epochs. NoAs/RD E30/E50 and CE/DET E30 are reused without new predictions or scoring. Only the actual As64 rows lose satellite KL; DET also bypasses existing dropout outputs while retaining native mask draws. This is training organization, not a new IQ augmentation.','', '| View | NoAs30 | RD30 | CE30 | DET30 | NoAs50 | RD50 | CE50 | DET50 | DET50−CE50 pp | Budget interaction pp |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for view in VIEWS:
  vals=[100*rows[arm][view]['accuracy'] for arm in ARMS];interaction=100*interaction_value({arm:rows[arm][view]['accuracy'] for arm in ('DET50','CE50','DET30','CE30')});text.append('| '+view+' | '+' | '.join(f'{v:.6f}' for v in vals)+f" | {vals[7]-vals[6]:+.6f} | {interaction:+.6f} |")
 text += ['', '| View | DET50 macro F1 % | Worst RX % | Default margin/30 | Fixed NoAs30 clean-correct retention % |','|---|---:|---:|---:|---:|']
 for view in VIEWS:
  r=rows['DET50'][view];text.append(f"| {view} | {100*r['macro_f1']:.6f} | {100*r['worst_RX']:.6f} | {r['mean_margin_over30']:.8f} | {100*r['satellite_retention']:.6f} |")
 text += ['',f"DET50 clean versus NoAs50: {f['stage_clean_delta_pp']:+.6f} absolute pp; at most0.1 pp clean-cost stage: {f['stage_clean_within_0p1pp']}.", 'The source_matrix_decision.json is read before formal truth and retained unchanged. Source scientific FAIL still requires both new all7 formal readouts; no E70, seed extension, parameter scan or target-driven promotion is authorized by this analysis.','', 'Only one model seed is present. The two channel draws repeat the same1800 source packets. Full V27000, source panel1800 and formal target are distinct scopes; formal channel state labels are unavailable. A source state association is not a unique physical cause.']
 if result.get('source'):text += ['', 'Stored source decision: '+json.dumps(result['source']['stored_source_decision'],ensure_ascii=False)]
 text += ['',f"Costs: new{cost['new_model_epochs']} model epochs/{cost['new_updates']} updates, actual native/source/prediction/smoke NN reported separately. Unique represented training: {cost['unique_total_model_epochs']} model epochs/{cost['unique_total_updates']} updates. Shared P43→budget50 and SAT18→new50 parent segments are counted once by actual directory; SAT18 CE and DET scratch20 are separate origins.",f"New remote NN known: {cost['new_total_remote_NN_known']}; source readout NN{cost['new_source_readout_NN']}; formal prediction NN{cost['new_formal_prediction_NN']}; real-parent smoke NN{cost['new_remote_smoke_NN']}. External local verification NN: {cost.get('local_verification_NN')}; unknown if omitted, never guessed. Historical shared launch smoke is unallocated, not silently assigned to every child.",'Actual student/teacher train/eval counts and model wall times are in costs.json. Budget exhaustion is not convergence.','', 'Missing/partial: '+json.dumps(result['missing'],ensure_ascii=False),'', 'Figures:']
 text += ['- '+str(p) for p in result['figures']];text += ['','All per-view accuracy/F1, RX/day/TX, paired repairs/new errors, margins/30, fixed NoAs30 cohort retention, state/lock/added-noise bins and empty/small cells are preserved in CSV/JSON. Added-noise SNR is not total SNR. This CPU recount reads no IQ or checkpoint, performs no inference/optimizer/head update and launches no scorer/prediction.']
 (out/'final_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def run(project,out,truth=None,code_commit=None,local_NN=None):
 project=Path(project);truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json';out=Path(out);require(not out.exists(),'Immutable new analysis output required; reconcile existing directory');out.mkdir(parents=False);start=time.perf_counter();specs=row_specs(project);missing=[];source=None;sr=[];tr=None;curves=[]
 require((project/'runs'/RUN/'source_matrix_decision.json').is_file(),'Existing source decision required before formal readout')
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
 formal=formal_analysis(project,specs,out,truth);cost=cost_analysis(project,specs,source);cost['local_verification_NN']=local_NN;write(out/'costs.json',cost)
 try:figures=plots(out,curves,sr,formal)
 except Exception as error:figures=[];failure=dict(component='plot',error=repr(error),traceback=traceback.format_exc());missing.append(failure);write(out/'plot_failure.json',failure)
 result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,code_commit=code_commit,source=source,formal=formal,cost=cost,missing=missing,figures=figures,training=tr,model_seed_count=1,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,source_decision_recomputed=False,target_feedback=False,crossseed_confirmed=False,goal_complete=False,wall_seconds=time.perf_counter()-start)
 report(out,result);write(out/'complete.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--truth',type=Path);p.add_argument('--code-commit');p.add_argument('--local-nn',type=int);a=p.parse_args();r=run(a.project,a.out,a.truth,a.code_commit,a.local_nn);print(json.dumps(dict(status=r['status'],out=str(a.out),missing_components=[x['component'] for x in r['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)
if __name__=='__main__':main()
