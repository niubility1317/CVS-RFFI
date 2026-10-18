"""P66 same-mass U-quality allocation CPU recount; no model/IQ/CP access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse,csv,gzip,json,math,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
RUN='20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01'
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
COUNTERS=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')
NEW_ARMS=('MASS_UNIFORM','OBSERVATION_WEIGHTED')
CANDIDATE='OBSERVATION_WEIGHTED'
PAIRS=tuple((new,control) for new in NEW_ARMS for control in ('AS30','NOAS30'))+((CANDIDATE,'MASS_UNIFORM'),)
QUALITY_SUMS=('KL','weighted_KL','unweighted_KL','positive_q','native_packets','original_selected','extra_original_selected','q_mass','none_snr_count')
EXTRA_STUDENT=284120
EXTRA_FROZEN=284120

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
 p=Path(project)/'runs'
 refs=[dict(key=key,run=REF,row_id=arm+'-s'+str(SEED),arm=arm,root=p/REF/(arm+'-s'+str(SEED)),reused=True) for key,arm in [('NOAS30','B00'),('AS30','B01')]]
 return refs+[dict(key=arm,run=RUN,row_id=arm+'-s'+str(SEED),arm=arm,root=p/RUN/(arm+'-s'+str(SEED)),reused=False) for arm in NEW_ARMS]

def training_complete(spec,project):
 root=spec['root'];runs=Path(project)/'runs';roles=read(runs/CONTRACT_RUN/'source_contract.json')['role_ids']
 parent=runs/REF/'B01-s2026092701'/'stage20' if not spec['reused'] else root/'stage20'
 parent_row='B01-s2026092701' if not spec['reused'] else spec['row_id']
 stages=[(parent,20,13,7,4440,True,parent_row,spec['key']!='NOAS30')]
 stages.append((root/'source',10,0,10,2220,False,spec['row_id'],spec['key']!='NOAS30'))
 for folder,e,l,u,steps,is_parent,expected_row,expected_As in stages:
  done=read(folder/'completion.json');c=done['config'];frozen=read(folder/'source_frozen.json');init=read(folder/'initialization.json');result=read(folder/'result.json')
  expected_run=REF if is_parent else spec['run']
  require((done.get('epoch'),done.get('optimizer_steps'),c.get('epochs'),c.get('label_epochs'),c.get('pseudo_epochs'),c.get('updates'))==(e,steps,e,l,u,steps),'Exact own20+10 training calendar required')
  require(c.get('run_id')==expected_run and c.get('row_id')==expected_row and c.get('model_seed')==SEED and c.get('As') is expected_As,'Source row/run/seed/As role differs')
  require(not done.get('target_access') and done.get('head_fit_updates')==0 and done.get('native_fitted_probes_disabled') is True,'Source default head/no target flags differ')
  require(frozen.get('status')=='SOURCE_FROZEN' and frozen.get('config')==c and frozen.get('selection')=='final_only' and not frozen.get('target_access'),'Final-only source freeze required')
  expected=[] if is_parent else [str(parent/'final_ssdg.pth')]
  require(c.get('checkpoint_sources')==expected and init.get('checkpoint_sources')==expected and init.get('scratch_only')==is_parent and not init.get('target_contact') and init.get('source_roles')=='EXACT_MATCH','Own legal single As20 student/teacher provenance differs')
  require(read(folder/'source_contract.json')['role_ids']==roles,'Actual L/U/V roles differ')
  require(result.get('config')==c and result.get('head_fit_updates')==0 and result.get('U_truth_read') is False and result.get('target_access') is False,'Joint defaulthead/hidden U/source result differs')
  require(c.get('Ag_mode')=='NONE' and c.get('new_Ag') is False and c.get('L_NN_per_step')==768 and c.get('total_L_loss_scale')==1.25,'Fixed no-Ag sixL loss contract required')
  require(result.get('source_RX')==list(RX) and result.get('source_RX_holdout') is None,'All source RX required')
  require(done['final_source_metrics'].get('source_val_count')==27000,'Full source V27000 required')
  if not is_parent:
   require(c.get('parent_source')==str(parent) and c.get('epoch_offset')==20 and c.get('step_offset')==4440,'Exact own20 parent and cumulative clock required')
   if not spec['reused']:
    require(c.get('arm')==spec['key'] and c.get('teacher_sources')=={'B01':str(parent/'final_ssdg.pth')},'Only the same original As20 EMA teacher is allowed')
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
 saved=read(root/'scoring/scores.json')
 require(saved.get('truth_last') is True and saved.get('target_feedback_forbidden') is True,'Truth closed: independent score flags differ')
 for view in VIEWS:
  rows=[r for r in saved.get('results',[]) if r.get('view')==view and r.get('dimension')=='overall' and r.get('stratum')=='ALL']
  require(len(rows)==1 and rows[0].get('query_count')==count,'Truth closed: all seven independent score views required')
 ids=np.asarray(index_ids);require(ids.shape==(count,) and ids.dtype.kind in 'US' and len(set(ids.tolist()))==count,'Truth closed: unique formal IDs required')
 with np.load(folder/'predictions.npz',allow_pickle=False) as f:
  require(set(f.files)=={'ids',*VIEWS} and np.array_equal(f['ids'],ids),'Truth closed: prediction IDs/views differ');pred={v:f[v].copy() for v in VIEWS}
 logits={}
 for view in VIEWS:
  a=pred[view];require(a.shape==(count,) and a.dtype.kind in 'iu' and np.all((a>=0)&(a<6)),'Truth closed: bad decisions')
  with np.load(folder/(view+'_scores.npz'),allow_pickle=False) as f:
   z=f['logits'];require(np.array_equal(f['ids'],ids) and z.shape==(count,6) and np.isfinite(z).all() and np.array_equal(z.argmax(1),a),'Truth closed: bad logits/IDs/argmax');logits[view]=z.copy()
 return dict(ids=ids.copy(),pred=pred,logits=logits,prediction=p,scoring=s,saved_scores=saved)

def truth_last_arrays(specs,index_ids,truth_path,project,count=COUNT,checker=None):
 """All logical rows and all7 scores are checked before the sole truth read."""
 checker=checker or training_complete;data={}
 for spec in specs:data[spec['key']]=frozen_predictions(spec,checker(spec,project),index_ids,count)
 truth=read(truth_path) # Deliberately after all four logical rows and all seven views.
 ids=np.asarray(index_ids);require(set(truth)==set(ids.tolist()),'Truth physical IDs differ')
 y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids]);day=np.asarray([str(truth[s]['day']) for s in ids])
 require(y.dtype.kind in 'iu' and y.shape==(count,) and np.all((y>=0)&(y<6)),'Truth class map differs')
 if count==COUNT:
  require(set(rx)==set(TARGET_RX) and set(day)==set(TARGET_DAYS),'Fixed heldout RX/day differ')
  for label in range(6):
   for receiver in TARGET_RX:
    for date in TARGET_DAYS:require(int(((y==label)&(rx==receiver)&(day==date)).sum())==1000,'Fixed formal cell count differs')
 return data,dict(ids=ids,y=y,rx=rx,day=day)

def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);require(y.ndim==1 and p.shape==y.shape and y.dtype.kind in 'iu' and p.dtype.kind in 'iu' and np.all((y>=0)&(y<6)&(p>=0)&(p<6)),'Invalid class decisions')
 n=len(y);cm=np.bincount((y*6+p).astype(int),minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(count=n,correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/n) if n else None,macro_f1=float(f.mean()) if n else None,confusion=cm.tolist(),small_cell=n<30,empty_cell=n==0)
def paired(y,old,new,cohort=None):
 y=np.asarray(y);a=np.asarray(old)==y;b=np.asarray(new)==y;require(a.shape==b.shape==y.shape,'Paired lengths differ');c=np.ones(len(y),bool) if cohort is None else np.asarray(cohort,bool);require(c.shape==y.shape,'Cohort length differs');n=len(y);nc=int(c.sum());repair=int((~a&b).sum());damage=int((a&~b).sum())
 return dict(count=n,control_correct=int(a.sum()),new_correct=int(b.sum()),repairs=repair,new_errors=damage,net_repairs=repair-damage,accuracy_delta_pp=100*(repair-damage)/n if n else None,fixed_NOAS30_clean_correct_count=nc,cohort_repairs=int((~a&b&c).sum()),cohort_new_errors=int((a&~b&c).sum()),control_retention=float(a[c].mean()) if nc else None,new_retention=float(b[c].mean()) if nc else None)
def logit_metric(y,z,cohort):
 z=np.asarray(z);cohort=np.asarray(cohort,bool)
 require(z.shape==(len(y),6) and np.isfinite(z).all() and cohort.shape==(len(y),),'Finite default-head logits and matching cohort required')
 m=metric(y,z.argmax(1));n=len(y);other=z.copy()
 if n:
  other[np.arange(n),y]=-np.inf;margin=(z[np.arange(n),y]-other.max(1))/SCALE;mean=float(margin.mean());median=float(np.median(margin))
 else:mean=median=None
 nc=int(cohort.sum());retained=int(((z.argmax(1)==y)&cohort).sum());m.update(mean_margin_over30=mean,median_margin_over30=median,fixed_NOAS30_clean_correct_count=nc,fixed_NOAS30_clean_correct_retained=retained,satellite_retention=retained/nc if nc else None);return m

def reconcile_partitions(overall,parts):
 cm=np.asarray(overall['confusion']);require(parts,'Partition cannot be empty');require(np.array_equal(sum((np.asarray(x['confusion']) for x in parts),np.zeros((6,6),int)),cm),'Partition confusion/count does not reconcile');return True

DECISION_COMPLETE={'SOURCE_READOUT_DECIDED'}
DECISION_PARTIAL={'SOURCE_READOUT_PARTIAL'}

def read_source_decision(project):
 """Read the producer's already fixed verdict; never recalculate promotion."""
 decision=read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
 require(decision.get('status') in DECISION_COMPLETE|DECISION_PARTIAL,'Stored source producer status is unknown')
 require(decision.get('formal_scores_read') is False and decision.get('target_feedback') is False and decision.get('model_seed_count')==1 and decision.get('NN')==0,'Stored source decision must precede formal access')
 if decision['status'] in DECISION_COMPLETE:
  require(type(decision.get('source_qualified')) is bool,'Completed source decision needs an explicit verdict')
  checks=decision.get('checks');require(isinstance(checks,dict) and len(checks)==5 and all(type(v) is bool for v in checks.values()),'Completed source checks must be five Python bools')
  require(decision['source_qualified']==all(checks.values()),'Stored source verdict/checks disagree')
 return decision

def source_analysis(project,specs,out,decision):
 panel=Path(project)/'runs'/PANEL_RUN/'diagnostic';role=read(panel/'data_roles.json');ids=np.asarray(role['physical_ids']);n=1800
 require(role.get('role')=='V' and role.get('counts')=={'L_s':6300,'U_s':56700,'V':27000} and role.get('held_RX') is None and role.get('target_access') is False and role.get('U_truth_used') is False and role.get('updates')==0 and role.get('V_panel')==n and ids.shape==(n,) and len(set(ids.tolist()))==n,'Exact legal source V panel required')
 pc=read(panel/'observations/completion.json');require(pc.get('status')=='SOURCE_OBSERVATIONS_COMPLETE' and pc.get('unique_V')==n and pc.get('draws')==2 and tuple(pc.get('scenes',()))==SCENES and pc.get('conditions',[None])[0]=='full' and pc.get('target_access') is False and pc.get('training_updates')==0,'Original full source condition/draw order required')
 with np.load(panel/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in ('state','lock','quality_snr_db','signal_to_added_noise_db')}
 for k,x in physics.items():require(x.shape==(2,3,len(pc['conditions']),n),'Source physics shape differs '+k)
 data={};health={};qproof={}
 for spec in specs:
  done=training_complete(spec,project);q=read(spec['root']/'source_qualification/complete.json');require(q.get('status')=='SOURCE_READOUT_COMPLETE' and q.get('config')==done['config'] and q.get('NN')==12600 and all(q.get(k)==0 for k in ('training_updates','head_fit_updates')) and q.get('target_access') is False and q.get('formal_scores_read') is False,'Frozen source readout incomplete')
  with np.load(spec['root']/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in ('ids','TX','RX','day','clean','satellite')}
  require(np.array_equal(x['ids'],ids) and x['clean'].shape==(n,6) and x['satellite'].shape==(2,3,n,6) and np.isfinite(x['clean']).all() and np.isfinite(x['satellite']).all(),'Source panel IDs/logits differ')
  for k,allowed in [('TX',range(6)),('RX',RX),('day',DAYS)]:require(x[k].shape==(n,) and x[k].dtype.kind in 'iu' and set(x[k])==set(allowed),'Source role metadata differs '+k)
  require(abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])<1e-12 and np.allclose((x['satellite'].argmax(-1)==x['TX']).mean(-1),q['source_accuracies'],atol=1e-12,rtol=0),'Source readout does not recount')
  data[spec['key']]=x;health[spec['key']]=done['final_source_metrics'];qproof[spec['key']]=q
 ref=data['NOAS30'];cohort=ref['clean'].argmax(1)==ref['TX'];records=[];effects=[]
 for x in data.values():require(all(np.array_equal(x[k],ref[k]) for k in ('ids','TX','RX','day')),'Matched source physical roles differ')
 for scene in ('clean',*SCENES):
  si=SCENES.index(scene) if scene!='clean' else None
  for draw in ((-1,) if scene=='clean' else (-1,0,1)):
   ds=(0,1) if draw==-1 else (draw,);copies=1 if scene=='clean' else len(ds);y=np.tile(ref['TX'],copies);cc=np.tile(cohort,copies);rx=np.tile(ref['RX'],copies);day=np.tile(ref['day'],copies)
   zz={a:x['clean'] if scene=='clean' else np.concatenate([x['satellite'][d,si] for d in ds]) for a,x in data.items()};masks=[('overall','ALL',np.ones(len(y),bool))]
   for dim,values,allowed in [('RX',rx,RX),('day',day,DAYS),('TX',y,range(6))]:masks.extend((dim,str(v),values==v) for v in allowed)
   if scene!='clean':
    state=np.concatenate([physics['state'][d,si,0] for d in ds]);lock=np.concatenate([physics['lock'][d,si,0] for d in ds]);quality=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in ds]);proxy=np.concatenate([physics['quality_snr_db'][d,si,0] for d in ds])
    require(np.isfinite(state).all() and np.isfinite(lock).all() and np.all((state>=0)&(state<3)&(state==state.astype(int))) and np.all((lock>=0)&(lock<3)&(lock==lock.astype(int))),'State/lock indices differ')
    masks.extend(('state',s,state==i) for i,s in enumerate(STATES));masks.extend(('lock',s,lock==i) for i,s in enumerate(LOCKS));masks.extend(('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(STATES) for j,l in enumerate(LOCKS))
    masks.extend(('signal_to_added_noise_db',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]);masks.append(('signal_to_added_noise_db','MISSING',np.isnan(quality)))
    require(np.all(np.isfinite(quality)|np.isnan(quality)) and np.isfinite(proxy).all(),'Source added-noise ratio/receiver-quality proxy differs')
    masks.extend(('receiver_quality_proxy_db',label,(proxy>=lo)&(proxy<hi)) for label,lo,hi in [('<5',-np.inf,5),('5-15',5,15),('>=15',15,np.inf)])
   for dim,label,mask in masks:
    context=dict(scene=scene,draw=draw,dimension=dim,stratum=label);current={}
    for arm,z in zz.items():current[arm]=logit_metric(y[mask],z[mask],cc[mask]);records.append(dict(row_id=arm,model_seed=SEED,**context,**current[arm]))
    for new,control in PAIRS:
     p=paired(y[mask],zz[control][mask].argmax(1),zz[new][mask].argmax(1),cc[mask]);p.update(new=new,control=control,**context,mean_margin_over30_delta=current[new]['mean_margin_over30']-current[control]['mean_margin_over30'] if mask.any() else None);effects.append(p)
 csv_write(out/'source_strata.csv',({k:v for k,v in r.items() if k!='confusion'} for r in records));write(out/'source_confusions.json',records);csv_write(out/'source_paired_effects.csv',effects)
 overview={}
 for arm in data:
  overview[arm]={}
  for scene in ('clean',*SCENES):
   rr=[r for r in records if r['row_id']==arm and r['scene']==scene and r['draw']==-1];overall=next(r for r in rr if r['dimension']=='overall')
   for dim in ('RX','day','TX'):reconcile_partitions(overall,[r for r in rr if r['dimension']==dim])
   overview[arm][scene]={k:overall[k] for k in ('accuracy','macro_f1','mean_margin_over30','satellite_retention')}
   overview[arm][scene]['worst_RX']=min(r['accuracy'] for r in rr if r['dimension']=='RX')
 summary=dict(status='SOURCE_ANALYZED' if decision['status'] in DECISION_COMPLETE else 'SOURCE_ANALYZED_WITH_STORED_DECISION_PARTIAL',rows=overview,health_V27000=health,readout={k:{a:v for a,v in q.items() if a!='config'} for k,q in qproof.items()},stored_source_decision=decision,source_decision_recomputed=False,source_unique_physical=1800,channel_draws=2,model_seed_count=1,fixed_NOAS30_clean_correct_count=int(cohort.sum()),source_high_urban_available=False,fullV_F1_available=False,total_SNR_known=False,physical_causality_identified=False,formal_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0)
 write(out/'source_summary.json',summary);return summary,records

def parse_tail(folder,teacher_arm=None):
 """Read all 2220 actual updates; compare physical loss decomposition, not proxies."""
 folder=Path(folder);result=read(folder/'result.json');done=read(folder/'completion.json');require(result.get('L_NN')==1704960 and result.get('L_As')==142080 and result.get('optimizer_steps')==2220 and result.get('Ag_packets')==0 and result.get('Ag_control_NN')==568320 and result.get('extra_U_NN')==(EXTRA_STUDENT if teacher_arm else 0) and result.get('U_As')==(EXTRA_STUDENT if teacher_arm else 0),'Actual sixL/As/originalU counters differ')
 sums=defaultdict(Counter);counts=Counter();ordinary=[];max_error=0.;selected=total_u=0;teacher_rows=[];gradient_fields=Counter()
 streams=[lines(folder/f) for f in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz')]
 if teacher_arm:streams.append(lines(folder/'u_quality_steps.jsonl.gz'))
 for i,items in enumerate(zip_longest(*streams),1):
  require(all(x is not None for x in items),'Incomplete reference/shape/native/Sinc/U-quality logs');r,s,n,u=items[:4];local=(i-1)//222+1;ep=local+20;m=n['metrics']
  require(all(x.get('step')==i for x in items) and all(x.get('cumulative_step')==4440+i for x in (r,s,u)) and u.get('physical_clip_calls')==i and n.get('epoch')==local,'Contiguous E21-30 native update clocks differ')
  require(r.get('L_forwards')==6 and r.get('As_packets')==64 and r.get('Ag_packets')==0 and r.get('Ag_mode')=='NONE' and r.get('extra_control_packets')==128 and r.get('ordinary_redraws')==0 and r.get('same_IQ_each_pair') is True and r.get('all_satellite_L_supervised') is True,'Actual sixL/sameIQ/ungated labeled As differs')
  require(s.get('mode')=='NONE' and s.get('NN')==256 and s.get('packets')==128 and s.get('applied')==0 and s.get('weight')==.25 and s.get('same_IQ_each_pair') is True and not s.get('new_boundary_samples') and not s.get('relative_noise_RMS'),'Unchanged raw control slot differs')
  for k in ('common_CE','common_KL','extra_CE','extra_KL','SupCon','SupCon_weight','total_L','base_L_before_shape','ordinary_raw_MSE'):finite(r[k],k)
  for k in ('shape_CE','shape_KL','shape_RD'):finite(s[k],k)
  base_loss=.5*(r['common_CE']+r['common_KL'])+.5*(r['extra_CE']+r['extra_KL'])+.1*r['SupCon']+.25*s['shape_RD'];base_logged=r['total_L'];error=abs(base_loss-base_logged);max_error=max(max_error,error)
  require(r['SupCon_weight']==.1 and abs(s['shape_RD']-s['shape_CE']-s['shape_KL'])<3e-5 and error<3e-5 and abs(base_logged-r['base_L_before_shape']-.25*s['shape_RD'])<1e-6,'Original normalized total1.25/CE-KL-SupCon loss differs')
  finite(m['train/loss'],'native training loss');require(not any(m.get(k) or 0 for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad','train/pseudo_truth_available')),'Invalid or U-truth-exposed update')
  for key,value in m.items():
   if isinstance(value,(int,float)) and not isinstance(value,bool) and ('grad' in key.lower()) and not key.endswith(('skipped_nonfinite_grad',)):
    finite(value,key);gradient_fields[key]+=1
  tu=int(m.get('train/pseudo_total') or 0);su=int(m.get('train/pseudo_selected') or 0);require(0<=su<=tu<=256 and tu>0,'Old U sample counts differ');total_u+=tu;selected+=su
  if teacher_arm:
   teaching=items[4];validate_quality_record(teaching,i,tu,su,teacher_arm);teacher_rows.append(teaching)
   for key in ('KL','weighted_KL','unweighted_KL','q_mean','q_mass','positive_q','extra_student_NN','extra_frozen_teacher_NN'):sums[local][key]+=teaching[key]
   sums[local]['positive_q_fraction']+=teaching['positive_q']/teaching['packets']
   for key in ('q_teacher','q_observation','effective_q'):sums[local][key+'_mean']+=teaching[key]['mean']
   sums[local]['uniform_multiplier']+=teaching['mass_match']['uniform_multiplier']
   sums[local]['mass_absolute_difference']+=teaching['mass_match']['absolute_difference']
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
 summary=check_quality_summary(result.get('u_quality_summary'),teacher_rows,teacher_arm) if teacher_arm else None
 if teacher_arm:
  check_quality_summary(read(folder/'u_quality_progress.json'),teacher_rows,teacher_arm)
  require(result.get('native_extra_U_NN')==0 and result.get('native_U_As')==0 and result.get('extra_student_U_NN')==EXTRA_STUDENT and result.get('extra_frozen_teacher_U_NN')==EXTRA_FROZEN,'Separate native and extra teacher/student counters differ')
 return dict(status='FULL_TAIL_LOGS_VERIFIED',updates=2220,epochs=10,actual_L_NN=1704960,actual_As=142080,old_U_total=total_u,old_U_selected=selected,maximum_L_decomposition_error=max_error,gradient_telemetry_field_counts=dict(gradient_fields),teacher_parameter_gradients_saved=False,gradient_scope='Native global/Sinc/epoch gradient health only; no invented isolated teaching-gradient trajectory',u_quality_summary=summary,teacher_rows=teacher_rows,teacher_offset_not_added=True,curve=curves,ordinary=ordinary,cost=native_cost(folder))

def quality_statistics(stat,n,quantiles=True):
 require(isinstance(stat,dict) and stat.get('count')==n and type(stat.get('positive')) is int,'Complete quality statistic required')
 mass=finite(stat.get('mass'),'quality mass');mean=finite(stat.get('mean'),'quality mean')
 require(0<=mass<=stat['positive']+1e-5<=n+1e-5 and abs(mean-mass/n)<1e-6,'Quality mass/mean/positive count differs')
 h=stat.get('histogram');require(isinstance(h,list) and len(h)==5 and all(type(x) is int and x>=0 for x in h) and sum(h)==n and sum(h[1:])==stat['positive'],'Quality histogram differs')
 if quantiles:
  q=stat.get('quantiles');require(isinstance(q,list) and len(q)==5 and all(isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=1 for x in q) and all(a<=b for a,b in zip(q,q[1:])),'Five ordered per-step q quantiles required')
 return stat

def validate_quality_record(p,step,native_packets,original_selected,arm):
 local=(step-1)//222+1;n=min(128,native_packets);weight=.16*min(local/5.,1.)
 require(arm in NEW_ARMS and p.get('arm')==arm and p.get('step')==step and p.get('cumulative_step')==4440+step and p.get('local_epoch')==local and p.get('epoch')==local+20,'Contiguous U-quality arm/update clock differs')
 require(p.get('native_packets')==native_packets and p.get('packets')==n and p.get('selected_teacher')=='FROZEN_AS20_EMA' and abs(p.get('weight',-1)-weight)<1e-12,'Fixed raw U prefix/teacher/ramp differs')
 require(p.get('original_selected')==original_selected and type(p.get('extra_original_selected')) is int and 0<=p['extra_original_selected']<=min(n,original_selected),'Original full native U mask or prefix count differs')
 require(p.get('original_mask_unchanged') is True and p.get('old_U_masks_unchanged') is True and p.get('U_TX_truth_read') is False and p.get('urban_confidence_filter') is False and p.get('dropout_rng_restored') is True and p.get('source_health_gate') is False and p.get('subset_rule')=='first_min_128_native_U_batch_no_padding' and p.get('selected_count_normalization') is False and p.get('temperature')==1. and p.get('observation_proxy_is_recoverability_truth') is False,'Hidden U/no student quality hard gate/batch denominator required')
 require(p.get('extra_student_NN')==n and p.get('extra_frozen_teacher_NN')==n and p.get('frozen_teacher_NN')==dict(B01=n),'Exactly one matched extra student/frozen teacher forward required')
 for k in ('KL','weighted_KL','unweighted_KL','q_mean','q_mass'):finite(p[k],k)
 require(p['KL']>=-1e-6 and p['unweighted_KL']>=-1e-6 and abs(p['weighted_KL']-weight*p['KL'])<1e-7 and type(p.get('positive_q')) is int and 0<=p['positive_q']<=n,'Batch-mean KL accounting differs')
 require(isinstance(p.get('selected_physical_ids'),list) and len(p['selected_physical_ids'])==n and all(isinstance(pid,list) and len(pid)==5 and all(type(v) is int for v in pid) and pid[0]==1 and pid[1] in RX and pid[2] in DAYS for pid in p['selected_physical_ids']),'Hidden U source physical prefix IDs required')
 for name in ('q_teacher','q_observation','effective_q'):quality_statistics(p.get(name),n)
 require(abs(p['q_mass']-p['effective_q']['mass'])<1e-5 and abs(p['q_mean']-p['effective_q']['mean'])<1e-6 and p['positive_q']==p['effective_q']['positive'],'Selected allocation q differs')
 m=p.get('mass_match');require(isinstance(m,dict),'Same-mass allocation evidence required')
 for k in ('q_teacher_mass','observation_weighted_mass','mass_uniform_mass','difference','absolute_difference','uniform_multiplier'):finite(m[k],k)
 require(abs(m['q_teacher_mass']-p['q_teacher']['mass'])<1e-5 and 0<=m['uniform_multiplier']<=1 and abs(m['difference']-(m['observation_weighted_mass']-m['mass_uniform_mass']))<1e-7 and abs(m['absolute_difference']-abs(m['difference']))<1e-7 and m['absolute_difference']<3e-5,'Per-update observation/uniform q masses must match')
 require(abs(m['mass_uniform_mass']-m['uniform_multiplier']*m['q_teacher_mass'])<3e-5 and -1e-5<=m['observation_weighted_mass']<=m['q_teacher_mass']+1e-5,'Same-mass teacher scaling differs')
 chosen='mass_uniform_mass' if arm=='MASS_UNIFORM' else 'observation_weighted_mass'
 require(abs(p['q_mass']-m[chosen])<3e-5,'Active q mass must match selected allocation')
 strata=p.get('strata');require(isinstance(strata,list) and strata and sum(s['count'] for s in strata)==n,'Full per-update state/quality strata required')
 require(type(p.get('none_snr_count')) is int and p['none_snr_count']==sum(s['none_snr_count'] for s in strata),'Explicit absent added-noise ratio accounting differs')
 keys=set()
 for s in strata:
  key=(s['scene'],s['state'],s['lock']);require(key not in keys and key[0] in SCENES and key[1] in STATES and key[2] in LOCKS and type(s['count']) is int and s['count']>0 and 0<=s['none_snr_count']<=s['count'],'Bad or duplicate quality stratum');keys.add(key)
  for k in ('q_teacher_mass','q_observation_mass','mass_uniform_mass','observation_weighted_mass','q_mass'):require(-1e-6<=finite(s[k],k)<=s['count']+1e-5,'Stratum q mass outside support')
  if s['count']>s['none_snr_count']:finite(s['signal_to_added_noise_db_mean'],'finite added-noise mean')
  else:require(s['signal_to_added_noise_db_mean'] is None,'All-missing stratum needs explicit None')
 for sk,pk in [('q_teacher_mass','q_teacher'),('q_observation_mass','q_observation'),('q_mass','effective_q')]:require(abs(sum(s[sk] for s in strata)-p[pk]['mass'])<3e-5,'Quality strata mass partition differs '+sk)

def check_quality_summary(summary,records,arm):
 require(isinstance(summary,dict) and len(records)==2220 and summary.get('steps')==2220 and summary.get('arm')==arm,'Complete U-quality summary required')
 for k in ('KL','weighted_KL','native_packets','original_selected','extra_original_selected','extra_student_NN','extra_frozen_teacher_NN'):
  require(abs(summary[k+'_sum']-sum(r[k] for r in records))<1e-5,'Quality summary sum differs '+k)
 require(summary.get('packets_sum')==EXTRA_STUDENT and summary.get('extra_student_NN')==EXTRA_STUDENT and summary.get('extra_frozen_teacher_NN')==EXTRA_FROZEN and summary.get('frozen_teacher_NN')==dict(B01=EXTRA_STUDENT),'Full unpadded prefix/single frozen teacher accounting differs')
 for name in ('q_teacher','q_observation','effective_q'):
  q=quality_statistics(summary['quality_statistics'][name],EXTRA_STUDENT,False)
  require(abs(q['mass']-sum(r[name]['mass'] for r in records))<1e-5 and q['positive']==sum(r[name]['positive'] for r in records) and q['histogram']==[sum(r[name]['histogram'][i] for r in records) for i in range(5)],'Whole-run q aggregation differs '+name)
 for k,rk in [('q_mass_sum','q_mass'),('positive_q_sum','positive_q'),('none_snr_count','none_snr_count')]:require(abs(summary[k]-sum(r[rk] for r in records))<1e-5,'Actual q/None sum differs')
 for k in ('q_teacher_mass','observation_weighted_mass','mass_uniform_mass','difference'):require(abs(summary['mass_match'][k+'_sum']-sum(r['mass_match'][k] for r in records))<1e-5,'Whole-run mass matching sum differs')
 require(abs(summary['mass_match']['max_absolute_difference']-max(r['mass_match']['absolute_difference'] for r in records))<1e-10,'Maximum mass mismatch differs')
 require(summary.get('all_frozen_eval') is True and summary.get('teacher_parameters_registered_in_student') is False and summary.get('student_state_keys')==206 and summary.get('student_parameters')==900722 and summary.get('extra_teacher_training_updates')==0 and summary.get('extra_deployment_parameters')==0,'Frozen external teacher and fixed deployed model shape required')
 require(all(summary.get(k) is True for k in ('native_U_mask_unchanged','all_labeled_As_CE_retained')) and all(summary.get(k) is False for k in ('U_TX_truth_read','source_global_disable','student_urban_confidence_gate','selected_count_normalization','observation_proxy_is_recoverability_truth')),'Original supervision/hidden U/no extra gates required')
 return summary

def teaching_csv_row(record):
 row={k:v for k,v in record.items() if not isinstance(v,(dict,list,tuple))}
 for name in ('q_teacher','q_observation','effective_q'):
  for key in ('mean','mass','positive'):row[name+'_'+key]=record[name][key]
 row.update({'mass_match_'+k:v for k,v in record['mass_match'].items()});return row

def native_cost(folder):
 r=read(Path(folder)/'result.json');c={k:r[k] for k in COUNTERS};require(all(type(v) is int and v>=0 for v in c.values()),'Actual native counters required');native=sum(c.values());extra_student=r.get('extra_student_U_NN',0);extra_frozen=r.get('extra_frozen_teacher_U_NN',0)
 require(all(type(v) is int and v>=0 for v in (extra_student,extra_frozen)),'Separate extra packet counters required')
 if extra_student or extra_frozen:
  require(r.get('native_NN')==native and r.get('observed_student_train_NN_including_extra')==c['student_train_NN']+extra_student and r.get('training_total_NN')==native+extra_student+extra_frozen,'Observed encoder/native/extra NN must reconcile without double counting')
 return dict(**c,native_NN=native,extra_student_U_NN=extra_student,extra_frozen_teacher_U_NN=extra_frozen,training_total_NN=native+extra_student+extra_frozen,wall_seconds=float(r['wall_seconds']))

def epoch_curve(folder,offset):
 folder=Path(folder);sums=Counter();counts=Counter()
 for step in lines(folder/'step_metrics.jsonl.gz'):
  e=step['epoch'];sums[e]+=finite(step['metrics']['train/loss'],'native loss');counts[e]+=1
 records=list(lines(folder/'epoch_metrics.jsonl'));require([x['epoch'] for x in records]==list(range(1,len(records)+1)) and all(counts[x['epoch']]==222 for x in records),'Complete reference epoch/step curves required')
 return [dict(epoch=x['epoch']+offset,source_val_accuracy_pct=x['val_tx_acc'],native_loss=sums[x['epoch']]/222,epoch_wall_seconds=x['epoch_time_s'],lr=x['lr'],grad_backbone=x['train_grad_backbone']) for x in records]
def physics_recount(path,teaching):
 """Read saved simulator metadata only; never load or regenerate any IQ."""
 grouped=defaultdict(lambda:dict(count=0,none_snr_count=0,q_observation_mass=0.,finite_snr_count=0,signal_to_added_noise_db_sum=0.))
 steps=packets=0;max_mass_error=0.
 for metadata,record in zip_longest(lines(path),teaching):
  require(metadata is not None and record is not None and metadata.get('step')==record['cumulative_step'],'Complete matching U physical log required')
  observations=metadata.get('observations');require(isinstance(observations,list) and len(observations)==record['packets'],'U physical observation count differs')
  current=defaultdict(lambda:dict(count=0,none_snr_count=0,q_observation_mass=0.))
  for index,r in enumerate(observations):
   require(r.get('sample_id')==record['selected_physical_ids'][index] and r.get('scene') in SCENES and r.get('state_end') in STATES and r.get('receiver_lock_status') in LOCKS and r.get('channel_equalization_applied') is False,'Saved full nominal U physical identity/state/EQ differs')
   db=r.get('signal_to_added_noise_db');q=0. if db is None else 1./(1.+math.exp(-max(-60.,min(60.,(finite(db,'added-noise ratio')-5.)/5.))))
   key=(r['scene'],r['state_end'],r['receiver_lock_status']);s=current[key];s['count']+=1;s['none_snr_count']+=db is None;s['q_observation_mass']+=q
   qbin='MISSING' if db is None else '<3' if db<3 else '3-8' if db<8 else '8-15' if db<15 else '>=15'
   for dimension,stratum in [('state_lock',key[1]+'/'+key[2]),('state',key[1]),('lock',key[2]),('signal_to_added_noise_db',qbin)]:
    g=grouped[(key[0],dimension,stratum)];g['count']+=1;g['none_snr_count']+=db is None;g['q_observation_mass']+=q
    if db is not None:g['finite_snr_count']+=1;g['signal_to_added_noise_db_sum']+=db
  saved={(s['scene'],s['state'],s['lock']):s for s in record['strata']};require(set(saved)==set(current),'Step quality and actual physical strata differ')
  for key,s in current.items():
   z=saved[key];error=abs(s['q_observation_mass']-z['q_observation_mass']);max_mass_error=max(max_mass_error,error)
   require(s['count']==z['count'] and s['none_snr_count']==z['none_snr_count'] and error<3e-5,'Observation q must match actual saved added-noise ratio')
  steps+=1;packets+=len(observations)
 require(steps==2220 and packets==EXTRA_STUDENT,'Complete2220 physical steps and unpadded prefix support required')
 strata=[dict(scene=k[0],dimension=k[1],stratum=k[2],**s,q_observation_mean=s['q_observation_mass']/s['count'],signal_to_added_noise_db_mean=s['signal_to_added_noise_db_sum']/s['finite_snr_count'] if s['finite_snr_count'] else None) for k,s in sorted(grouped.items())]
 return dict(status='SAVED_PHYSICAL_METADATA_RECOUNTED',steps=steps,packets=packets,max_q_observation_mass_error=max_mass_error,strata=strata,NN=0,IQ_reads=0,checkpoint_reads=0,total_SNR_known=False,recoverability_truth_observed=False)

def training_analysis(project,specs,out):
 old=next(s for s in specs if s['key']=='AS30');oldtail=parse_tail(old['root']/'source');oldordinary=oldtail.pop('ordinary');tails={};curves=[];coverage_rows=[];U_coverage_rows=[];U_sequence=None;U_channel=None
 for new in [s for s in specs if not s['reused']]:
  arm=new['key'];tail=parse_tail(new['root']/'source',arm);seq_n=seq_bad=0
  for a,b in zip_longest(lines(new['root']/'source/training_sequence.jsonl.gz'),lines(old['root']/'source/training_sequence.jsonl.gz')):seq_n+=1;seq_bad+=a!=b
  require(seq_n>0 and seq_bad==0,'Same source physical L/U sequence differs from original As30 tail: '+arm)
  ordinary_bad=sum(a!=b for a,b in zip_longest(tail.pop('ordinary'),oldordinary));require(ordinary_bad==0,'Same ordinary generated IQ MSE stream differs: '+arm)
  result=read(new['root']/'source/result.json');channel=result['L_channel'];require(channel.get('session_count')==15 and set(channel.get('sessions',{}))=={f'rx{rx}:day{day}:eq1' for rx in RX for day in DAYS},'All15 RX/day hardware sessions required');coverage=result['TX_RX_scene_state_lock'];require(sum(x['count'] for x in coverage)==142080,'Actual As state coverage packet count differs')
  require(all(type(x['count']) is int and x['count']>0 and int(x['TX']) in range(6) and int(x['RX']) in RX and x['scene'] in SCENES and x['state'] in STATES and x['lock'] in LOCKS for x in coverage),'Training coverage members/counts differ')
  table=Counter({(int(x['TX']),int(x['RX']),x['scene'],x['state'],x['lock']):int(x['count']) for x in coverage});require(len(table)==len(coverage),'Duplicate training coverage cell')
  full=[]
  for tx in range(6):
   for rx in RX:
    for scene in SCENES:
     require(sum(v for (t,r,s,_,_),v in table.items() if (t,r,s)==(tx,rx,scene))>0,'Missing training TX x source RX x scene support')
     for state in STATES:
      for lock in LOCKS:full.append(dict(row_id=arm,TX=tx,RX=rx,scene=scene,state=state,lock=lock,count=table[(tx,rx,scene,state,lock)]))
  require(all(sum(v for (_,_,s,_,_),v in table.items() if s==scene)>0 for scene in SCENES) and all(sum(v for (_,_,_,st,_),v in table.items() if st==state)>0 for state in STATES) and all(sum(v for (_,_,_,_,lk),v in table.items() if lk==lock)>0 for lock in LOCKS),'All original scenes/states/lock support including unlocked required')
  ordinary_actual=read(new['root']/'source/ordinary_augmentation_actual.json');require(ordinary_actual.get('no_ordinary_before_practical') is True and ordinary_actual.get('RDrop_same_IQ') is True,'No stacked ordinary practical damage/same IQ pair metadata differs')
  teaching=tail.pop('teacher_rows');current_sequence=[dict(step=r['step'],physical_ids=r['selected_physical_ids']) for r in teaching]
  if U_sequence is None:U_sequence=current_sequence
  else:require(current_sequence==U_sequence,'All new arms must use exactly the same raw U prefix sequence')
  current_channel=result['U_channel'];require(current_channel.get('packets')==EXTRA_STUDENT and current_channel.get('calls')==2220 and current_channel.get('session_count')==15 and current_channel.get('dropped')==0 and current_channel.get('confidence_filter') is False and current_channel.get('ordinary_before_satellite') is False and current_channel.get('formal_test_changed') is False,'Nominal full practical hidden U coverage differs')
  if U_channel is None:U_channel=current_channel
  else:require(current_channel==U_channel,'Matched arms must share physical U channel/session observations')
  U_cells=current_channel['strata'];require(sum(x['count'] for x in U_cells)==EXTRA_STUDENT and {x['RX'] for x in U_cells}==set(RX) and {x['day'] for x in U_cells}==set(DAYS) and {x['scene'] for x in U_cells}==set(SCENES),'All source RX/day/scene hidden U support required')
  U_coverage_rows.extend(dict(row_id=arm,**x) for x in U_cells)
  bank=result['training_only_teachers'];require(bank.get('frozen_parameters_and_buffers_unchanged') is True and bank.get('forward_NN')==dict(B01=EXTRA_STUDENT) and bank.get('total_forward_NN')==EXTRA_FROZEN and bank.get('parameters_each')==dict(B01=900722) and bank.get('state_keys_each')==dict(B01=206) and bank.get('trainable_parameters')==0 and bank.get('registered_in_student') is False and bank.get('EMA_updated') is False and bank.get('teacher_checkpoint_key')=='ema_model' and bank.get('teacher_epoch')==20 and bank.get('target_access') is False,'Single unchanged As20 source-only training teacher required')
  physics=physics_recount(new['root']/'source/u_quality_physics.jsonl.gz',teaching)
  csv_write(out/(arm+'_u_quality_steps.csv'),(teaching_csv_row(r) for r in teaching))
  csv_write(out/(arm+'_u_quality_strata.csv'),tail['u_quality_summary']['strata'])
  csv_write(out/(arm+'_u_quality_physical_strata.csv'),physics['strata'])
  tail.update(actual_U_channel_summary=current_channel,training_only_teachers=bank,U_prefix_sequence_matches_all_arms=True)
  tail['physical_metadata_recount']=physics
  tail.update(actual_channel_summary=channel,ordinary_augmentation_actual=ordinary_actual,sequence_records=seq_n,sequence_mismatch=seq_bad,ordinary_MSE_mismatch=ordinary_bad,training_coverage_records=len(full),original_support_retained=True)
  tails[arm]=tail;coverage_rows.extend(full)
 for spec in specs:
  c=tails[spec['key']]['curve'] if not spec['reused'] else epoch_curve(spec['root']/'stage20',0)+epoch_curve(spec['root']/'source',20)
  curves.extend(dict(row_id=spec['key'],reused=spec['reused'],model_seed=SEED,**x) for x in c)
 csv_write(out/'training_curves.csv',curves);csv_write(out/'training_coverage.csv',coverage_rows);csv_write(out/'training_U_coverage.csv',U_coverage_rows)
 summary=dict(status='ALL_NEW_TAILS_VERIFIED',rows=tails,new_updates=4440,new_model_epochs=20,ordinary_streams_match_original=True,sample_sequences_match_original=True,all_new_U_prefixes_match=True,all_new_U_channel_summaries_match=True,U_truth_read=False)
 write(out/'new_tail_training.json',summary);return summary,curves

def formal_analysis(project,specs,out,truth):
 viewroot=Path(project)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views';manifest=read(viewroot/'manifest.json');require(manifest.get('status')=='VALIDATED_ONCE' and manifest.get('count')==COUNT and manifest.get('scenes')==list(VIEWS[1:]) and manifest.get('channel')=='residual/post_sync/noeq' and manifest.get('truth_read') is False and manifest.get('source_target_disjoint') is True and manifest.get('rows_share_observations') is True,'Fixed formal received observations differ')
 with np.load(viewroot/'index.npz',allow_pickle=False) as f:ids=f['ids'].copy()
 data,meta=truth_last_arrays(specs,ids,truth,project);y=meta['y'];rx=meta['rx'];day=meta['day'];cohort=data['NOAS30']['pred']['clean']==y;records=[];effects=[];summary={};partitions=0
 for spec in specs:
  arm=spec['key'];saved=data[arm]['saved_scores'];summary[arm]={}
  for view in VIEWS:
   p=data[arm]['pred'][view];z=data[arm]['logits'][view];current=[]
   masks=[('overall','ALL',np.ones(COUNT,bool))]+[('receiver',r,rx==r) for r in sorted(set(rx))]+[('day',d,day==d) for d in sorted(set(day))]+[('transmitter',str(t),y==t) for t in range(6)]
   for dim,label,mask in masks:
    m=logit_metric(y[mask],z[mask],cohort[mask]);r=dict(row_id=arm,model_seed=SEED,reused=spec['reused'],view=view,dimension=dim,stratum=str(label),**m);current.append(r);records.append(r)
    old=next(x for x in saved['results'] if x['view']==view and x['dimension']==dim and x['stratum']==str(label));require(m['confusion']==old['confusion'] and m['count']==old['query_count'] and abs(m['accuracy']-old['accuracy'])<1e-12 and abs(m['macro_f1']-old['macro_f1'])<1e-12,'Independent formal CM/metric recount differs')
   overall=current[0]
   for dim in ('receiver','day','transmitter'):reconcile_partitions(overall,[r for r in current if r['dimension']==dim]);partitions+=1
   worst=min(r['accuracy'] for r in current if r['dimension']=='receiver')
   summary[arm][view]=dict(accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_RX=worst,worst_RX_ids=[r['stratum'] for r in current if r['dimension']=='receiver' and r['accuracy']==worst],mean_margin_over30=overall['mean_margin_over30'],satellite_retention=overall['satellite_retention'])
 for view in VIEWS:
  for new,control in PAIRS:
   for dim,label,mask in [('overall','ALL',np.ones(COUNT,bool))]+[('receiver',r,rx==r) for r in sorted(set(rx))]+[('day',d,day==d) for d in sorted(set(day))]+[('transmitter',str(t),y==t) for t in range(6)]:effects.append(dict(new=new,control=control,view=view,dimension=dim,stratum=str(label),**paired(y[mask],data[control]['pred'][view][mask],data[new]['pred'][view][mask],cohort[mask])))
 csv_write(out/'formal_metrics.csv',({k:v for k,v in r.items() if k!='confusion'} for r in records));write(out/'formal_confusions.json',records);csv_write(out/'formal_paired_effects.csv',effects)
 pair_summary=[]
 for e in effects:
  if e['dimension']=='overall':
   a=summary[e['new']][e['view']];b=summary[e['control']][e['view']]
   pair_summary.append(dict(new=e['new'],control=e['control'],view=e['view'],accuracy_delta_pp=e['accuracy_delta_pp'],macro_f1_delta_pp=100*(a['macro_f1']-b['macro_f1']),worst_RX_delta_pp=100*(a['worst_RX']-b['worst_RX']),mean_margin_over30_delta=a['mean_margin_over30']-b['mean_margin_over30'],fixed_NOAS30_retention_delta_pp=100*(a['satellite_retention']-b['satellite_retention']),repairs=e['repairs'],new_errors=e['new_errors'],count=e['count']))
 csv_write(out/'formal_pair_summary.csv',pair_summary)
 stage={}
 for arm in NEW_ARMS:
  clean=100*(summary[arm]['clean']['accuracy']-summary['NOAS30']['clean']['accuracy']);urban=float(100*np.mean([summary[arm][v]['accuracy']-summary['NOAS30'][v]['accuracy'] for v in VIEWS[-3:]]))
  stage[arm]=dict(clean_delta_vs_NoAs30_pp=clean,clean_max_drop_pp=.1,clean_within_0p1pp=clean>=-.1-1e-10,urban_mean_vs_NoAs30_pp=urban,urban_mean_vs_As30_pp=float(100*np.mean([summary[arm][v]['accuracy']-summary['AS30'][v]['accuracy'] for v in VIEWS[-3:]])),urban_accuracy_mean=float(np.mean([summary[arm][v]['accuracy'] for v in VIEWS[-3:]])),single_seed_observed_dual_gain=clean>0 and urban>0,reproducible_dual_gain=False)
 result=dict(status='FORMAL_ANALYZED',rows=summary,independent_confusion_recounts=len(records),partition_reconciliations=partitions,all4_rows_all7_verified_before_truth=True,formal_state_labels_available=False,fixed_NOAS30_clean_correct_count=int(cohort.sum()),stage=stage,model_seed_count=1,reproducible_dual_gain=False,target_feedback=False,NN=0,IQ_reads=0,checkpoint_reads=0)
 write(out/'formal_summary.json',result);return result

def cost_analysis(project,specs,source):
 runs=Path(project)/'runs';refs={};new={}
 for spec in specs:
  if spec['reused']:
   refs[spec['key']]={stage:native_cost(spec['root']/stage) for stage in ('stage20','source')}
   refs[spec['key']]['source_readout_NN']=12600 if (spec['root']/'source_qualification/complete.json').is_file() else None
   refs[spec['key']]['formal_prediction_NN']=read(spec['root']/'prediction/complete.json')['actual_NN']
  else:
   q=read(spec['root']/'source_qualification/complete.json');pred=read(spec['root']/'prediction/complete.json')
   require(q.get('status')=='SOURCE_READOUT_COMPLETE' and q.get('NN')==12600 and pred.get('actual_NN')==1176002,'Actual new source/formal NN counters differ')
   new[spec['key']]=dict(native=native_cost(spec['root']/'source'),source_readout_NN=q['NN'],source_readout_wall_seconds=q.get('wall_seconds'),formal_prediction_NN=pred['actual_NN'],formal_prediction_wall_seconds=pred.get('wall_seconds'))
 smoke=read(runs/RUN/'real_checkpoint_smoke.json');require(smoke.get('status')=='PASS' and smoke.get('NN')==4,'Actual four-packet shared remote checkpoint smoke differs')
 parent=refs['AS30']['stage20'];parent_NN=parent['training_total_NN'];native_NN=sum(x['native']['native_NN'] for x in new.values());extra_student=sum(x['native']['extra_student_U_NN'] for x in new.values());extra_frozen=sum(x['native']['extra_frozen_teacher_U_NN'] for x in new.values());training_NN=sum(x['native']['training_total_NN'] for x in new.values());source_NN=sum(x['source_readout_NN'] for x in new.values());formal_NN=sum(x['formal_prediction_NN'] for x in new.values());new_total=training_NN+source_NN+formal_NN+4
 require(all(r['native']['native_NN']==3162962 and r['native']['extra_student_U_NN']==EXTRA_STUDENT and r['native']['extra_frozen_teacher_U_NN']==EXTRA_FROZEN for r in new.values()) and new_total==9839612,'Actual two-row experiment forward cost differs')
 historical=sum(row[s]['training_total_NN'] for row in refs.values() for s in ('stage20','source'))+sum((row['source_readout_NN'] or 0)+row['formal_prediction_NN'] for row in refs.values())
 return dict(new_rows=new,new_native_NN=native_NN,new_extra_student_U_NN=extra_student,new_extra_frozen_teacher_U_NN=extra_frozen,new_training_total_NN=training_NN,new_native_wall_seconds=sum(x['native']['wall_seconds'] for x in new.values()),new_model_epochs=20,new_updates=4440,inherited_As20=parent,student_parent_epochs=20,student_parent_updates=4440,unique_shared_prerequisite_epochs=20,unique_shared_prerequisite_updates=4440,candidate_student_cumulative_epochs_each=30,candidate_student_cumulative_updates_each=6660,candidate_prerequisites_plus_new_epochs_each=30,candidate_prerequisites_plus_new_updates_each=6660,new_source_readout_NN=source_NN,new_formal_prediction_NN=formal_NN,new_source_readout_wall_seconds=sum(x['source_readout_wall_seconds'] for x in new.values()),new_formal_prediction_wall_seconds=sum(x['formal_prediction_wall_seconds'] for x in new.values()),new_remote_smoke_NN=4,new_teacher_training_NN=0,new_total_remote_NN_known=new_total,historical_references=refs,historical_reference_NN_known=historical,unique_total_remote_NN_known=new_total+historical,candidate_lifetime_NN_each={arm:x['native']['training_total_NN']+x['source_readout_NN']+x['formal_prediction_NN']+parent_NN for arm,x in new.items()},candidate_lifetime_smoke_shared_unallocated=True,historical_shared_smoke_NN_unallocated=True,historical_reference_model_epochs=60,historical_reference_updates=13320,unique_total_model_epochs_including_reused=80,unique_total_updates_including_reused=17760,As20_student_ownEMA_frozenTeacher_counted_once=True,reference_reprediction=0,reference_rescoring=0,analysis_NN=0,budget_exhaustion_is_not_convergence=True)

def plots(out,training,source_rows,formal,training_summary=None):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'NOAS30':'#64748b','AS30':'#d97706','MASS_UNIFORM':'#8b5cf6','OBSERVATION_WEIGHTED':'#2563eb'};files=[]
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
  axes[0].legend();fig.suptitle('All recorded epochs; new quality tails only E21-30; no imputed connecting points');save(fig,'training_curves')
  fig,axes=plt.subplots(1,4,figsize=(19,4.7))
  for arm in NEW_ARMS:
   tail=[r for r in training if r['row_id']==arm];xx=[r['epoch'] for r in tail]
   for ax,key in zip(axes,('q_mean','weighted_KL','uniform_multiplier','mass_absolute_difference')):ax.plot(xx,[r[key] for r in tail],label=arm,color=colors[arm],marker='.')
  for ax,title in zip(axes,('Effective mean q (whole U subset)','Weighted extra batch-mean KL','Matched uniform multiplier','Mean per-step mass difference')):ax.set(xlabel='Cumulative epoch',title=title);ax.legend(fontsize=7);ax.grid(alpha=.2)
  fig.suptitle('Matched full-As U and fixed As20 teacher forwards; same q mass, different allocation');save(fig,'u_quality_training')

 if training_summary:
  fig,axes=plt.subplots(1,3,figsize=(18,5))
  for ax,scene in zip(axes,SCENES):
   for i,arm in enumerate(NEW_ARMS):
    seq=training_summary['rows'][arm]['u_quality_summary']['strata'];values=[];counts=[]
    for state in STATES:
     ss=[s for s in seq if s['scene']==scene and s['state']==state];n=sum(s['count'] for s in ss);counts.append(n);values.append(sum(s['q_mass'] for s in ss)/n if n else np.nan)
    ax.bar(np.arange(3)+(i-.5)*.32,values,.32,color=colors[arm],label=arm)
   ax.set(xticks=np.arange(3),xticklabels=[s+'\nn='+str(n) for s,n in zip(STATES,counts)],ylim=(0,1),title=scene,ylabel='Effective q mass / observed packet');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
  fig.suptitle('Hidden-U allocation by simulator state; same whole-step mass, no TX truth / no recoverability labels');save(fig,'u_quality_physical_allocation')

 if source_rows:
  for scene in SCENES[1:]:
   fig,axes=plt.subplots(1,2,figsize=(15,5))
   for ax,dimension in zip(axes,('state_lock','signal_to_added_noise_db')):
    rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dimension];labels=list(dict.fromkeys(r['stratum'] for r in rr));xx=np.arange(len(labels));ns=[]
    for i,(arm,color) in enumerate(colors.items()):
     seq=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels];vals=[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in seq];ax.bar(xx+(i-1.5)*.2,vals,.2,label=arm,color=color)
     if not ns:ns=[r['count'] for r in seq]
    ax.set(xticks=xx,xticklabels=[l+'\nn='+str(n) for l,n in zip(labels,ns)],ylim=(0,104),ylabel='Source accuracy (%)',title=dimension);ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2)
   axes[0].legend();fig.suptitle(scene+'; two repeated source draws, one model seed; empty cells retained');save(fig,scene+'_source_states')
 if formal:
  fig,ax=plt.subplots(figsize=(12,5));xx=np.arange(7)
  for i,(arm,color) in enumerate(colors.items()):ax.bar(xx+(i-1.5)*.2,[100*formal['rows'][arm][v]['accuracy'] for v in VIEWS],.2,label=arm,color=color)
  ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urban','mid urban','low urban'],ylim=(0,103),ylabel='Formal accuracy (%)',title='Fixed received IQ / joint default cosine head / matched E30 / single seed');ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'formal_accuracy')
 return files

def report(out,result):
 formal=result['formal'];cost=result['cost'];rows=formal['rows'];source=result['stored_source_decision']
 text=['# P66：相同质量总量的 U 星地教学分配','','两条新增 E21 至 E30 路线都严格复用原 P43 As20 的学生、自身 EMA 和同一个冻结 As20 EMA 教师。默认 cosine30 头与骨干联合更新，原六次 L 前向、总 L 尺度 1.25、跨 RX SupCon、原 U 门控、全 L 星地 CE 均保持。每个原 U batch 取前 min(128,n) 包，尾 batch 保留124包；每行每包增加一次完整 practical 学生和一次 raw 教师前向。没有新增教师训练、结构或部署参数。','','定义 qT=clip((clean teacher pmax-.95)/.05,0,1)，qN=sigmoid((signal_to_added_noise_db-5)/5)，显式 None 取0。OBSERVATION_WEIGHTED 使用 qT*qN；MASS_UNIFORM 使用 qT*sum(qT*qN)/sum(qT)，分母为0时取0。两项都对全部额外子集取均值，权重在五个本地 epoch 内升至 .16。逐步核对两种分配的 q 总量相等；KL 值或梯度并不因此相等。qN 仅是新增噪声比的连续代理，不是真实总 SNR、教师正确性或可恢复性真值。所有三个原训练场景保留。','','| 环境 | NoAs30 % | As30 % | 相同总量均匀 % | 观测加权 % | 观测−均匀 pp |','|---|---:|---:|---:|---:|---:|']
 for v in VIEWS:
  vals=[100*rows[k][v]['accuracy'] for k in ('NOAS30','AS30',*NEW_ARMS)];delta=100*(rows[CANDIDATE][v]['accuracy']-rows['MASS_UNIFORM'][v]['accuracy']);text.append('| '+v+' | '+' | '.join(f'{x:.6f}' for x in vals)+f' | {delta:+.6f} |')
 text += ['', '| 行 | urban 三场景均值 % | clean−NoAs pp | clean下降≤0.1pp |','|---|---:|---:|---|']
 for arm,stage in formal['stage'].items():text.append(f"| {arm} | {100*stage['urban_accuracy_mean']:.6f} | {stage['clean_delta_vs_NoAs30_pp']:+.6f} | {stage['clean_within_0p1pp']} |")
 text += ['', '源域判定在正式结果前已固定：'+json.dumps(source,ensure_ascii=False),'', '唯一候选是 OBSERVATION_WEIGHTED；MASS_UNIFORM 只作相同前向与 q 总量的分配控制。只有原 source PASS 才支持后续另行固定的总 E50 匹配验证。正式分数只报告，不改变晋级、系数、种子或重跑。source FAIL 仍完成全部七环境。一个模型 seed、两个重复源信道 draw 尚不能证明可重复双提升。','','成本：新增两条10轮尾段，共20 model epochs／4440更新；每个学生累计30轮／6660更新。共同 As20 父训练20轮／4440更新只计一次，同一个 EMA 用作 frozen teacher，不增加另一个教师 E20。历史 NoAs30／As30 两条参照共60轮／13320更新；连同新尾段，唯一训练合计80轮／17760更新。',f"新增 native NN={cost['new_native_NN']}；额外学生 U NN={cost['new_extra_student_U_NN']}；冻结 As20 教师 U NN={cost['new_extra_frozen_teacher_U_NN']}；新训练总 NN={cost['new_training_total_NN']}；源读出 NN={cost['new_source_readout_NN']}；正式预测 NN={cost['new_formal_prediction_NN']}；远端共享 smoke NN={cost['new_remote_smoke_NN']}；本地验证 NN={cost.get('local_verification_NN')}。新增远端合计={cost['new_total_remote_NN_known']}。实际 native 墙钟之和={cost['new_native_wall_seconds']:.3f} 秒；这是并行行耗时之和，不是矩阵经过时间。",'', '新两行之间具有匹配成本；相对旧 As30／NoAs30 参照增加了 U 前向，不能把全部差额归因于逐包分配。预算耗尽不等于收敛。源 V27000 健康与 V1800×2draw 面板分别报告。固定 NoAs30 clean 正确集合用于保持率，它不是 As20 父模型 clean 正确集合。source 高 urban 和完整 V F1 未保存，正式测试逐包信道状态未提供，状态关联不是物理因果。未保存独立教学梯度或全部内部门控轨迹，不作推断。','','逐行 CSV 同时报告七环境 accuracy／Macro-F1／最差 RX、RX／day／TX、五个匹配比较、修复与新增错误、固定正确集合保持及 margin/30。完整2220步尾段日志、10轮曲线、原 L/U 序列、ordinary 输入流、模拟物理状态及质量分配均作 CPU 复算。原始 U TX 真值保持隐藏。','','缺项：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：']
 text += ['- '+str(p) for p in result.get('figures',[])];text += ['', '本分析新增 NN、IQ／checkpoint读取、训练更新、头拟合、重新预测或 scorer 启动均为0。']
 (out/'final_analysis.md').write_text(chr(10).join(text)+chr(10),encoding='utf-8')

def run(project,out,truth=None,code_commit=None,local_NN=None):
 project=Path(project);decision=read_source_decision(project) # First read, before any formal artifacts or truth.
 truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json';out=Path(out);require(not out.exists(),'Immutable new analysis output required; reconcile existing directory');out.mkdir(parents=False);write(out/'stored_source_decision.json',decision);start=time.perf_counter();specs=row_specs(project);missing=[];source=None;sr=[];tr=None;curves=[]
 if local_NN is not None:require(type(local_NN) is int and local_NN>=0,'External local verification NN must be a nonnegative integer')
 # Source stages run first. Their technical partial state does not skip legal formal closure.
 for name,fn in [('training',lambda:training_analysis(project,specs,out)),('source',lambda:source_analysis(project,specs,out,decision))]:
  try:
   value=fn()
   if name=='training':tr,curves=value
   else:
    source,sr=value
    if source['status']!='SOURCE_ANALYZED':missing.append(dict(component='stored_source_decision',error='Producer source decision is PARTIAL; not recomputed'))
  except Exception as error:
   failure=dict(component=name,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True);missing.append(failure);write(out/(name+'_failure.json'),failure)
   if name=='source':
    source=dict(status='SOURCE_ANALYZED_WITH_INPUT_PARTIAL',stored_source_decision=decision,source_decision_recomputed=False,missing_source_artifacts=True,formal_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0)
    if not (out/'source_summary.json').exists():write(out/'source_summary.json',source)
 formal=formal_analysis(project,specs,out,truth);cost=cost_analysis(project,specs,source);cost['local_verification_NN']=local_NN;cost['local_verification_scope']='external implementation/reviewer costs supplied by owner; unknown when omitted';write(out/'costs.json',cost)
 try:
  figures=plots(out,curves,sr,formal,tr)
  if tr and sr:require(len(figures)==12,'Six actual PNG/PDF figure pairs required')
 except Exception as error:figures=[];failure=dict(component='plot',error=repr(error),traceback=traceback.format_exc());missing.append(failure);write(out/'plot_failure.json',failure)
 result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,code_commit=code_commit,source=source,stored_source_decision=decision,formal=formal,cost=cost,missing=missing,figures=figures,training=tr,model_seed_count=1,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,source_decision_recomputed=False,target_feedback=False,crossseed_confirmed=False,goal_complete=False,wall_seconds=time.perf_counter()-start)
 report(out,result);write(out/'summary.json',dict(status=result['status'],run_id=RUN,code_commit=code_commit,rows=formal['rows'],stage=formal['stage'],source=source,stored_source_decision=decision,cost=cost,missing=missing,figures=figures,model_seed_count=1,reproducible_dual_gain=False,target_feedback=False,NN=0,IQ_reads=0,checkpoint_reads=0));write(out/'complete.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--truth',type=Path);p.add_argument('--code-commit');p.add_argument('--local-nn',type=int,help='Recorded external local verification NN; report-only, not a scientific gate');a=p.parse_args();r=run(a.project,a.out,a.truth,a.code_commit,a.local_nn);print(json.dumps(dict(status=r['status'],out=str(a.out),missing_components=[x['component'] for x in r['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)
if __name__=='__main__':main()
