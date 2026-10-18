"""Independent three-seed As baseline analysis; no IQ/checkpoint/model imports.

Source mode never reads formal predictions, scorers or truth. Final mode first
checks all six completed rows and a prior source-only decision, then recounts
fixed logits/decisions and joins truth through the existing independent reader.
"""
import argparse,csv,json,math,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import cvs_impulse_factorial_E30_results_20261018 as audit
base=audit.base
PROJECT=base.PROJECT
RUN='20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
ROOT=PROJECT/'runs'/RUN
SEEDS=(2026092701,2026092702,2026092703)
ARMS=('B00','B01')
VIEWS=base.VIEWS
SCENES=audit.SCENES
PANEL=audit.PANEL

def rows():
 return [dict(key=a+'-s'+str(s),arm=a,seed=s,reused=s==SEEDS[0],run=REF if s==SEEDS[0] else RUN,root=PROJECT/'runs'/(REF if s==SEEDS[0] else RUN)/(a+'-s'+str(s))) for s in SEEDS for a in ARMS]

def write(p,x):base.write(p,x)
def csv_write(p,records):
 if not records:raise ValueError('No fabricated empty table')
 fields=list(dict.fromkeys(k for r in records for k in r))
 with Path(p).open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(records)

@contextmanager
def row_context(row):
 prior=(audit.RUN,audit.ROOT)
 audit.RUN=row['run'];audit.ROOT=row['root'].parent
 try:yield
 finally:audit.RUN,audit.ROOT=prior

def source_complete(row):
 with row_context(row):done=audit.completed_training(row['root'])
 if done['config']['model_seed']!=row['seed'] or done['config']['arm']!=row['arm'] or done['config']['new_Ag'] or done['config']['Ag_mode']!='NONE':raise ValueError('Exact seed/arm/noAg source identity')
 if not row['reused']:
  for segment in ('stage20','source'):
   p=row['root']/segment;s=base.read(p/'seed_binding.json');a=base.read(p/'actual_native_args.json')
   if s['model_seed']!=row['seed'] or s['actual_F_factory_seed']!=row['seed'] or a['seed']!=row['seed'] or s['physical_roles']!='EXACT_MATCH':raise ValueError('Declared seed did not reach actual model/factory')
 q=base.read(row['root']/'source_qualification/complete.json')
 if q['status']!='SOURCE_READOUT_COMPLETE' or q['NN']!=12600 or q['training_updates'] or q['head_fit_updates'] or q['target_access'] or q['formal_scores_read'] or q['config']!=done['config']:raise ValueError('Frozen source-only panel required')
 return done,q

def formal_complete(row):
 source_complete(row)
 with row_context(row):return audit.completed(row['root'])

def require_all_complete(records,checker):
 # This function never opens truth; caller may not proceed after any failure.
 for row in records:checker(row)

def parse_training(row):
 with row_context(row):result=audit.parse_source(row['root'])
 for c in result['curve']:c.update(model_seed=row['seed'],arm=row['arm'],row_id=row['key'],reused_seed01=row['reused'])
 return result

def qualify_source(records):
 if set(records)!=set(SEEDS):raise ValueError('All three model seeds, no best-seed selection')
 out=[]
 for seed in SEEDS:
  a=records[seed]['B00'];b=records[seed]['B01'];ga=100*(np.asarray(b['urban'])[:,1:]-np.asarray(a['urban'])[:,1:]);dr={r:100*(b['rx'][r]-a['rx'][r]) for r in ('1','3','4','6','8')};clean=100*(b['clean']-a['clean'])
  if ga.shape!=(2,2) or not np.isfinite(ga).all() or not math.isfinite(clean) or not all(math.isfinite(x) for x in dr.values()):raise ValueError('Finite complete source rates')
  checks=dict(mid_low_both_draws_at_least5pp=bool((ga>=5).all()),clean_decline_at_most_half_pp=clean>=-.5,each_source_RX_decline_at_most1pp=min(dr.values())>=-1)
  out.append(dict(model_seed=seed,clean_delta_pp=clean,mid_low_delta_pp=ga.tolist(),RX_clean_delta_pp=dr,checks=checks,source_As_replicated=all(checks.values())))
 return dict(records=out,source_As_replicated_all3=all(r['source_As_replicated'] for r in out),formal_scores_read=False,new_Ag=False,Ag_proved=False,goal_complete=False)

def clean_cohort_effect(y,clean_control,pred_new,pred_control):
 y=np.asarray(y);clean_control=np.asarray(clean_control);n=np.asarray(pred_new)==y;o=np.asarray(pred_control)==y;c=clean_control==y
 if any(x.shape!=y.shape for x in [clean_control,pred_new,pred_control]) or y.ndim!=1 or len(y)==0:raise ValueError('Paired decisions must align')
 den=int(c.sum());rep=int((n&~o).sum());br=int((~n&o).sum())
 return dict(repairs=rep,new_errors=br,net=rep-br,accuracy_delta_pp=100*(rep-br)/len(y),fixed_B00_clean_correct_count=den,B00_retained=int((o&c).sum()),B01_retained=int((n&c).sum()),B00_retention=float(o[c].mean()) if den else None,B01_retention=float(n[c].mean()) if den else None)

def paired_summary(records):
 out=[]
 for view in VIEWS:
  found=[r for r in records if r['view']==view]
  if len(found)!=3 or {r['model_seed'] for r in found}!=set(SEEDS):raise ValueError('All3 independent model seeds required')
  x=np.asarray([r['accuracy_delta_pp'] for r in found]);f=np.asarray([r['macro_F1_delta_pp'] for r in found])
  out.append(dict(view=view,models=3,paired_accuracy_mean_pp=float(x.mean()),paired_accuracy_SD_pp=float(x.std(ddof=1)),paired_F1_mean_pp=float(f.mean()),paired_F1_SD_pp=float(f.std(ddof=1)),all_seeds_accuracy_positive=bool((x>0).all()),seed_values_pp={str(r['model_seed']):r['accuracy_delta_pp'] for r in found}))
 return out

def stamp_row_identity(result,row):
 if row['seed'] not in SEEDS or row['arm'] not in ARMS or row['key']!=row['arm']+'-s'+str(row['seed']):raise ValueError('Unregistered row identity')
 result.update(arm=row['arm'],model_seed=row['seed'],reused_seed01=row['reused'])
 for record in result['metrics']:record.update(arm=row['arm'],model_seed=row['seed'],reused_seed01=row['reused'])
 for record in result['views'].values():record.update(arm=row['arm'],model_seed=row['seed'],reused_seed01=row['reused'])
 return result

def compare_training_streams(left,right):
 import gzip
 from itertools import zip_longest
 segments=[]
 for stage in ('stage20','source'):
  count=mismatch=0
  with gzip.open(left/stage/'training_sequence.jsonl.gz','rt',encoding='utf8') as a,gzip.open(right/stage/'training_sequence.jsonl.gz','rt',encoding='utf8') as b:
   for x,y in zip_longest(a,b):
    count+=1;mismatch+=x is None or y is None or json.loads(x)!=json.loads(y)
  segments.append(dict(stage=stage,records_compared=count,mismatched_records=mismatch,identical=mismatch==0))
 return dict(segments=segments,all_identical=all(x['identical'] for x in segments),scope='Opaque L/U sequence identity only;no hidden U labels decoded',no_hash_gate=True)

def source_readout(out):
 physics=None;source_rows=[];source_matrices=[];clean={s:{} for s in SEEDS};training={};curves=[]
 # This source stage has no formal path/truth argument or data access.
 require_all_complete(rows(),source_complete)
 with np.load(PANEL/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
 role=base.read(PANEL/'data_roles.json');all_data={}
 for row in rows():
  root=row['root'];done,q=source_complete(row);parsed=parse_training(row);training[row['key']]=parsed;curves.extend(parsed['curve'])
  m=done['final_source_metrics'];clean[row['seed']][row['arm']]=dict(clean=m['source_val_accuracy'],rx=m['source_val_rx_accuracy'],urban=q['source_accuracies'])
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in f.files}
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.array_equal(x['ids'],role['physical_ids']) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Original source V panel/finite logits')
  if not np.allclose((x['satellite'].argmax(-1)==x['TX']).mean(-1),q['source_accuracies'],atol=1e-12,rtol=0):raise ValueError('Source accuracies differ from independent argmax')
  all_data[row['key']]=x
 for row in rows():
  x=all_data[row['key']];control=all_data['B00-s'+str(row['seed'])]
  if any(not np.array_equal(x[k],control[k]) for k in ('ids','TX','RX','day')):raise ValueError('Paired source panel IDs/roles differ')
  y0=x['TX'];cohort0=control['clean'].argmax(1)==y0
  observations=[('clean',-1,x['clean'],y0,x['RX'],x['day'],cohort0,None,None,None)]
  for si,scene in enumerate(SCENES):
   for draw in (-1,0,1):
    ds=(0,1) if draw==-1 else (draw,)
    observations.append((scene,draw,np.concatenate([x['satellite'][j,si] for j in ds]),np.tile(y0,len(ds)),np.tile(x['RX'],len(ds)),np.tile(x['day'],len(ds)),np.tile(cohort0,len(ds)),np.concatenate([physics['state'][j,si,0] for j in ds]),np.concatenate([physics['lock'][j,si,0] for j in ds]),np.concatenate([physics['signal_to_added_noise_db'][j,si,0] for j in ds])))
  for scene,draw,z,y,rx,day,cohort,state,lock,quality in observations:
   p=z.argmax(1);other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=(z[np.arange(len(y)),y]-other.max(1))/30
   masks=[('overall','ALL',np.ones(len(y),bool))]+[('RX',str(r),rx==r) for r in (1,3,4,6,8)]+[('day',str(d),day==d) for d in (1,2,3)]+[('TX',str(t),y==t) for t in range(6)]
   if state is not None:
    masks += [('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(['LOS','PARTIAL','BLOCKED']) for j,l in enumerate(['locked','degraded','unlocked'])]
    masks += [('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
   for dimension,stratum,mask in masks:
    if not mask.any():continue
    metric=base.metric(y[mask],p[mask]);den=int((mask&cohort).sum());r=dict(row_id=row['key'],model_seed=row['seed'],arm=row['arm'],scene=scene,draw=draw,dimension=dimension,stratum=stratum,**metric,mean_margin_over30=float(margin[mask].mean()),fixed_same_seed_B00_clean_correct_count=den,satellite_retention=float((p==y)[mask&cohort].mean()) if den else None)
    source_matrices.append(r);source_rows.append({k:v for k,v in r.items() if k!='confusion'})
 sequences={}
 for seed in SEEDS:
  pair=[r for r in rows() if r['seed']==seed];record=compare_training_streams(pair[0]['root'],pair[1]['root']);a=training[pair[0]['key']]['ordinary_observed'];b=training[pair[1]['key']]['ordinary_observed']
  if len(a)!=6660 or len(b)!=6660:raise ValueError('Completed ordinary streams required')
  record.update(ordinary_MSE_records_compared=6660,ordinary_MSE_records_different=sum(x!=y for x,y in zip(a,b)),maximum_absolute_ordinary_MSE_delta=max(abs(x['MSE']-y['MSE']) for x,y in zip(a,b)))
  sequences[str(seed)]=record
 write(out/'training_stream_comparability.json',sequences)
 decision=qualify_source(clean);saved=base.read(ROOT/'source_matrix_decision.json')
 if saved['source_As_replicated_all3']!=decision['source_As_replicated_all3']:raise ValueError('Online and independent source decisions differ')
 for found,expected in zip(decision['records'],saved['records']):
  if found['model_seed']!=expected['model_seed'] or not np.allclose(found['mid_low_delta_pp'],expected['source_urban_mid_low_gain_pp'],atol=1e-12,rtol=0) or abs(found['clean_delta_pp']-expected['source_clean_delta_pp'])>1e-12 or any(abs(v-expected['source_clean_RX_delta_pp'][k])>1e-12 for k,v in found['RX_clean_delta_pp'].items()):raise ValueError('Source per-seed decision mismatch')
 csv_write(out/'training_curves.csv',curves);csv_write(out/'source_strata.csv',source_rows);write(out/'source_confusions.json',source_matrices)
 for key,value in training.items():write(out/(key+'_training.json'),value)
 write(out/'source_decision.json',decision);write(out/'source_summary.json',dict(source_unique_V=1800,model_seed_count=3,channel_draws=2,source_confusions=len(source_matrices),source_rows=source_rows,decision=decision,NN=0,target_access=False,added_noise_ratio_is_total_SNR=False,formal_state_accuracy_available=False))
 plot_source(out,source_rows)
 write(out/'complete.json',dict(status='SOURCE_ANALYZED',model_seed_count=3,training_model_rows=6,new_rows=4,reused_rows=2,epoch_records=len(curves),source_confusions=len(source_matrices),formal_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0))
 return training,decision

def plot_source(out,source_rows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 states=['LOS/locked','PARTIAL/locked','PARTIAL/degraded','BLOCKED/degraded','BLOCKED/unlocked']
 fig,axes=plt.subplots(1,3,figsize=(15,4.9),layout='constrained')
 for ax,seed in zip(axes,SEEDS):
  for arm,shift in [('B00',-.18),('B01',.18)]:
   values=[]
   for state in states:
    found=[r for r in source_rows if (r['model_seed'],r['arm'],r['scene'],r['draw'],r['dimension'],r['stratum'])==(seed,arm,'practical_low_urban',-1,'state_lock',state)]
    values.append(100*found[0]['accuracy'] if found else np.nan)
   ax.bar(np.arange(len(states))+shift,values,width=.36,label=arm)
  ax.set(xticks=range(len(states)),xticklabels=states,ylim=(0,100),ylabel='Source V accuracy (%)',title='seed '+str(seed)[-2:]);ax.tick_params(axis='x',rotation=55,labelsize=8);ax.legend();ax.grid(axis='y',alpha=.2)
 fig.suptitle('Source low urban / pooled two fixed channel draws / states are not target labels')
 for ext in ('png','pdf'):fig.savefig(out/('source_lowurban_states.'+ext),dpi=170)
 plt.close(fig)

def checked_final_inputs(source_dir):
 proof=base.read(source_dir/'complete.json')
 if proof['status']!='SOURCE_ANALYZED' or proof['formal_scores_read'] or proof['model_seed_count']!=3:raise ValueError('Independent source-only decision must finish first')
 require_all_complete(rows(),formal_complete)
 return proof

def final_analysis(out,source_dir,truth):
 proof=checked_final_inputs(source_dir)
 # Only now may the existing independent reader access the truth-side file.
 old_completed,old_parse,old_rows=base.completed,base.parse_source,base.ROWS
 cached={r['key']:base.read(source_dir/(r['key']+'_training.json')) for r in rows()}
 byroot={str(r['root']):r for r in rows()}
 base.completed=lambda root:formal_complete(byroot[str(root)])
 base.parse_source=lambda root,done:cached[root.name]
 base.ROWS={r['key']:(r['run'],r['key']) for r in rows()}
 try:
  results={};data={}
  for row in rows():
   result,private=base.collect_row(row['key'],truth)
   # The reused reader carries a historical seed01 display constant only.
   # Replace it with the already verified row seed, never infer from scores.
   stamp_row_identity(result,row)
   results[row['key']]=result;data[row['key']]=private
 finally:base.completed,base.parse_source,base.ROWS=old_completed,old_parse,old_rows
 first=data[rows()[0]['key']]
 if any(not np.array_equal(v['ids'],first['ids']) or not np.array_equal(v['y'],first['y']) for v in data.values()):raise ValueError('All six formal rows must share original physical observations')
 paired=[];retained=[];overall=[];metrics=[];gaps=[]
 for row in rows():
  result=results[row['key']];metrics.extend(result['metrics'])
  for view in VIEWS:
   q=result['views'][view];overall.append(dict(row_id=row['key'],model_seed=row['seed'],arm=row['arm'],view=view,accuracy=q['accuracy'],macro_f1=q['macro_f1'],worst_RX_accuracy=q['worst_RX_accuracy'],mean_margin_over30=q['mean_true_margin_over30']))
  clean=result['views']['clean']['accuracy']
  for view in VIEWS[1:4]:gaps.append(dict(row_id=row['key'],model_seed=row['seed'],arm=row['arm'],view=view,clean_accuracy=clean,suburban_accuracy=result['views'][view]['accuracy'],gap_pp=100*(clean-result['views'][view]['accuracy'])))
 for seed in SEEDS:
  a=data['B00-s'+str(seed)];b=data['B01-s'+str(seed)];y=a['y']
  for view in VIEWS:
   effect=clean_cohort_effect(y,a['pred']['clean'],b['pred'][view],a['pred'][view]);effect.update(model_seed=seed,view=view,macro_F1_delta_pp=100*(results['B01-s'+str(seed)]['views'][view]['macro_f1']-results['B00-s'+str(seed)]['views'][view]['macro_f1']),worst_RX_delta_pp=100*(results['B01-s'+str(seed)]['views'][view]['worst_RX_accuracy']-results['B00-s'+str(seed)]['views'][view]['worst_RX_accuracy']));paired.append(effect)
  for arm in ARMS:
   for view in VIEWS:
    c=a['pred']['clean']==y;p=data[arm+'-s'+str(seed)]['pred'][view]==y;retained.append(dict(model_seed=seed,arm=arm,view=view,denominator=int(c.sum()),retained=int((p&c).sum()),retention=float(p[c].mean()) if c.any() else None))
 summary=paired_summary(paired)
 absolute=[];urban=[]
 for arm in ARMS:
  for view in VIEWS:
   stats=[results[arm+'-s'+str(s)]['views'][view] for s in SEEDS];r=dict(arm=arm,view=view,models=3)
   for field in ('accuracy','macro_f1','worst_RX_accuracy'):
    v=np.asarray([x[field] for x in stats]);r[field+'_mean']=float(v.mean());r[field+'_SD']=float(v.std(ddof=1))
   absolute.append(r)
 for seed in SEEDS:
  for arm in ARMS:urban.append(dict(model_seed=seed,arm=arm,urban_mean=float(np.mean([results[arm+'-s'+str(seed)]['views'][v]['accuracy'] for v in VIEWS[-3:]]))))
 csv_write(out/'absolute_three_seed_summary.csv',absolute);csv_write(out/'urban_means.csv',urban)
 for name,records in [('overall_by_row_seed.csv',overall),('per_row_seed.csv',[{k:v for k,v in q.items() if k!='confusion'} for q in metrics]),('paired_repairs.csv',paired),('fixed_B00_retention.csv',retained),('suburban_gaps.csv',gaps)]:csv_write(out/name,records)
 smoke=base.read(ROOT/'synthetic_smoke.json');cp=base.read(ROOT/'real_checkpoint_smoke.json')
 if smoke['NN']!=3072 or cp['NN']!=2 or not smoke['different_seed_head_and_pool'] or not smoke['matched_initial_state_per_seed']:raise ValueError('Actual crossseed initialization evidence missing')
 costs={key:dict(reused_seed01=result['reused_seed01'],source=result['source']['cost']['source_identity_NN'],prediction=result['source']['cost']['prediction_identity_NN'],total=result['source']['cost']['source_identity_NN']+result['source']['cost']['prediction_identity_NN']) for key,result in results.items()}
 result=dict(status='ANALYZED',run_id=RUN,rows=results,paired_summary=summary,absolute_summary=absolute,urban_means=urban,paired_repairs=paired,fixed_B00_retention=retained,suburban_gaps=gaps,cost=costs,new_training_models=4,reused_models=2,model_seed_count=3,new_model_epochs=120,reused_model_epochs=60,new_updates=26640,reused_updates=13320,shared_new_smoke_NN=3074,new_total_NN=sum(c['total'] for c in costs.values() if not c['reused_seed01'])+3074,reused_total_NN=sum(c['total'] for c in costs.values() if c['reused_seed01']),historical_P43_sixrow_shared_smoke_NN_not_allocated_to_reused_pair=4610,source_decision=base.read(source_dir/'source_decision.json'),formal_confusions=len(metrics),source_confusions=proof['source_confusions'],analysis_NN=0,IQ_reads=0,checkpoint_reads=0,Ag_proved=False,factorial_interaction_not_applicable=True,goal_complete=False,target_feedback=False,interpretation_limits=['As-only baseline confirmation;no newAg or full2x2interaction','Three modelseeds;two fixedsource channel draws are not extra modelseeds','State/lock/quality available only for sourcepanel,not formal perpacket targetstates','Fixed cohort is eachseed B00FINAL30 correct set,not one shared pretrainedfather','Fullbranch/gate trajectories unavailable;no invented gate causation'])
 write(out/'analysis_summary.json',result);plot_final(out,result)
 write(out/'complete.json',dict(status='ANALYZED',model_seed_count=3,new_models=4,reused_models=2,formal_confusions=len(metrics),source_confusions=proof['source_confusions'],NN=0,IQ_reads=0,checkpoint_reads=0,goal_complete=False));return result

def plot_final(out,result):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 labels=['Clean','High sub','Mid sub','Low sub','High urb','Mid urb','Low urb'];colors={2026092701:'#1b9e77',2026092702:'#d95f02',2026092703:'#7570b3'};xx=np.arange(7)
 def save(fig,name):
  for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=170)
  plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,4.7),layout='constrained')
 for seed in SEEDS:
  effect=[next(q for q in result['paired_repairs'] if q['model_seed']==seed and q['view']==view)['accuracy_delta_pp'] for view in VIEWS];ax.plot(xx,effect,'o-',color=colors[seed],label='seed '+str(seed)[-2:])
 avg=[r['paired_accuracy_mean_pp'] for r in result['paired_summary']];sd=[r['paired_accuracy_SD_pp'] for r in result['paired_summary']];ax.errorbar(xx+.06,avg,yerr=sd,fmt='ks--',label='paired mean ± sample SD',capsize=4)
 ax.axhline(0,c='gray',lw=1);ax.set(xticks=xx,xticklabels=labels,ylabel='B01 - B00 accuracy (percentage points)',title='All three model seeds; frozen default head; fixed formal observations');ax.legend();ax.grid(alpha=.2);save(fig,'As_paired_effects')
 fig,axes=plt.subplots(1,3,figsize=(14.5,4.7),layout='constrained')
 for ax,seed in zip(axes,SEEDS):
  for arm,style in [('B00','o--'),('B01','o-')]:
   r=result['rows'][arm+'-s'+str(seed)];ax.plot(xx,[100*r['views'][v]['accuracy'] for v in VIEWS],style,label=arm)
  ax.axhline(60,c='gray',ls=':',lw=.8);ax.set(xticks=xx,xticklabels=labels,ylim=(0,100),title='seed '+str(seed)[-2:],ylabel='Accuracy (%)');ax.tick_params(axis='x',rotation=40);ax.legend();ax.grid(alpha=.2)
 save(fig,'As_absolute_accuracy')
 fig,axes=plt.subplots(2,3,figsize=(14.5,8),layout='constrained')
 for si,seed in enumerate(SEEDS):
  for arm,style in [('B00','--'),('B01','-')]:
   c=result['rows'][arm+'-s'+str(seed)]['source']['curve'];e=[r['epoch'] for r in c];axes[0,si].plot(e,[r['full/RDrop_CE'] for r in c],style,label=arm);axes[1,si].plot(e,[r['source_val_accuracy_pct'] for r in c],style,label=arm)
  axes[0,si].set(title='seed '+str(seed)[-2:],ylabel='L normalized CE',xlabel='Epoch');axes[1,si].set(ylabel='Source V accuracy (%)',xlabel='Epoch')
  for ax in axes[:,si]:ax.axvline(20.5,c='gray',ls=':');ax.grid(alpha=.2);ax.legend()
 fig.suptitle('All 180 model-epochs; source V is health, not unseen-RX proof');save(fig,'As_full_training_curves')
 fig,axes=plt.subplots(1,3,figsize=(14.5,4.7),layout='constrained')
 for ax,seed in zip(axes,SEEDS):
  for arm in ARMS:
   q=[r for r in result['suburban_gaps'] if r['model_seed']==seed and r['arm']==arm];ax.plot(range(3),[r['gap_pp'] for r in q],'o-',label=arm)
  ax.axhline(2,c='gray',ls=':');ax.set(xticks=range(3),xticklabels=['High sub','Mid sub','Low sub'],ylabel='Clean - suburban (percentage points)',title='seed '+str(seed)[-2:]);ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Read gap with absolute accuracy; lower clean is not a breakthrough');save(fig,'As_suburban_gaps')

 fig,axes=plt.subplots(3,3,figsize=(14.5,11),layout='constrained')
 for si,seed in enumerate(SEEDS):
  for arm,style in [('B00','--'),('B01','-')]:
   curves=result['rows'][arm+'-s'+str(seed)]['source']['curve'];e=[r['epoch'] for r in curves];axes[0,si].plot(e,[r['full/RDrop_KL'] for r in curves],style,label=arm);axes[1,si].plot(e,[r['pseudo_selected']/r['pseudo_total'] if r['pseudo_total'] else np.nan for r in curves],style,label=arm);axes[2,si].plot(e,[r['grad_total'] for r in curves],style,label=arm)
  for ax in axes[:,si]:ax.axvline(20.5,c='gray',ls=':');ax.grid(alpha=.2);ax.legend();ax.set(xlabel='Epoch')
  axes[0,si].set(title='seed '+str(seed)[-2:],ylabel='Existing RDrop KL');axes[1,si].set(ylabel='Original U accepted fraction');axes[2,si].set(ylabel='Logged total gradient norm')
 fig.suptitle('Full training diagnostics; accepted U count is not pseudo-label precision');save(fig,'As_optimization_curves')
 fig,axes=plt.subplots(2,3,figsize=(15,9.3),layout='constrained')
 for si,seed in enumerate(SEEDS):
  a=result['rows']['B00-s'+str(seed)];b=result['rows']['B01-s'+str(seed)];rxs=sorted({r['stratum'] for r in b['metrics'] if r['dimension']=='receiver'})
  matrices={}
  for arm,row in [('B00',a),('B01',b)]:matrices[arm]=np.asarray([[100*next(r['accuracy'] for r in row['metrics'] if (r['view'],r['dimension'],r['stratum'])==(v,'receiver',rx)) for v in VIEWS] for rx in rxs])
  for ri,(matrix,title,cmap,lo,hi) in enumerate([(matrices['B01'],'B01 absolute','YlGnBu',0,100),(matrices['B01']-matrices['B00'],'B01 - B00','RdBu_r',-50,50)]):
   ax=axes[ri,si];im=ax.imshow(matrix,vmin=lo,vmax=hi,cmap=cmap,aspect='auto');ax.set(xticks=range(7),xticklabels=labels,yticks=range(len(rxs)),yticklabels=['RX'+r for r in rxs],title=title+' / seed '+str(seed)[-2:]);ax.tick_params(axis='x',rotation=50,labelsize=8)
   for j in range(len(rxs)):
    for k in range(7):ax.text(k,j,f'{matrix[j,k]:.1f}',ha='center',va='center',fontsize=7)
   fig.colorbar(im,ax=ax,shrink=.8,label='Accuracy (%)' if ri==0 else 'Change (pp)')
 fig.suptitle('All fixed formal RX; source state labels are not assigned to these target packets');save(fig,'As_formal_RX_accuracy')

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source','final'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-analysis',type=Path);p.add_argument('--truth',type=Path);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Immutable analysis output already exists')
 if a.mode=='source':
  if a.truth is not None:raise ValueError('Source mode never accepts truth-side input')
  require_all_complete(rows(),source_complete)
 else:
  if a.truth is None or a.source_analysis is None:raise ValueError('Explicit completedsource and truthside required')
  checked_final_inputs(a.source_analysis)
 a.output.mkdir();started=time.perf_counter();result=source_readout(a.output) if a.mode=='source' else final_analysis(a.output,a.source_analysis,a.truth)
 write(a.output/'execution.json',dict(mode=a.mode,wall_seconds=time.perf_counter()-started,NN=0,IQ_reads=0,checkpoint_reads=0,target_access=a.mode=='final'))
 print(json.dumps(dict(status='SOURCE_ANALYZED' if a.mode=='source' else 'ANALYZED',model_seed_count=3,NN=0,output=str(a.output))))
if __name__=='__main__':main()
