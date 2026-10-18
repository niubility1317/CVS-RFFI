"""Zero-model E30/E50 source/formal recount; one immutable CPU owner."""
import argparse,csv,gzip,json,os,re,sys,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_satellite_dropout_results_20261018 as stable
import cvs_satellite_dropout_logs_20261018 as logs
RUN='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
REFERENCE_RUN=stable.REFERENCE_RUN
OLD_ANALYSIS_RUN=stable.RUN
TRAINING_RELEASE='cvs_FcrossRX_As_budget_E50_20261018_r01'
VIEWS=stable.VIEWS;SOURCE_SCENES=stable.SOURCE_SCENES;SEED=2026092701;COUNT=168000
PAIRS=(('AS50','NOAS50'),('AS30','NOAS30'),('AS50','AS30'),('NOAS50','NOAS30'))
read=stable.read;write=stable.write;metric=stable.metric;paired=stable.paired;write_csv=stable.write_csv
AwaitingArtifacts=stable.AwaitingArtifacts

def rows(project):
 p=Path(project)/'runs'
 return [('NOAS30',p/REFERENCE_RUN/'B00-s2026092701'),('AS30',p/REFERENCE_RUN/'B01-s2026092701'),('NOAS50',p/RUN/'B00-s2026092701'),('AS50',p/RUN/'B01-s2026092701')]

def ensure_metadata(project,stage):
 result={}
 for arm,root in rows(project):
  names=['source/completion.json','source/source_frozen.json','source_qualification/complete.json']
  if stage=='final':names+=['prediction/complete.json','scoring/complete.json','complete.json']
  found={}
  for name in names:
   try:found[name]=read(root/name)
   except (FileNotFoundError,json.JSONDecodeError) as exc:raise AwaitingArtifacts(arm+'/'+name) from exc
  c=found['source/completion.json'];f=found['source/source_frozen.json'];q=found['source_qualification/complete.json'];new=arm.endswith('50')
  if (c['epoch'],c['optimizer_steps'])!=((20,4440) if new else (10,2220)) or c['target_access'] or c.get('head_fit_updates',0) or c['config']['model_seed']!=SEED or c['config']['row_id']!=root.name or c['config']['run_id']!=(RUN if new else REFERENCE_RUN):raise ValueError('Actual fixed row/budget/defaulthead differs')
  if new and (c['cumulative_epochs']!=50 or c['cumulative_optimizer_steps']!=11100):raise ValueError('Cumulative E50 metadata differs')
  if f['selection']!='final_only' or f['target_access'] or f['config']!=c['config'] or q['target_access'] or q['NN']!=12600 or q.get('training_updates',0) or q.get('head_fit_updates',0):raise ValueError('Fixed final source readout/freeze differs')
  if stage=='final':
   p=found['prediction/complete.json'];s=found['scoring/complete.json']
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed:every logical row needs all7 frozen predictions and independent scores')
  result[arm]=found
 return result

def compare_values(actual,expected,path='value'):
 if isinstance(expected,dict):
  if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Saved aggregate keys differ: '+path)
  for k,v in expected.items():compare_values(actual[k],v,path+'.'+k)
 elif isinstance(expected,(list,tuple)):
  if not isinstance(actual,(list,tuple)) or len(actual)!=len(expected):raise ValueError('Saved aggregate shape differs: '+path)
  for i,v in enumerate(expected):compare_values(actual[i],v,path+'['+str(i)+']')
 elif isinstance(expected,(float,np.floating)):
  if not isinstance(actual,(int,float)) or not np.isfinite(actual) or abs(actual-expected)>1e-10:raise ValueError('Saved aggregate number differs: '+path)
 elif actual!=expected:raise ValueError('Saved aggregate value differs: '+path)

def source_analysis(project):
 started=time.perf_counter();meta=ensure_metadata(project,'source');data={};summary={};metrics=[];producer_rows={}
 for arm,root in rows(project):
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in f.files}
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Source fixed panel/logits incomplete')
  if data and any(not np.array_equal(x[k],data['NOAS30']['x'][k]) for k in ('ids','TX','RX','day')):raise ValueError('Same source panel physical roles required')
  if len({tuple(np.atleast_1d(v).tolist()) for v in x['ids']})!=1800:raise ValueError('Source physical panel must be unique')
  q=meta[arm]['source_qualification/complete.json'];acc=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if not np.allclose(acc,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Source original scorer differs')
  native=meta[arm]['source/completion.json']['final_source_metrics']
  if native['source_val_count']!=27000:raise ValueError('Original fullV native count differs')
  pred={'clean':x['clean'].argmax(-1)}
  for i,s in enumerate(SOURCE_SCENES):pred[s]=x['satellite'][:,i].argmax(-1).reshape(-1)
  panel={};data[arm]=dict(x=x,pred=pred)
  for scene,p in pred.items():
   y=x['TX'] if scene=='clean' else np.tile(x['TX'],2);rx=x['RX'] if scene=='clean' else np.tile(x['RX'],2)
   m=metric(y,p);worst=min(float((p[rx==r]==y[rx==r]).mean()) for r in np.unique(rx));panel[scene]=dict(accuracy=m['accuracy'],macro_F1=m['macro_f1'],worst_RX=worst)
  producer_rows[arm]=dict(full_V_clean=native['source_val_accuracy'],full_V_RX_accuracy=native['source_val_rx_accuracy'],panel=panel,mid_low_mean=float(np.mean([panel[s]['accuracy'] for s in SOURCE_SCENES[1:]])))
  summary[arm]=dict(**producer_rows[arm],mid_low_by_draw_accuracy=acc[:,1:].tolist(),source_readout_NN=q['NN'])
  observations=[('clean',-1,x['clean'],x['TX'],x['RX'],x['day'])]
  for si,scene in enumerate(SOURCE_SCENES):
   for draw in (-1,0,1):
    ds=(0,1) if draw==-1 else (draw,);observations.append((scene,draw,np.concatenate([x['satellite'][d,si] for d in ds]),np.tile(x['TX'],len(ds)),np.tile(x['RX'],len(ds)),np.tile(x['day'],len(ds))))
  for scene,draw,z,y,rx,day in observations:
   p=z.argmax(1)
   for dim,values in [('overall',['ALL']),('RX',sorted(set(rx))),('day',sorted(set(day))),('TX',range(6))]:
    for value in values:
     mask=np.ones(len(y),bool) if dim=='overall' else rx==value if dim=='RX' else day==value if dim=='day' else y==value
     if mask.any():metrics.append(dict(arm=arm,scene=scene,draw=draw,dimension=dim,stratum=str(value),**metric(y[mask],p[mask])))
 pairs={};producer_pairs={};cohort=data['NOAS30']['pred']['clean']==data['NOAS30']['x']['TX']
 for a,b in PAIRS:
  pairs[a+'__'+b]={};producer_pairs[a+'_minus_'+b]={}
  for scene in ('clean',*SOURCE_SCENES):
   y=data[a]['x']['TX'] if scene=='clean' else np.tile(data[a]['x']['TX'],2);c=cohort if scene=='clean' else np.tile(cohort,2)
   q=paired(y,data[b]['pred'][scene],data[a]['pred'][scene],c);pairs[a+'__'+b][scene]=q
   producer_pairs[a+'_minus_'+b][scene]=dict(repaired=q['repairs'],new_errors=q['new_errors'],net=q['net_repairs'],accuracy_pp=q['accuracy_delta_pp'],count=q['count'],macro_F1_pp=100*(producer_rows[a]['panel'][scene]['macro_F1']-producer_rows[b]['panel'][scene]['macro_F1']),worst_RX_pp=100*(producer_rows[a]['panel'][scene]['worst_RX']-producer_rows[b]['panel'][scene]['worst_RX']))
 clean50=100*(summary['AS50']['full_V_clean']-summary['NOAS50']['full_V_clean']);clean30=100*(summary['AS30']['full_V_clean']-summary['NOAS30']['full_V_clean'])
 interaction={s:pairs['AS50__NOAS50'][s]['accuracy_delta_pp']-pairs['AS30__NOAS30'][s]['accuracy_delta_pp'] for s in ('clean',*SOURCE_SCENES)}
 expected=dict(rows=producer_rows,paired=producer_pairs,fullV_As50_minus_NoAs50_clean_pp=clean50,source_clean_reference_within_0p1pp=clean50>=-.1-1e-12,fullV_clean_budget_interaction_pp=clean50-clean30,panel_accuracy_budget_interaction_pp=interaction,mid_low_mean_budget_interaction_pp=100*((summary['AS50']['mid_low_mean']-summary['NOAS50']['mid_low_mean'])-(summary['AS30']['mid_low_mean']-summary['NOAS30']['mid_low_mean'])))
 saved=read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
 if saved['status']!='SOURCE_BUDGET_READOUT_COMPLETE' or saved['target_scores_read']:raise ValueError('Source budget result not complete')
 for k,v in expected.items():compare_values(saved[k],v,'source.'+k)
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,paired=pairs,decision=expected,source_unique_V=1800,channel_draws=2,model_seed_count=1,fullV_metrics_from_native_not_individual_prediction_recount=True,source_high_urban_available=False,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,wall_seconds=time.perf_counter()-started)

def actual_counter_cost(result):
 keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')
 if any(type(result.get(k)) is not int or result[k]<0 for k in keys):raise ValueError('Actual native four counters required')
 counters={k:result[k] for k in keys};total=sum(counters.values())
 if result.get('native_NN')!=total:raise ValueError('Declared native NN differs from actual counters')
 return dict(**counters,native_training_and_validation_NN=total)

def parse_new_logs(root,arm):
 b=root/'source';done=read(b/'completion.json');r0=read(b/'result.json');cfg=done['config'];sat=arm=='AS50'
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or cfg['updates']!=4440 or r0['optimizer_steps']!=4440 or r0['cumulative_epochs']!=50 or r0['cumulative_optimizer_steps']!=11100:raise ValueError('Fixed actual E31..50 budget differs')
 if any(r0[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or r0['L_NN']!=4440*768 or r0['L_As']!=4440*(64 if sat else 0):raise ValueError('Actual source data/view budget differs')
 cost=actual_counter_cost(r0);epochs,telemetry=logs.native_epochs(b,20);sums=defaultdict(Counter);count=Counter();mse=[];maxerr=0.
 streams=[logs.records(b/n) for n in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(q is None for q in items):raise logs.IncompleteTrainingLogs('E50 source step stream prefix')
  if i>4440:raise ValueError('E50 extra update')
  r,s,n,u=items;ep=(i-1)//222+1;m=n['metrics']
  if any(q['step']!=i for q in items) or any(q['cumulative_step']!=6660+i for q in (r,s,u)) or n['epoch']!=ep or u['physical_clip_calls']!=i:raise ValueError('E50 contiguous cumulative counters differ')
  logs.scan_numbers(items)
  if (r['L_forwards'],r['As_packets'],r['Ag_packets'],r['extra_control_packets'])!=(6,64 if sat else 0,0,128) or r['Ag_mode']!='NONE' or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('E50 original sixforward scope differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('E50 raw control slot differs')
  for prefix,v in [('common',r),('extra',r),('shape',s)]:maxerr=max(maxerr,logs._near(v[prefix+'_RD'],v[prefix+'_CE']+v[prefix+'_KL'],'Original RDrop arithmetic differs'))
  logs._near(r['SupCon_weight'],.1,'Cumulative E31..50 SupCon differs',1e-12);valid=logs.relation(r,True)
  total=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon']+.25*s['shape_RD'];maxerr=max(maxerr,logs._near(total,r['total_L'],'Full 1.25L reduction differs'),logs._near(r['total_L'],r['base_L_before_shape']+.25*s['shape_RD'],'Third pair reduction differs'))
  if m['concat/L_forwards']!=4 or m['concat/As_packets']!=64 or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Original U hidden/numerics differ')
  logs._near(m['concat/total_L'],r['base_L_before_shape'],'Inherited4L telemetry differs');logs._near(m['shape/RD'],s['shape_RD'],'Raw slot telemetry differs');logs.finite(m['train/loss'],'native loss')
  number=m['train/pseudo_total'];selected=m['train/pseudo_selected']
  if int(number)!=number or not 0<number<=256 or not 0<=selected<=number:raise ValueError('Actual U packet counts differ')
  count['U_forward_packets']+=int(number);count['U_selected']+=int(selected);count['steps']+=1;count['L_NN']+=768;count['As_packets']+=r['As_packets'];count['SupCon_valid_anchor_terms']+=valid
  if sum(r['scene_counts'].values())!=64 or sum(r['state_counts'].values())!=64:raise ValueError('Complete 64packet channel observations required')
  if sat and set(r['scene_counts'])-set(SOURCE_SCENES):raise ValueError('Unexpected practical training scene')
  mse.append(r['ordinary_raw_MSE']);sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','SupCon_weight','ordinary_raw_MSE')});sums[ep].update({k:s[k] for k in ('shape_CE','shape_KL','shape_RD')});sums[ep]['native_loss']+=m['train/loss']
 if count['steps']!=4440:raise logs.IncompleteTrainingLogs('E50 last update not present')
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter()
 for q in logs.records(b/'training_sequence.jsonl.gz'):
  role=q['role'];ids=q['ids']
  if role not in unique:raise ValueError('Only original L/U loader roles')
  if len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Actual fullsource loader packet size differs')
  for z in ids:
   if len(z)!=5 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source RX/day metadata differs')
   unique[role].add(tuple(z))
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=4440,U=4440) or packets!=dict(L=568320,U=1134000) or len(unique['L'])!=6300 or len(unique['U'])!=56700 or unique['L']&unique['U'] or count['U_forward_packets']!=packets['U']:raise ValueError('Complete actual source L6300/U56700 sequence differs')
 if cost['student_train_NN']!=count['L_NN']+count['U_forward_packets'] or cost['teacher_eval_NN']!=count['U_forward_packets'] or cost['teacher_train_NN'] or cost['student_eval_NN']!=20*27000+54002:raise ValueError('Observed source counters do not reconcile with complete loops')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error in completed own log')
 args=read(b/'actual_native_args.json');args=args.get('args',args)
 ordinary=read(b/'ordinary_augmentation_actual.json')
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ']:raise ValueError('Actual ordinary augmentation schedule differs')
 curves=[dict(epoch=q['epoch']+30,local_epoch=q['epoch'],**{k:v/222 for k,v in sums[q['epoch']].items()},**{k:q[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for q in epochs]
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=r0['wall_seconds'],local_epochs=20,local_updates=4440,parent_epochs=30,parent_updates=6660,cumulative_epochs=50,cumulative_updates=11100,L_NN=r0['L_NN'],As_packets=r0['L_As']),counts=dict(count),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True),teacher_counter_offset=r0['teacher_counter_offset']),dict(args=args,ordinary_mse=mse)

def training_analysis(project):
 historical=read(Path(project)/'runs'/OLD_ANALYSIS_RUN/'analysis_auto_r03/training_logs.json')
 if historical['status']!='VERIFIED_COMPLETE_SOURCE_TRAINING_LOGS':raise ValueError('Only completed existing E30 recount')
 result={};private={};lookup=dict(rows(project))
 for arm,old in [('NOAS30','B00'),('AS30','B01')]:
  value=historical['rows'][old]
  if value['cost']['new_updates']!=6660 or len(value['epoch_curves'])!=30:raise ValueError('Existing E30 complete cost scope differs')
  result[arm]=dict(reused=True,cost=value['cost'],epoch_curves=value['epoch_curves'],evidence='Existing satelliteDropout analysis_auto_r03 B00/B01 full recount;old steps not reopened')
 for arm in ('NOAS50','AS50'):result[arm],private[arm]=parse_new_logs(lookup[arm],arm)
 sequence=logs.compare_streams(lookup['NOAS50']/'source/training_sequence.jsonl.gz',lookup['AS50']/'source/training_sequence.jsonl.gz')
 if not sequence['all_equal'] or private['NOAS50']['ordinary_mse']!=private['AS50']['ordinary_mse']:raise ValueError('New matched L/U/ordinary sequence differs')
 a,b=private['NOAS50']['args'],private['AS50']['args'];different=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
 if set(different)-{'candidate_id','output_dir','baseline_ckpt'}:raise ValueError('New native training parameters differ')
 for arm in ('NOAS50','AS50'):
  c=result[arm]['cost'];parent=result[arm.replace('50','30')]['cost'];c['cumulative_native_NN']=parent['native_training_and_validation_NN']+c['native_training_and_validation_NN']
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=result,matched_new_rows=dict(physical_sequence=sequence,ordinary_MSE_records=4440,ordinary_MSE_mismatch=0,different_actual_arg_keys=different),new_training_native_NN=sum(result[a]['cost']['native_training_and_validation_NN'] for a in ('NOAS50','AS50')),parent_native_NN=sum(result[a]['cost']['native_training_and_validation_NN'] for a in ('NOAS30','AS30')),new_model_epochs=40,new_updates=8880,parent_model_epochs=60,parent_updates=13320,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,target_scores_read=False)

def formal_analysis(project,truth_path,source):
 started=time.perf_counter();metadata=ensure_metadata(project,'final')
 if source['status']!='SOURCE_ANALYZED' or source['run_id']!=RUN or source['target_scores_read']:raise ValueError('Complete independent source analysis first')
 # Only now may fixed predictions be joined to truth.
 truth=read(truth_path);results={};private={};all_metrics=[];partitions=0
 for arm,root in rows(project):
  with np.load(root/'prediction/predictions.npz',allow_pickle=False) as f:
   if set(f.files)!={'ids',*VIEWS}:raise ValueError('Exact seven decisions required')
   ids=f['ids'].copy();pred={v:f[v].copy() for v in VIEWS}
  if len(ids)!=COUNT or len(set(ids.tolist()))!=COUNT or set(truth)!=set(ids.tolist()):raise ValueError('Exact frozen physical IDs/truth alignment required')
  y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids]);day=np.asarray([str(truth[s]['day']) for s in ids]);saved=read(root/'scoring/scores.json')
  if not saved['truth_last'] or not saved['target_feedback_forbidden']:raise ValueError('Independent scorer boundary differs')
  if private and (not np.array_equal(ids,private['NOAS30']['ids']) or not np.array_equal(y,private['NOAS30']['y'])):raise ValueError('Same paired physical samples required')
  metrics=[];views={}
  for view in VIEWS:
   with np.load(root/'prediction'/(view+'_scores.npz'),allow_pickle=False) as f:
    logits=f['logits']
    if not np.array_equal(ids,f['ids']) or logits.shape!=(COUNT,6) or not np.isfinite(logits).all() or not np.array_equal(logits.argmax(1),pred[view]):raise ValueError('Default logits/decisions differ')
    other=logits.copy();other[np.arange(COUNT),y]=-np.inf;margin=(logits[np.arange(COUNT),y]-other.max(1))/30
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    subtotal=np.zeros((6,6),dtype=np.int64)
    for value in values:
     mask=np.ones(COUNT,dtype=bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=metric(y[mask],pred[view][mask]);old=next(r for r in saved['results'] if r['view']==view and r['dimension']==dim and r['stratum']==str(value))
     if q['confusion']!=old['confusion'] or q['count']!=old['query_count'] or abs(q['accuracy']-old['accuracy'])>1e-12 or abs(q['macro_f1']-old['macro_f1'])>1e-12:raise ValueError('Independent formal metric differs')
     metrics.append(dict(arm=arm,row_id=root.name,model_seed=SEED,view=view,dimension=dim,stratum=str(value),**q));subtotal+=np.asarray(q['confusion'])
    if dim!='overall':
     expected=metric(y,pred[view])['confusion']
     if not np.array_equal(subtotal,expected):raise ValueError('Formal stratum partition differs')
     partitions+=1
   overall=next(q for q in metrics if q['view']==view and q['dimension']=='overall');views[view]=dict(overall,mean_true_margin_over30=float(margin.mean()),margin_quantiles_over30=np.quantile(margin,[0,.1,.5,.9,1]).tolist(),worst_RX_accuracy=min(q['accuracy'] for q in metrics if q['view']==view and q['dimension']=='receiver'))
  results[arm]=dict(views=views,urban_mean_accuracy=float(np.mean([views[v]['accuracy'] for v in VIEWS[4:]])),prediction_NN=metadata[arm]['prediction/complete.json']['actual_NN'],prediction_wall_seconds=metadata[arm]['prediction/complete.json']['wall_seconds'],reused=arm.endswith('30'));private[arm]=dict(ids=ids,y=y,rx=rx,day=day,pred=pred);all_metrics.extend(metrics)
 pairs={};comparisons={};cohort=private['NOAS30']['pred']['clean']==private['NOAS30']['y'];stage={}
 for new,old in PAIRS:
  key=new+'__'+old;pairs[key]={};comparisons[key]={}
  for view in VIEWS:
   pairs[key][view]=paired(private[new]['y'],private[old]['pred'][view],private[new]['pred'][view],cohort);comparisons[key][view]=dict(accuracy_delta_pp=pairs[key][view]['accuracy_delta_pp'],macro_F1_delta_pp=100*(results[new]['views'][view]['macro_f1']-results[old]['views'][view]['macro_f1']),worst_RX_delta_pp=100*(results[new]['views'][view]['worst_RX_accuracy']-results[old]['views'][view]['worst_RX_accuracy']))
  comparisons[key]['urban_mean_accuracy_delta_pp']=float(np.mean([comparisons[key][v]['accuracy_delta_pp'] for v in VIEWS[4:]]))
 for budget in (30,50):
  key='AS'+str(budget)+'__NOAS'+str(budget);diff=pairs[key]['clean'];gain=diff['accuracy_delta_pp']
  stage[str(budget)]=dict(clean_delta_pp=gain,clean_absolute_0p1pp_guard_met=stable.stage_clean_pass(diff['net_repairs'],COUNT),clean_net_repairs=diff['net_repairs'],clean_positive_observed=gain>0,positive_below_0p1pp_is_micro_observation=0<gain<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 value=dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,comparisons=comparisons,stage=stage,suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,source_decision=source['decision'],independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,target_feedback=False,goal_complete=False,wall_seconds=time.perf_counter()-started)
 interaction={v:comparisons['AS50__NOAS50'][v]['accuracy_delta_pp']-comparisons['AS30__NOAS30'][v]['accuracy_delta_pp'] for v in VIEWS}
 interaction['urban_mean']=comparisons['AS50__NOAS50']['urban_mean_accuracy_delta_pp']-comparisons['AS30__NOAS30']['urban_mean_accuracy_delta_pp']
 value['budget_interaction_pp']=interaction
 auto=read(Path(project)/'runs'/RUN/'automatic_summary.json')
 for arm in results:
  for view in VIEWS:
   q=results[arm]['views'][view];compare_values(auto['summary'][arm][view],dict(accuracy=q['accuracy'],macro_F1=q['macro_f1'],worst_RX=q['worst_RX_accuracy']),'automatic.'+arm+'.'+view)
  compare_values(auto['summary'][arm]['urban_mean']['accuracy'],results[arm]['urban_mean_accuracy'])
 compare_values(auto['budget_interaction_pp'],interaction,'automatic.budget_interaction')
 for new,old in PAIRS:
  expected={v:pairs[new+'__'+old][v]['accuracy_delta_pp'] for v in VIEWS};expected['urban_mean']=comparisons[new+'__'+old]['urban_mean_accuracy_delta_pp']
  compare_values(auto['paired_accuracy_pp'][new+'_minus_'+old],expected)
 if auto['stage_clean_0p1pp']!=stage['50']['clean_absolute_0p1pp_guard_met']:raise ValueError('Absolute0.1pp stage differs')
 return value,private



def cost_summary(project,training,formal):
 meta=ensure_metadata(project,'final');allrows={}
 for arm,_ in rows(project):
  native=training['rows'][arm]['cost']['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json']
  allrows[arm]=dict(native_NN=native,frozen_source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],total_for_this_training_segment_and_readouts_NN=native+q['NN']+p['actual_NN'],native_cost=training['rows'][arm]['cost'],prediction_wall_seconds=p['wall_seconds'],reused=arm.endswith('30'))
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=smoke['NN']+real['NN']
 return dict(rows=allrows,new_segment_and_readouts_NN=sum(allrows[a]['total_for_this_training_segment_and_readouts_NN'] for a in ('NOAS50','AS50')),shared_smoke_NN=shared,new_total_NN_including_shared_smoke=sum(allrows[a]['total_for_this_training_segment_and_readouts_NN'] for a in ('NOAS50','AS50'))+shared,parent_training_and_readouts_NN=sum(allrows[a]['total_for_this_training_segment_and_readouts_NN'] for a in ('NOAS30','AS30')),new_model_epochs=40,new_updates=8880,parent_model_epochs=60,parent_updates=13320,analysis_NN=0,teacher_is_original_own_EMA=True,wall_scope='Per-model native wall includes validation and preparation;sum is model-seconds,not concurrent elapsed;no speed claim')

def render_report(source,formal,training,costs):
 lines=['# 完整 practical 配方：匹配 E30→E50 预算实验','',RUN+'；单种子2026092701。原As完整KL/dropout保留，F/defaultcos/joint160/NoExplicitCFO与原L6300/U56700隐藏/V27000不变。','', 'E50是各自固定FINAL30 student/EMA追加20epoch，resetAdamW/cos20；不能把额外预算全部归因于As。阶段目标为同预算As−NoAs clean下降不超过0.1个绝对百分点，最终仍需可重复clean正收益。','', '|视图|NOAS30 %|AS30 %|NOAS50 %|AS50 %|As50−NoAs50 pp|As50−As30 pp|预算交互 pp|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for v in VIEWS:
  vals=[formal['summary'][a]['views'][v]['accuracy']*100 for a in ('NOAS30','AS30','NOAS50','AS50')]+[formal['comparisons']['AS50__NOAS50'][v]['accuracy_delta_pp'],formal['comparisons']['AS50__AS30'][v]['accuracy_delta_pp'],formal['budget_interaction_pp'][v]]
  lines.append('|'+v+'|'+'|'.join(f'{x:.6f}' for x in vals)+'|')
 vals=[formal['summary'][a]['urban_mean_accuracy']*100 for a in ('NOAS30','AS30','NOAS50','AS50')]+[formal['comparisons']['AS50__NOAS50']['urban_mean_accuracy_delta_pp'],formal['comparisons']['AS50__AS30']['urban_mean_accuracy_delta_pp'],formal['budget_interaction_pp']['urban_mean']]
 lines.append('|urban三场景均值|'+'|'.join(f'{x:.6f}' for x in vals)+'|')
 q=formal['stage']['50'];lines+=['',f"正式E50 clean代价{q['clean_delta_pp']:+.6f}pp，净修复{q['clean_net_repairs']}包；0.1pp整数判定net×1000≥−N：{'PASS' if q['clean_absolute_0p1pp_guard_met'] else 'FAIL'}。low/mid urban相对同预算NoAs分别{q['lowurban_gain_pp']:+.6f}/{q['midurban_gain_pp']:+.6f}pp。",'', '预算交互=(As50−NoAs50)−(As30−NoAs30)。NOAS50−NOAS30、AS50−AS30和同预算As−NoAs均保留，不混为机制效应。','', '|比较|视图|修复|新增错误|净修复|','|---|---|---:|---:|---:|']
 for a,b in PAIRS:
  for v in ('clean','practical_mid_urban','practical_low_urban'):
   z=formal['paired'][a+'__'+b][v];lines.append(f"|{a}−{b}|{v}|{z['repairs']}|{z['new_errors']}|{z['net_repairs']}|")
 lines+=['', '|模型|视图|macro-F1 %|最差RX %|','|---|---|---:|---:|']
 for a in ('NOAS30','AS30','NOAS50','AS50'):
  for v in VIEWS:
   z=formal['summary'][a]['views'][v];lines.append(f"|{a}|{v}|{100*z['macro_f1']:.6f}|{100*z['worst_RX_accuracy']:.6f}|")
 lines+=['', '源完整V27000指标来自原生冻结读出，未保存逐包fullV预测，不宣称逐包独立重算。固定源V1800两draw的raw/三practical logits独立核对；无源highurban，不用mid/low冒称urban三场景均值。source面板fc2.45GHz、训练/正式2.462GHz既有差异保留披露。','',f"源As50−NoAs50 fullV clean：{source['decision']['fullV_As50_minus_NoAs50_clean_pp']:+.6f}pp；源面板mid/low预算交互：{source['decision']['mid_low_mean_budget_interaction_pp']:+.6f}pp。source结果仅描述，不增加晋级门槛、不跳过正式收尾。",'', '全4440新增step与20epoch CSV/JSONL、6L RDrop尺度、原U/teacher、物理序列、actual student/teacher counters均完整读取；旧30轮使用既有完整recount，不重复扫描旧step。source与formal技术缺项不得冒充完整。','', '|行|student train NN|student eval NN|teacher eval NN|native NN|训练段wall s|','|---|---:|---:|---:|---:|---:|']
 for a in ('NOAS30','AS30','NOAS50','AS50'):
  c=training['rows'][a]['cost'];lines.append(f"|{a}|{c['student_train_NN']}|{c['student_eval_NN']}|{c['teacher_eval_NN']}|{c['native_training_and_validation_NN']}|{c['wall_seconds']:.3f}|")
 lines+=['',f"新增两行40模型epoch/8880更新，训练段+source+formal共{costs['new_segment_and_readouts_NN']}NN，另共享smoke{costs['shared_smoke_NN']}NN；父两行60模型epoch/13320更新及其原读出{costs['parent_training_and_readouts_NN']}NN单列。teacher_train_NN=0；独立分析0NN/0CP/0IQ。wall是各模型实际段时间，不能从并行差异推断加速。",'',f"正式独立重算{formal['independent_confusion_matrices']}个CM、{formal['partition_reconciliations']}次RX/day/TX分区合并，与固定argmax和独立scorer逐项一致。全部4logical rows×7view预测固定及独立评分完成后才连接truth；测试IQ/SNR/信道/困难样本均未改变。",'', '单模型seed无种子标准差，不证明重复性；小于0.1pp的正变化只记微小观察。阶段clean约束通过也不等于最终可重复双提升。测试不回流调参、候选或选择性重跑；本报告不据正式分数提出后续参数建议。']
 return '\n'.join(lines)+'\n'

def metadata_paths(project,stage):
 paths=[]
 for arm,root in rows(project):
  paths += [root/'source/completion.json',root/'source/source_frozen.json',root/'source_qualification/complete.json']
  if stage=='FINAL':paths += [root/'prediction/complete.json',root/'scoring/complete.json',root/'complete.json']
 paths.append(Path(project)/'runs'/RUN/'source_matrix_decision.json')
 if stage=='FINAL':paths.append(Path(project)/'runs'/RUN/'automatic_summary.json')
 return paths

def ready(paths):
 try:
  for p in paths:read(p)
 except (FileNotFoundError,json.JSONDecodeError):return False
 return True

def owned_handles(project,release):
 base=Path(project)/'runs'/RUN;ids=[];unknown=False
 for name in ('dispatcher.json','launch.json'):
  try:r=read(base/name);ids += [r['pid']] if name=='dispatcher.json' else [v['pid'] for v in r['rows']]
  except FileNotFoundError:pass
  except json.JSONDecodeError:unknown=True
 live=[]
 for pid in sorted(set(ids)):
  p=Path('/proc')/str(pid)
  try:
   argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_practical_budget50/runner.py' in argv:live.append(pid)
  except FileNotFoundError:pass
  except OSError:unknown=True
 return live,unknown

def wait_until(project,release,stage,out,readiness=ready,probe=owned_handles,sleep=time.sleep):
 while not readiness(metadata_paths(project,stage)):
  root=Path(project)/'runs'/RUN
  if (root/'failure.json').is_file():raise RuntimeError('Owned run technical failure preserved')
  for arm,r in rows(project):
   if arm.endswith('50') and (r/'source_qualification_failure.json').is_file():raise AwaitingArtifacts('Actual source readout failed;preserve outputs,formal closure remains owned by training runner')
  live,unknown=probe(project,release)
  if not live and not unknown:raise AwaitingArtifacts('Owned handles absent before '+stage+' artifacts complete')
  write(out/'state.json',dict(status='WAIT_'+stage,live_owned_PIDs=live,observation_unknown=unknown,truth_opened=False,NN=0));sleep(15)

def validate_output(project,output):
 out=Path(output)
 if out.parent!=Path(project)/'runs'/RUN or re.fullmatch(r'analysis_auto_r[0-9]{2}',out.name) is None:raise ValueError('Only own immutable analysis output root')
 if out.exists():raise FileExistsError('Reconcile existing analysis output;no relaunch')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
 if a.release.name!=TRAINING_RELEASE:raise ValueError('Only registered actual training release')
 out=validate_output(a.project,a.output);out.mkdir();started=time.perf_counter();write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,analysis_code_commit=a.code_commit,NN=0,truth_opened=False))
 try:
  wait_until(a.project,a.release,'SOURCE',out);training=training_analysis(a.project);write(out/'training_logs.json',training);source=source_analysis(a.project);write(out/'source_summary.json',{k:v for k,v in source.items() if k!='metrics'});write(out/'source_confusions.json',source['metrics']);write_csv(out/'source_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in source['metrics']]);write(out/'source_complete.json',dict(status='SOURCE_ANALYZED',NN=0,target_scores_read=False));print('SOURCE_ANALYSIS_COMPLETE',flush=True)
  wait_until(a.project,a.release,'FINAL',out);formal,_=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training,formal);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=1,new_model_rows=2,reused_model_rows=2,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,pipeline_wall_including_wait_seconds=time.perf_counter()-started,repeatable_clean_positive_proven=False,goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as e:
  write(out/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
