"""All-three-seed, zero-model analysis of the fixed original practical E50 recipe."""
import argparse,json,os,re,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_practical_budget50_analysis_20261018 as old
RUN='20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01'
TRAINING_RELEASE='cvs_FcrossRX_As_budget_E50_s0203_20261018_r01'
E30_REPLICATION='20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02'
SEEDS=(2026092701,2026092702,2026092703);NEW_SEEDS=SEEDS[1:]
CONDITIONS=('NOAS30','AS30','NOAS50','AS50');VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;COUNT=168000
PAIR_NAMES=('AS50_minus_NOAS50','AS30_minus_NOAS30','AS50_minus_AS30','NOAS50_minus_NOAS30')
read=old.read;write=old.write;write_csv=old.write_csv;metric=old.metric;paired=old.paired;AwaitingArtifacts=old.AwaitingArtifacts;compare_values=old.compare_values;stage_clean_pass=old.stable.stage_clean_pass

def label(condition,seed):return condition+'-s'+str(seed)
def split_label(value):
 condition,seed=value.rsplit('-s',1);return condition,int(seed)
def rows(project):
 root=Path(project)/'runs';result=[]
 for seed in SEEDS:
  for condition in CONDITIONS:
   budget=int(condition[-2:]);run=(old.REFERENCE_RUN if seed==SEEDS[0] else E30_REPLICATION) if budget==30 else old.RUN if seed==SEEDS[0] else RUN
   arm='B00' if condition.startswith('NOAS') else 'B01';result.append((label(condition,seed),root/run/(arm+'-s'+str(seed))))
 return result

def pairs_for(seed):
 return [(name,label(a,seed),label(b,seed)) for name,(a,b) in zip(PAIR_NAMES,old.PAIRS)]

def stats3(values):
 if len(values)!=3:raise ValueError('All three fixed seeds required;partial is not complete')
 a=np.asarray(values,float)
 if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Finite three-seed scalar values required')
 return dict(values=a.tolist(),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),model_seed_count=3)

def ensure_metadata(project,stage):
 result={}
 for arm,root in rows(project):
  condition,seed=split_label(arm);budget=int(condition[-2:]);names=['source/completion.json','source/source_frozen.json','source_qualification/complete.json']
  if stage=='final':names+=['prediction/complete.json','scoring/complete.json','complete.json']
  found={}
  for name in names:
   try:found[name]=read(root/name)
   except (FileNotFoundError,json.JSONDecodeError) as exc:raise AwaitingArtifacts(arm+'/'+name) from exc
  c=found['source/completion.json'];f=found['source/source_frozen.json'];q=found['source_qualification/complete.json']
  if (c['epoch'],c['optimizer_steps'])!=((20,4440) if budget==50 else (10,2220)) or c['target_access'] or c.get('head_fit_updates',0) or c['config']['model_seed']!=seed or c['config']['row_id']!=root.name or c['config']['run_id']!=root.parent.name:raise ValueError('Fixed logical row/seed/native budget differs')
  if budget==50 and (c['cumulative_epochs']!=50 or c['cumulative_optimizer_steps']!=11100):raise ValueError('CumulativeE50 scope differs')
  if f['selection']!='final_only' or f['target_access'] or f['config']!=c['config'] or q['target_access'] or q['NN']!=12600 or q.get('head_fit_updates',0) or q.get('training_updates',0):raise ValueError('Frozen source/defaulthead boundary differs')
  if stage=='final':
   p=found['prediction/complete.json'];s=found['scoring/complete.json']
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed until all12 logical rows complete all7 frozen predictions and independent scores')
  result[arm]=found
 return result

def source_analysis(project):
 started=time.perf_counter();metadata=ensure_metadata(project,'source');summary={};data={};metrics=[];producer_rows={}
 for arm,root in rows(project):
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in f.files}
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Fixed finite source panel absent')
  if data and any(not np.array_equal(x[k],data[label('NOAS30',SEEDS[0])]['x'][k]) for k in ('ids','TX','RX','day')):raise ValueError('All source panel physical roles must match')
  q=metadata[arm]['source_qualification/complete.json'];accuracy=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if not np.allclose(accuracy,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(-1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Independent source logits versus original readout differ')
  native=metadata[arm]['source/completion.json']['final_source_metrics']
  if native['source_val_count']!=27000:raise ValueError('FullV native role count differs')
  pred=dict(clean=x['clean'].argmax(-1))
  for i,v in enumerate(SOURCE_SCENES):pred[v]=x['satellite'][:,i].argmax(-1).reshape(-1)
  panel={};data[arm]=dict(x=x,pred=pred)
  for v,p in pred.items():
   y=x['TX'] if v=='clean' else np.tile(x['TX'],2);rx=x['RX'] if v=='clean' else np.tile(x['RX'],2);m=metric(y,p);panel[v]=dict(accuracy=m['accuracy'],macro_F1=m['macro_f1'],worst_RX=min(float((p[rx==r]==y[rx==r]).mean()) for r in np.unique(rx)))
  producer_rows[arm]=dict(full_V_clean=native['source_val_accuracy'],full_V_RX_accuracy=native['source_val_rx_accuracy'],panel=panel,mid_low_mean=float(np.mean([panel[v]['accuracy'] for v in SOURCE_SCENES[1:]])))
  summary[arm]=dict(**producer_rows[arm],mid_low_by_draw_accuracy=accuracy[:,1:].tolist())
  obs=[('clean',-1,x['clean'],x['TX'],x['RX'],x['day'])]
  for si,v in enumerate(SOURCE_SCENES):
   for draw in (-1,0,1):
    ds=(0,1) if draw==-1 else (draw,);obs.append((v,draw,np.concatenate([x['satellite'][d,si] for d in ds]),np.tile(x['TX'],len(ds)),np.tile(x['RX'],len(ds)),np.tile(x['day'],len(ds))))
  for scene,draw,z,y,rx,day in obs:
   p=z.argmax(1)
   for dim,vals in [('overall',['ALL']),('RX',sorted(set(rx))),('day',sorted(set(day))),('TX',range(6))]:
    for val in vals:
     mask=np.ones(len(y),bool) if dim=='overall' else rx==val if dim=='RX' else day==val if dim=='day' else y==val
     if mask.any():metrics.append(dict(arm=arm,scene=scene,draw=draw,dimension=dim,stratum=str(val),**metric(y[mask],p[mask])))
 per_seed={};pairs={};sourceviews=('clean',*SOURCE_SCENES)
 for seed in SEEDS:
  producer_pairs={};cohort=data[label('NOAS30',seed)]['pred']['clean']==data[label('NOAS30',seed)]['x']['TX']
  for name,a,b in pairs_for(seed):
   key=str(seed)+'__'+name;pairs[key]={};producer_pairs[name]={}
   for view in sourceviews:
    y=data[a]['x']['TX'] if view=='clean' else np.tile(data[a]['x']['TX'],2);co=cohort if view=='clean' else np.tile(cohort,2);d=paired(y,data[b]['pred'][view],data[a]['pred'][view],co);pairs[key][view]=d
    producer_pairs[name][view]=dict(repaired=d['repairs'],new_errors=d['new_errors'],net=d['net_repairs'],accuracy_pp=d['accuracy_delta_pp'],count=d['count'],macro_F1_pp=100*(summary[a]['panel'][view]['macro_F1']-summary[b]['panel'][view]['macro_F1']),worst_RX_pp=100*(summary[a]['panel'][view]['worst_RX']-summary[b]['panel'][view]['worst_RX']))
   producer_pairs[name]['fullV_clean_pp']=100*(summary[a]['full_V_clean']-summary[b]['full_V_clean']);producer_pairs[name]['mid_low_mean_pp']=100*(summary[a]['mid_low_mean']-summary[b]['mid_low_mean'])
  as50=producer_pairs['AS50_minus_NOAS50'];as30=producer_pairs['AS30_minus_NOAS30']
  per_seed[str(seed)]=dict(paired=producer_pairs,source_clean_reference_within_0p1pp=as50['fullV_clean_pp']>=-.1-1e-12,panel_accuracy_budget_interaction_pp={v:as50[v]['accuracy_pp']-as30[v]['accuracy_pp'] for v in sourceviews},fullV_clean_budget_interaction_pp=as50['fullV_clean_pp']-as30['fullV_clean_pp'],mid_low_mean_budget_interaction_pp=as50['mid_low_mean_pp']-as30['mid_low_mean_pp'])
 aggregate={name:{v:stats3([per_seed[str(s)]['paired'][name][v]['accuracy_pp'] for s in SEEDS]) for v in sourceviews} for name in PAIR_NAMES}
 saved=read(Path(project)/'runs'/RUN/'source_matrix_decision.json')
 if saved['status']!='SOURCE_BUDGET_REPLICATION_READOUT_COMPLETE' or saved['target_scores_read']:raise ValueError('Source independent completion absent')
 for k,v in dict(rows=producer_rows,per_seed=per_seed,paired_accuracy_pp_three_seed=aggregate).items():compare_values(saved[k],v,'source.'+k)
 fullV_aggregate={name:stats3([per_seed[str(s)]['paired'][name]['fullV_clean_pp'] for s in SEEDS]) for name in PAIR_NAMES}
 midlow_aggregate={name:stats3([per_seed[str(s)]['paired'][name]['mid_low_mean_pp'] for s in SEEDS]) for name in PAIR_NAMES}
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,paired=pairs,per_seed=per_seed,paired_accuracy_pp_three_seed=aggregate,fullV_clean_pp_three_seed=fullV_aggregate,mid_low_mean_pp_three_seed=midlow_aggregate,source_unique_V=1800,channel_draws=2,model_seeds=list(SEEDS),source_high_urban_available=False,fullV_metrics_from_native_not_individual_prediction_recount=True,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,scientific_negative_skips_formal=False,wall_seconds=time.perf_counter()-started)

def parent_audit_cost(audit):
 c=audit['cost'];counters=c['source_counters'];keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')
 if set(counters)!=set(keys) or any(type(counters[k]) is not int or counters[k]<0 for k in keys):raise ValueError('Original parent audit requires actual four counters')
 total=sum(counters.values())
 if total!=c['native_training_and_validation_identity_NN'] or c['new_updates']!=6660 or len(audit['curve'])!=30 or audit['full_records']['actual_steps']!=6660:raise ValueError('Parent full30 audit scope differs')
 return dict(**counters,native_training_and_validation_NN=total,wall_seconds=c['source_wall_seconds'],local_epochs=30,local_updates=6660,L_NN=c['L_identity_NN'],frozen_source_readout_NN_already_in_old_source_identity_NN=c['frozen_source_readout_NN'])

def training_analysis(project):
 project=Path(project);base=project/'runs';prior=read(base/old.RUN/'analysis_auto_r01/training_logs.json');auditroot=base/E30_REPLICATION/'analysis_auto_r01/source';streams=read(auditroot/'training_stream_comparability.json');out={};private={};roots=dict(rows(project))
 for condition in CONDITIONS:
  value=prior['rows'][condition];out[label(condition,SEEDS[0])]=dict(cost=value['cost'],epoch_curves=value.get('epoch_curves',[]),reused=True,evidence='Completed seed01 E30/E50 full-log recount reused;no old step scan')
 for seed in NEW_SEEDS:
  for condition,arm in [('NOAS30','B00'),('AS30','B01')]:
   a=read(auditroot/(arm+'-s'+str(seed)+'_training.json'));out[label(condition,seed)]=dict(cost=parent_audit_cost(a),epoch_curves=a['curve'],reused=True,evidence='Existing full E30 source audit actual four counters;old steps not reopened')
  for condition in ('NOAS50','AS50'):
   key=label(condition,seed);out[key],private[key]=old.parse_new_logs(roots[key],condition);out[key]['reused']=False
  left,right=label('NOAS50',seed),label('AS50',seed);sequence=old.logs.compare_streams(roots[left]/'source/training_sequence.jsonl.gz',roots[right]/'source/training_sequence.jsonl.gz')
  if not sequence['all_equal'] or private[left]['ordinary_mse']!=private[right]['ordinary_mse']:raise ValueError('New within-seed physical/ordinary sequence differs')
  oldscope=streams[str(seed)]
  if not oldscope['all_identical'] or oldscope['ordinary_MSE_records_different']:raise ValueError('Referenced original parent stream audit does not match')
  a,b=private[left]['args'],private[right]['args'];diff=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
  if set(diff)-{'candidate_id','output_dir','baseline_ckpt'}:raise ValueError('New paired native parameters differ')
  out[right]['matched_NoAs_log_stream']=dict(sequence=sequence,ordinary_MSE_records=4440,ordinary_MSE_mismatch=0,different_actual_arg_keys=diff)
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=out,new_model_rows=4,new_model_epochs=80,new_updates=17760,new_local_steps=sum(out[label(c,s)]['counts']['steps'] for s in NEW_SEEDS for c in ('NOAS50','AS50')),old_seed01_E50_model_epochs=40,old_seed01_E50_updates=8880,parent_E30_model_epochs=180,parent_E30_updates=39960,model_seeds=list(SEEDS),IQ_reads=0,checkpoint_reads=0,NN=0,target_scores_read=False)

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
  if private and (not np.array_equal(ids,private[label('NOAS30',SEEDS[0])]['ids']) or not np.array_equal(y,private[label('NOAS30',SEEDS[0])]['y'])):raise ValueError('Same paired physical samples required')
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
     metrics.append(dict(arm=arm,row_id=root.name,model_seed=split_label(arm)[1],view=view,dimension=dim,stratum=str(value),**q));subtotal+=np.asarray(q['confusion'])
    if dim!='overall':
     expected=metric(y,pred[view])['confusion']
     if not np.array_equal(subtotal,expected):raise ValueError('Formal stratum partition differs')
     partitions+=1
   overall=next(q for q in metrics if q['view']==view and q['dimension']=='overall');views[view]=dict(overall,mean_true_margin_over30=float(margin.mean()),margin_quantiles_over30=np.quantile(margin,[0,.1,.5,.9,1]).tolist(),worst_RX_accuracy=min(q['accuracy'] for q in metrics if q['view']==view and q['dimension']=='receiver'))
  results[arm]=dict(views=views,urban_mean_accuracy=float(np.mean([views[v]['accuracy'] for v in VIEWS[4:]])),prediction_NN=metadata[arm]['prediction/complete.json']['actual_NN'],prediction_wall_seconds=metadata[arm]['prediction/complete.json']['wall_seconds'],reused=not (split_label(arm)[1] in NEW_SEEDS and split_label(arm)[0].endswith('50')));private[arm]=dict(ids=ids,y=y,rx=rx,day=day,pred=pred);all_metrics.extend(metrics)

 pairs={};comparisons={};per_seed={}
 for seed in SEEDS:
  cohort=private[label('NOAS30',seed)]['pred']['clean']==private[label('NOAS30',seed)]['y'];pc={};stages={}
  for name,new,control in pairs_for(seed):
   key=str(seed)+'__'+name;pairs[key]={};comparisons[key]={}
   for view in VIEWS:
    d=paired(private[new]['y'],private[control]['pred'][view],private[new]['pred'][view],cohort);pairs[key][view]=d;comparisons[key][view]=dict(accuracy_delta_pp=d['accuracy_delta_pp'],macro_F1_delta_pp=100*(results[new]['views'][view]['macro_f1']-results[control]['views'][view]['macro_f1']),worst_RX_delta_pp=100*(results[new]['views'][view]['worst_RX_accuracy']-results[control]['views'][view]['worst_RX_accuracy']))
   comparisons[key]['urban_mean_accuracy_delta_pp']=float(np.mean([comparisons[key][v]['accuracy_delta_pp'] for v in VIEWS[4:]]));pc[name]={v:pairs[key][v]['accuracy_delta_pp'] for v in VIEWS};pc[name]['urban_mean']=comparisons[key]['urban_mean_accuracy_delta_pp']
  interaction={v:pc['AS50_minus_NOAS50'][v]-pc['AS30_minus_NOAS30'][v] for v in (*VIEWS,'urban_mean')}
  for budget in (30,50):
   key=str(seed)+'__AS'+str(budget)+'_minus_NOAS'+str(budget);c=pairs[key]['clean'];delta=c['accuracy_delta_pp'];stages[str(budget)]=dict(clean_delta_pp=delta,clean_net_repairs=c['net_repairs'],clean_absolute_0p1pp_guard_met=stage_clean_pass(c['net_repairs'],COUNT),clean_positive_observed=delta>0,positive_below_0p1pp_is_micro_observation=0<delta<.1,lowurban_gain_pp=pc['AS'+str(budget)+'_minus_NOAS'+str(budget)]['practical_low_urban'],midurban_gain_pp=pc['AS'+str(budget)+'_minus_NOAS'+str(budget)]['practical_mid_urban'])
  per_seed[str(seed)]=dict(paired_accuracy_pp=pc,budget_interaction_pp=interaction,stage=stages,stage_clean_0p1pp=stages['50']['clean_absolute_0p1pp_guard_met'])
 aggregate={name:{v:stats3([per_seed[str(s)]['paired_accuracy_pp'][name][v] for s in SEEDS]) for v in (*VIEWS,'urban_mean')} for name in PAIR_NAMES};interactions={v:stats3([per_seed[str(s)]['budget_interaction_pp'][v] for s in SEEDS]) for v in (*VIEWS,'urban_mean')}
 absolute={c:{v:{k:stats3([results[label(c,s)]['views'][v][k] for s in SEEDS]) for k in ('accuracy','macro_f1','worst_RX_accuracy')} for v in VIEWS} for c in CONDITIONS}
 macro_worst_pairs={name:{v:{k:stats3([comparisons[str(s)+'__'+name][v][k] for s in SEEDS]) for k in ('macro_F1_delta_pp','worst_RX_delta_pp')} for v in VIEWS} for name in PAIR_NAMES}
 declaration=declaration_from_pairs(pairs,COUNT)
 auto=read(Path(project)/'runs'/RUN/'automatic_summary.json')
 for arm in results:
  for view in VIEWS:
   q=results[arm]['views'][view];compare_values(auto['summary'][arm][view],dict(accuracy=q['accuracy'],macro_F1=q['macro_f1'],worst_RX=q['worst_RX_accuracy']))
  compare_values(auto['summary'][arm]['urban_mean']['accuracy'],results[arm]['urban_mean_accuracy'])
 for seed in SEEDS:
  for k in ('paired_accuracy_pp','budget_interaction_pp','stage_clean_0p1pp'):compare_values(auto['per_seed'][str(seed)][k],per_seed[str(seed)][k])
 compare_values(auto['paired_accuracy_pp_three_seed'],aggregate);compare_values(auto['budget_interaction_pp_three_seed'],interactions)
 absolute_urban={c:stats3([results[label(c,s)]['urban_mean_accuracy'] for s in SEEDS]) for c in CONDITIONS}
 value=dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,comparisons=comparisons,per_seed=per_seed,paired_accuracy_pp_three_seed=aggregate,budget_interaction_pp_three_seed=interactions,absolute_metrics_three_seed=absolute,absolute_urban_mean_accuracy_three_seed=absolute_urban,paired_F1_worstRX_pp_three_seed=macro_worst_pairs,declaration=declaration,suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=3,model_seeds=list(SEEDS),independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)
 return value,private

def declaration_from_pairs(pairs,n):
 checks={};nets=[]
 for seed in SEEDS:
  p=pairs[str(seed)+'__AS50_minus_NOAS50'];net=p['clean']['net_repairs'];nets.append(net)
  checks[str(seed)]=dict(clean_positive=net>0,midurban_positive=p['practical_mid_urban']['net_repairs']>0,lowurban_positive=p['practical_low_urban']['net_repairs']>0,stage_clean_within_0p1pp=stage_clean_pass(net,n))
 mean_guard=sum(nets)*1000>=-len(SEEDS)*n;positive_mean_at_least_0p1=sum(nets)*1000>=len(SEEDS)*n
 consistent=all(q['clean_positive'] and q['midurban_positive'] and q['lowurban_positive'] for q in checks.values()) and positive_mean_at_least_0p1
 return dict(per_seed=checks,mean_clean_within_0p1pp=mean_guard,all_seeds_clean_within_0p1pp=all(q['stage_clean_within_0p1pp'] for q in checks.values()),mean_clean_at_least_positive_0p1pp=positive_mean_at_least_0p1,fixed_three_seed_consistent_double_improvement=consistent,scope='Only these three fixed seeds,all paired samples;not seed-independent universal proof',no_selective_seed_or_rerun=True)

def cost_summary(project,training):
 meta=ensure_metadata(project,'final');result={}
 for arm,_ in rows(project):
  c=training['rows'][arm]['cost'];native=c['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json'];condition,seed=split_label(arm)
  group='new_four_E50_segments' if seed in NEW_SEEDS and condition.endswith('50') else 'old_seed01_E50_segments' if seed==SEEDS[0] and condition.endswith('50') else 'six_E30_parents'
  result[arm]=dict(group=group,native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],total_for_this_segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=p['wall_seconds'])
 groups={g:dict(rows=[a for a,r in result.items() if r['group']==g],native_NN=sum(r['native_NN'] for r in result.values() if r['group']==g),segment_and_readouts_NN=sum(r['total_for_this_segment_and_own_readouts_NN'] for r in result.values() if r['group']==g)) for g in ('new_four_E50_segments','old_seed01_E50_segments','six_E30_parents')}
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query']:raise ValueError('Actual shared smoke must be complete/noquery')
 shared=smoke['NN']+real['NN']
 return dict(rows=result,groups=groups,new_shared_smoke_NN=shared,new_total_NN=groups['new_four_E50_segments']['segment_and_readouts_NN']+shared,new_model_epochs=80,new_updates=17760,old_seed01_E50_model_epochs=40,old_seed01_E50_updates=8880,all_six_E30_parent_model_epochs=180,all_six_E30_parent_updates=39960,analysis_NN=0,historical_shared_smoke_not_recharged=True,wall_scope='Actual model-seconds;not parallel elapsed or speed evidence')

def render_report(source,formal,training,costs):
 d=formal['declaration'];lines=['# 固定完整 practical E50：三种子匹配复制','',RUN+'。全部固定种子2026092701/02/03均纳入；seed01原结果复用，新02/03各NoAs/As ownFINAL30+20，未依据正式分数选种子或改配方。','', '模型F/SIG_TIME/统一Sinc/joint160/默认cos头联合训练、L6300/U56700隐藏/V27000、全部源RX/day和固定正式IQ/信道保持。原As KL/dropout完整保留；这是同配方预算复制，不是新增结构或损失。','', '|seed|视图|NoAs30 %|As30 %|NoAs50 %|As50 %|As50−NoAs50 pp|预算交互 pp|','|---|---|---:|---:|---:|---:|---:|---:|']
 for seed in SEEDS:
  for v in (*VIEWS,'urban_mean'):
   vals=[100*(formal['summary'][label(c,seed)]['urban_mean_accuracy'] if v=='urban_mean' else formal['summary'][label(c,seed)]['views'][v]['accuracy']) for c in CONDITIONS];p=formal['per_seed'][str(seed)];vals += [p['paired_accuracy_pp']['AS50_minus_NOAS50'][v],p['budget_interaction_pp'][v]];lines.append('|'+str(seed)+'|'+v+'|'+'|'.join(f'{x:.6f}' for x in vals)+'|')
 lines+=['','以下均为同seed配对变化的均值±样本SD（ddof=1），不是挑选最好seed。预算交互=(As50−NoAs50)−(As30−NoAs30)。','', '|视图|As50−NoAs50 pp|As30−NoAs30 pp|As50−As30 pp|NoAs50−NoAs30 pp|预算交互 pp|','|---|---:|---:|---:|---:|---:|']
 for v in (*VIEWS,'urban_mean'):
  a=[formal['paired_accuracy_pp_three_seed'][n][v] for n in PAIR_NAMES]+[formal['budget_interaction_pp_three_seed'][v]];lines.append('|'+v+'|'+'|'.join(f"{q['mean']:+.6f} ± {q['sample_SD']:.6f}" for q in a)+'|')
 lines+=['', '|seed|E50 clean净修复|clean阶段≤0.1pp|clean正收益|mid正收益|low正收益|','|---|---:|---|---|---|---|']
 for seed in SEEDS:
  q=d['per_seed'][str(seed)];net=formal['paired'][str(seed)+'__AS50_minus_NOAS50']['clean']['net_repairs'];lines.append(f"|{seed}|{net}|{q['stage_clean_within_0p1pp']}|{q['clean_positive']}|{q['midurban_positive']}|{q['lowurban_positive']}|")
 lines+=['',f"clean配对均值阶段约束：{d['mean_clean_within_0p1pp']}；全部seed阶段约束：{d['all_seeds_clean_within_0p1pp']}。整数判定每seed net×1000≥−N，均值用Σnet×1000≥−3N，0.1是绝对百分点。",'',f"预固定陈述条件（三seed clean/low/mid均为正且clean配对mean≥+0.1pp）：{d['fixed_three_seed_consistent_double_improvement']}。此结论仅限这三个固定seed。小于0.1pp的正变化标为微小观察，阶段基本不降不等于最终稳定双提升。",'', '|seed|As50视图|macro-F1 %|最差RX %|','|---|---|---:|---:|']
 for seed in SEEDS:
  for v in VIEWS:
   q=formal['summary'][label('AS50',seed)]['views'][v];lines.append(f"|{seed}|{v}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['', '全部12逻辑行、七视图macro-F1/最差RX、绝对指标及配对均值/SD、RX/day/TX分层见formal_summary.json与formal_metrics.csv；以下保留四类关键修复和新增错误。','', '|seed|比较|view|修复|新增错误|净修复|','|---|---|---|---:|---:|---:|']
 for seed in SEEDS:
  for name in PAIR_NAMES:
   for v in ('clean','practical_mid_urban','practical_low_urban'):
    q=formal['paired'][str(seed)+'__'+name][v];lines.append(f"|{seed}|{name}|{v}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 lines+=['', '|seed|fullV As50−NoAs50 clean pp|V1800 clean pp|mid/low源平均 pp|','|---|---:|---:|---:|']
 for seed in SEEDS:
  q=source['per_seed'][str(seed)]['paired']['AS50_minus_NOAS50'];lines.append(f"|{seed}|{q['fullV_clean_pp']:+.6f}|{q['clean']['accuracy_pp']:+.6f}|{q['mid_low_mean_pp']:+.6f}|")
 lines+=['', '完整V27000 clean为原生冻结指标，未保存全V逐包预测，不声称独立逐包重算；固定V1800两draw的raw与三practical logits独立重算，保留与fullV可能相反的方向。源没有high urban，mid/low源均值不冒称urban三场景均值；面板fc2.45GHz与训练/正式2.462GHz原差异保留。source科学负值不阻断正式收尾。','', '|组|模型epoch|更新|本组训练段+自身源/正式读出NN|','|---|---:|---:|---:|']
 for group,epochs,updates in [('new_four_E50_segments',80,17760),('old_seed01_E50_segments',40,8880),('six_E30_parents',180,39960)]:lines.append(f"|{group}|{epochs}|{updates}|{costs['groups'][group]['segment_and_readouts_NN']}|")
 lines+=['',f"本run共享smoke实际{costs['new_shared_smoke_NN']}NN；本run总计{costs['new_total_NN']}NN。旧01E50与六个E30父历史单列，旧读出不计入本run新增成本。新增4×4440步骤完整解析；旧30日志直接复用既有完整audit，不重新扫描。学生train/eval、原ownEMA教师、每行wall及历史成本见costs.json。分析0模型/0CP/0IQ/0更新。",'',f"全部12逻辑行×7view预测固定及独立评分完成后才连接truth；独立核对{formal['independent_confusion_matrices']}个CM、{formal['partition_reconciliations']}次分区合并，与原scorer/argmax/F1一致。正式IQ、测试SNR/信道和困难样本未改；结果不回流选模、调参、挑seed或选择性重跑。"]
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
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_practical_budget50_replication/runner.py' in argv:live.append(pid)
  except FileNotFoundError:pass
  except OSError:unknown=True
 return live,unknown

def wait_until(project,release,stage,out,readiness=ready,probe=owned_handles,sleep=time.sleep):
 while not readiness(metadata_paths(project,stage)):
  root=Path(project)/'runs'/RUN
  if (root/'failure.json').is_file():raise RuntimeError('Owned run technical failure preserved')
  for arm,r in rows(project):
   if split_label(arm)[0].endswith('50') and (r/'source_qualification_failure.json').is_file():raise AwaitingArtifacts('Actual source readout failed;preserve outputs,formal closure remains owned by training runner')
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
  wait_until(a.project,a.release,'FINAL',out);formal,_=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=3,new_model_rows=4,reused_model_rows=8,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,pipeline_wall_including_wait_seconds=time.perf_counter()-started,fixed_three_seed_consistent_double_improvement=formal['declaration']['fixed_three_seed_consistent_double_improvement'],goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as e:
  write(out/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
