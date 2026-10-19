"""Fixed source-only raw-IQ decision, followed by reused-control target analysis."""
import math
import time
from . import design as d

COUNT=168000
SOURCE_N=1800
FULL_V=27000


def paths():
 return {r['arm']:d.BASE/r['row_id'] for r in d.rows()}


def old_paths():
 return {a:d.OLD_CONTROL_BASE/(a+'-s'+str(d.SEED)) for a in d.OLD_CONTROL_ARMS}


def source_counts(v,done,run_id,row_id):
 """Metadata/count arithmetic only; no target score or truth access."""
 config=done['config'];metrics=done['final_source_metrics']
 if (v['status']!='SOURCE_READOUT_COMPLETE' or v['NN']!=12600 or v['target_access']
     or v['formal_scores_read'] or done['target_access'] or done['head_fit_updates']
     or done['epoch']!=30 or done['segment_epoch']!=10 or done['parent_epoch']!=20
     or done['optimizer_steps']!=2220 or done['cumulative_optimizer_steps']!=6660
     or metrics['source_val_count']!=FULL_V or v['config']!=config
     or config['run_id']!=run_id or config['row_id']!=row_id):
  raise ValueError('Matching frozen FINAL30 source readouts/completion required')
 rates=v['source_accuracies']
 if len(rates)!=2 or any(len(x)!=3 or any(not isinstance(n,(int,float)) or not math.isfinite(n) or not 0<=n<=1 for n in x) for x in rates):
  raise ValueError('Two fixed source draws and three scenes required')
 counts=v.get('source_correct',[[round(rate*SOURCE_N) for rate in draw] for draw in rates])
 if len(counts)!=2 or any(len(x)!=3 for x in counts):
  raise ValueError('Source correct counts differ')
 if any(type(counts[i][j]) is not int or not 0<=counts[i][j]<=SOURCE_N
     or not math.isclose(counts[i][j]/SOURCE_N,rates[i][j],abs_tol=1e-12,rel_tol=0.)
     for i in range(2) for j in range(3)):
  raise ValueError('Source rates/correct counts disagree')
 clean=metrics['source_val_accuracy']
 if not isinstance(clean,(int,float)) or not math.isfinite(clean) or not 0<=clean<=1:
  raise ValueError('Full source-V clean accuracy differs')
 clean_correct=round(clean*FULL_V)
 if not math.isclose(clean_correct/FULL_V,clean,abs_tol=1e-12,rel_tol=0.):
  raise ValueError('Full source-V clean correct count is not integral')
 return dict(midlow_correct=sum(counts[i][j] for i in range(2) for j in (1,2)),
  midlow_n=7200,source_correct=counts,fullVclean_correct=clean_correct,
  fullVclean_n=FULL_V,fullVclean_accuracy=clean,readout=v)


def scientific_decision(new,old):
 """The preregistered three conditions; all counts are source-only."""
 local_delta=new['S3']['midlow_correct']-new['S2']['midlow_correct']
 architecture_delta=new['S3']['midlow_correct']-old['S3']['midlow_correct']
 noas=dict(new_NOAS_ADAPTER=new['NOAS_ADAPTER']['fullVclean_correct'],
  old_NOAS_FT=old['NOAS_FT']['fullVclean_correct'],
  old_NOAS_ADAPTER=old['NOAS_ADAPTER']['fullVclean_correct'])
 clean_delta=new['S3']['fullVclean_correct']-max(noas.values())
 criteria=dict(midlow_S3_minus_S2=local_delta>=36,
  midlow_S3_minus_old_local_S3=architecture_delta>=36,
  fullVclean_S3_minus_max_all_matched_NoAs=clean_delta>=-54)
 return dict(source_qualified=all(criteria.values()),criteria=criteria,
  midlow_increment_pp=100*local_delta/7200,
  midlow_vs_old_local_S3_pp=100*architecture_delta/7200,
  fullVclean_net_pp=100*clean_delta/FULL_V,
  correct_count_differences=dict(midlow_S3_minus_S2=local_delta,
   midlow_S3_minus_old_local_S3=architecture_delta,fullVclean_vs_best_NoAs=clean_delta),
  noas_fullVclean_correct=noas,
  thresholds=dict(midlow_S3_minus_S2_min_pp=.5,midlow_S3_minus_old_local_S3_min_pp=.5,
   fullVclean_S3_minus_max_all_matched_NoAs_min_pp=-.2,exact_correct_count_min=[36,36,-54]))


def source_decision():
 output=d.BASE/'source_matrix_decision.json'
 if output.exists():raise FileExistsError('Reuse the fixed source decision; do not overwrite')
 new={};old={}
 for arm,p in paths().items():
  row=arm+'-s'+str(d.SEED)
  new[arm]=source_counts(d.read(p/'source_qualification/complete.json'),
   d.read(p/'source/completion.json'),d.RUN,row)
 for arm in ('NOAS_FT','NOAS_ADAPTER','S3'):
  p=old_paths()[arm];row=arm+'-s'+str(d.SEED)
  old[arm]=source_counts(d.read(p/'source_qualification/complete.json'),
   d.read(p/'source/completion.json'),d.OLD_CONTROL_RUN,row)
 decision=scientific_decision(new,old)
 result=dict(status='SOURCE_DECISION_FIXED',run_id=d.RUN,timestamp=time.time(),
  primary='S3',new_source=new,old_source_controls=old,old_control_run_id=d.OLD_CONTROL_RUN,
  formal_scores_read=False,automatic_extension=False,goal_complete=False,
  source_panel='same fixed V1800, two full-physics draws, mid+lowUrban pooled denominator7200',
  clean_panel='full source V27000 from FINAL30 completion; not V1800 clean readout',
  decision=('SOURCE_ELIGIBLE_S3; matched continuation must be fixed before any extension'
   if decision['source_qualified'] else
   'CLOSE_FIXED_RAW_IQ16_GN8_DUALTEACHER_RECIPE; no coefficient/temperature/seed rescue'),**decision)
 d.write(output,result)
 return result


def _target_row(p,arm,source,new):
 import numpy as np
 pred=d.read(p/'prediction/complete.json');score=d.read(p/'scoring/complete.json')
 if (pred['status']!='PREDICTIONS_COMPLETE' or pred['count']!=COUNT or pred['views']!=list(d.VIEWS)
     or pred['truth_read'] or pred['query_fit'] or not pred['all_parameters_frozen'] or pred['head_fit_updates']
     or score['status']!='SCORED_COMPLETE' or score['views']!=list(d.VIEWS) or not score['truth_last']):
  raise ValueError('Complete fixed all7 predictions and independent scores required')
 if new and source['timestamp']>=(p/'scoring/complete.json').stat().st_mtime:
  raise ValueError('New row scoring must follow the fixed source decision')
 data=d.read(p/'scoring/scores.json')['results']
 overall_rows=[r for r in data if r['dimension']=='overall']
 overall={r['view']:r for r in overall_rows}
 if len(overall_rows)!=7 or set(overall)!=set(d.VIEWS):
  raise ValueError('Seven distinct scored views required')
 actual=d.read(p/'source/training_result.json')
 expected_run=d.RUN if new else d.OLD_CONTROL_RUN
 if (pred['config']['run_id']!=expected_run or pred['config']['row_id']!=arm+'-s'+str(d.SEED)
     or actual['config']!=pred['config']):raise ValueError('Own-run control prediction/training identity differs')
 costs={k:actual.get(k) for k in ('L_packets','student_train_packets','student_eval_packets',
  'student_counter_start','U_packets','own_EMA_train_packets','own_EMA_eval_packets',
  'own_EMA_successful_updates','teacher_packets','teacher_execution','fresh_As_packets',
  'student_As_packets','teacher_only_As_packets','ordinary_packets','ordinary_used_packets',
  'ordinary_shadow_packets','channel','joint160_actual','trainable_parameters','total_parameters',
  'Sinc_updates','head_frozen','F_frozen','optimizer_steps','cumulative_optimizer_steps')}
 summary=dict(overall=overall,worst_RX={v:min(r['accuracy'] for r in data if r['view']==v and r['dimension']=='receiver') for v in d.VIEWS},
  worst_RX_macroF1={v:min(r['macro_f1'] for r in data if r['view']==v and r['dimension']=='receiver') for v in d.VIEWS},
  costs=costs,new_epochs=0 if not new else 10,original_segment_epochs=10,cumulative_epochs=30,
  original_run_id=d.RUN if new else d.OLD_CONTROL_RUN,reused_complete_control=not new,
  new_NN_for_control_reuse=0 if not new else None,head_frozen=actual.get('head_frozen'),
  raw_IQ_adapter=actual.get('raw_IQ_adapter'),
  online_training=actual.get('online_training'),online_training_epochs=actual.get('online_training_epochs'),
  online_readout_semantics='Changing weights/train Dropout; never a frozen L-fit measurement')
 with np.load(p/'prediction/predictions.npz',allow_pickle=False) as f:
  ids=f['ids'].copy();decisions={v:f[v].copy() for v in d.VIEWS}
 return summary,ids,decisions


def _paired(after,before,y):
 import numpy as np
 result={}
 for v in d.VIEWS:
  a,b=after[v]==y,before[v]==y
  corrected=int((~b&a).sum());damaged=int((b&~a).sum());both=int((b&a).sum())
  result[v]=dict(n=len(y),corrected=corrected,new_errors=damaged,damaged=damaged,
   both_correct=both,both_wrong=int((~b&~a).sum()),net_pp=100*(corrected-damaged)/len(y),
   baseline_correct_retention=None if not both+damaged else both/(both+damaged),
   model_seed_count=1,sample_SD=None)
 return result


def _gaps(rows):
 return {a:dict(clean_gap_pp={v:100*(r['overall']['clean']['accuracy']-r['overall'][v]['accuracy']) for v in d.SCENES},
  urban_extra_gap_pp={level:100*(r['overall'][s]['accuracy']-r['overall'][u]['accuracy'])
   for level,s,u in zip(('high','mid','low'),d.SCENES[:3],d.SCENES[3:])}) for a,r in rows.items()}


def collect():
 import numpy as np
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 source=d.read(d.BASE/'source_matrix_decision.json')
 if source.get('status')!='SOURCE_DECISION_FIXED' or source.get('run_id')!=d.RUN or source['formal_scores_read'] is not False:
  raise ValueError('Source-only decision must already be fixed before target analysis')
 rows={};old={};decisions={};old_decisions={};ids=None;missing={}
 for arm,p in paths().items():
  summary,current,pred=_target_row(p,arm,source,True)
  if ids is not None and not np.array_equal(ids,current):raise ValueError('Same physical-ID new rows required')
  ids=current;rows[arm]=summary;decisions[arm]=pred
 # These are existing complete results, read only after the new source decision
 # and new all7 independent scores. Missing context never triggers new inference.
 for arm,p in old_paths().items():
  try:
   summary,current,pred=_target_row(p,arm,source,False)
   if not np.array_equal(ids,current):raise ValueError('Old control physical-ID order differs')
  except (FileNotFoundError,KeyError,ValueError) as error:
   missing[arm]=str(error);continue
  old[arm]=summary;old_decisions[arm]=pred
 y,_,_=truth_arrays(d.read(d.TRUTH),ids)
 pairs={};metric_deltas={}
 for model,base in (('S2','S1'),('S3','S2'),('S3','S1'),('S3','NOAS_ADAPTER')):
  pairs[model+'_minus_'+base]=_paired(decisions[model],decisions[base],y)
 for model in rows:
  for base in old:
   key=model+'_minus_OLD_'+base
   pairs[key]=_paired(decisions[model],old_decisions[base],y)
   metric_deltas[key]={v:dict(accuracy_pp=100*(rows[model]['overall'][v]['accuracy']-old[base]['overall'][v]['accuracy']),
    macroF1_pp=100*(rows[model]['overall'][v]['macro_f1']-old[base]['overall'][v]['macro_f1']),
    worst_RX_pp=100*(rows[model]['worst_RX'][v]-old[base]['worst_RX'][v]),
    worst_RX_macroF1_pp=100*(rows[model]['worst_RX_macroF1'][v]-old[base]['worst_RX_macroF1'][v])) for v in d.VIEWS}
 cohort=dict(availability='NOT_AVAILABLE',context_only_not_same_budget=True)
 parent=d.PARENT_SOURCE.parent/'prediction/predictions.npz'
 if parent.is_file():
  with np.load(parent,allow_pickle=False) as f:
   if np.array_equal(ids,f['ids']):
    mask=f['clean']==y;n=int(mask.sum())
    cohort=dict(availability='REUSED_FIXED_PARENT_PREDICTIONS',n=n,parent_epochs=20,context_only_not_same_budget=True,
     rows={a:{v:dict(retained_correct=int((pred[v][mask]==y[mask]).sum()),
      retention=None if not n else float((pred[v][mask]==y[mask]).mean())) for v in d.VIEWS}
      for a,pred in {**decisions,**{'OLD_'+a:p for a,p in old_decisions.items()}}.items()})
 result=dict(status='ANALYZED',run_id=d.RUN,source_decision=source,rows=rows,
  old_control_run_id=d.OLD_CONTROL_RUN,old_controls=old,old_control_context_missing=missing,
  old_controls_reused_NN=0,paired_errors=pairs,matched_E30_metric_deltas=metric_deltas,
  model_seed_count=1,sample_SD=None,crossseed_confirmed=False,goal_complete=False,
  gaps=dict(new=_gaps(rows),old_controls=_gaps(old)),
  suburban_gap_pp={a:{v:100*(r['overall']['clean']['accuracy']-r['overall'][v]['accuracy']) for v in d.SCENES[:3]} for a,r in rows.items()},
  fixed_parent_B0_clean_correct_cohort=cohort,
  cost_context=dict(parent_epochs=20,parent_updates=4440,incremental_epochs=10,incremental_updates=2220,
   cumulative_epochs=30,cumulative_updates=6660,new_raw_adapter_parameters=4256,old_local_adapter_parameters=4768,
   historical_teacher_training='Tc E20 and Ts E200 reused; no new teacher training',
   same_epoch_budget_is_not_equal_forward_budget=True),
  scientific_boundary='Source criteria were fixed before new scores; target metrics are reporting only, no tuning or selection feedback')
 d.write(d.BASE/'analysis.json',result)
 return result
