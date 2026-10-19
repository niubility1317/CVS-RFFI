"""Source-first fixed primary decision and same-budget six-row readout."""
import time
from . import design as d
def paths():return {r['arm']:d.BASE/r['row_id'] for r in d.rows()}
def source_decision():
 locations=paths();values={a:d.read(p/'source_qualification/complete.json') for a,p in locations.items()};clean={};correct={}
 for a,p in locations.items():
  v=values[a];done=d.read(p/'source/completion.json');m=done['final_source_metrics']
  if v['status']!='SOURCE_READOUT_COMPLETE' or v['NN']!=12600 or v['target_access'] or v['formal_scores_read'] or m['source_val_count']!=27000 or done['epoch']!=30:raise ValueError('Complete frozen FINAL30 source readouts required')
  rates=v['source_accuracies']
  if len(rates)!=2 or any(len(x)!=3 or any(not 0<=n<=1 for n in x) for x in rates):raise ValueError('Two fixed source draws/three scenes required')
  clean[a]=m['source_val_accuracy'];correct[a]=round(27000*clean[a])
 midlow={a:sum(round(values[a]['source_accuracies'][i][j]*1800) for i in range(2) for j in (1,2)) for a in locations}
 delta=midlow['S3']-midlow['S2'];net=correct['S3']-max(correct['NOAS_FT'],correct['NOAS_ADAPTER'])
 passed=delta>=36 and net>=-54
 result=dict(status='SOURCE_DECISION_FIXED',timestamp=time.time(),source_qualified=passed,primary='S3',midlow_increment_pp=100*delta/7200,fullVclean_net_pp=100*net/27000,
  midlow_correct=midlow,fullVclean_correct=correct,fullVclean=clean,readouts=values,formal_scores_read=False,automatic_extension=False,goal_complete=False,
  thresholds=dict(midlow_min_pp=.5,clean_min_pp=-.2,exact_correct_count_min=[36,-54]),
  decision='SOURCE_ELIGIBLE_S3;matched cumulative-budget transition must be fixed before any extension' if passed else 'CLOSE_FIXED_DUALTEACHER_TEMPORAL16_T4_RECIPE;no coefficient/temperature/seed rescue')
 d.write(d.BASE/'source_matrix_decision.json',result);return result
def collect():
 import numpy as np
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 source=d.read(d.BASE/'source_matrix_decision.json');locations=paths();rows={};decisions={};ids=None
 if source['formal_scores_read'] is not False:raise ValueError('Source decision must precede scores')
 for arm,p in locations.items():
  pred=d.read(p/'prediction/complete.json');score=d.read(p/'scoring/complete.json')
  if pred['status']!='PREDICTIONS_COMPLETE' or pred['count']!=168000 or pred['views']!=list(d.VIEWS) or pred['truth_read'] or score['status']!='SCORED_COMPLETE' or not score['truth_last'] or source['timestamp']>=(p/'scoring/complete.json').stat().st_mtime:raise ValueError('Complete all7 predictions/source-first independent score required')
  data=d.read(p/'scoring/scores.json')['results'];overall={r['view']:r for r in data if r['dimension']=='overall'}
  if set(overall)!=set(d.VIEWS):raise ValueError('Seven complete views required')
  actual=d.read(p/'source/training_result.json')
  rows[arm]=dict(overall=overall,worst_RX={v:min(r['accuracy'] for r in data if r['view']==v and r['dimension']=='receiver') for v in d.VIEWS},
    costs={k:actual.get(k) for k in ('L_packets','student_train_packets','teacher_packets','own_EMA_eval_packets','channel','teachers','joint160_actual')},new_epochs=10,cumulative_epochs=30,head_frozen=True)
  with np.load(p/'prediction/predictions.npz',allow_pickle=False) as f:
   current=f['ids'].copy()
   if ids is not None and not np.array_equal(ids,current):raise ValueError('Same physical-ID row pairs required')
   ids=current;decisions[arm]={v:f[v].copy() for v in d.VIEWS}
 y,_,_=truth_arrays(d.read(d.TRUTH),ids)
 parent=d.PARENT_SOURCE.parent/'prediction/predictions.npz'
 with np.load(parent,allow_pickle=False) as f:
  if not np.array_equal(ids,f['ids']):raise ValueError('Fixed inherited B0 parent cohort physical IDs differ')
  parent_clean=f['clean'].copy()
 cohort=parent_clean==y;n=int(cohort.sum());pairs={}
 for model,base in (('S1','S0'),('S2','S1'),('S3','S2'),('S3','S0'),('S3','NOAS_FT'),('S3','NOAS_ADAPTER')):
  values={}
  for v in d.VIEWS:
   before,after=decisions[base][v]==y,decisions[model][v]==y;damaged=int((before&~after).sum());both=int((before&after).sum());net=100*float(after.mean()-before.mean())
   values[v]=dict(n=len(y),corrected=int((~before&after).sum()),new_errors=damaged,damaged=damaged,both_correct=both,both_wrong=int((~before&~after).sum()),net_pp=net,mean_pp=net,
      baseline_correct_retention=None if not both+damaged else both/(both+damaged),positive_model_seeds=int(net>0),model_seed_count=1,sample_SD=None)
  pairs[model+'_minus_'+base]=values
 result=dict(status='ANALYZED',run_id=d.RUN,source_decision=source,rows=rows,paired_errors=pairs,model_seed_count=1,sample_SD=None,crossseed_confirmed=False,goal_complete=False,
   suburban_gap_pp={a:{v:100*(rows[a]['overall']['clean']['accuracy']-rows[a]['overall'][v]['accuracy']) for v in d.SCENES[:3]} for a in rows},
   fixed_parent_B0_clean_correct_cohort=dict(n=n,parent_epochs=20,context_only_not_same_budget=True,rows={a:{v:dict(retained_correct=int((decisions[a][v][cohort]==y[cohort]).sum()),retention=None if not n else float((decisions[a][v][cohort]==y[cohort]).mean())) for v in d.VIEWS} for a in rows}))
 d.write(d.BASE/'analysis.json',result);return result
