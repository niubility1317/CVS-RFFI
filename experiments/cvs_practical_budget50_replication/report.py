"""All fixed seeds and budgets; no candidate promotion or target feedback."""
from pathlib import Path
import numpy as np
from . import design as d
from experiments.cvs_practical_budget50.report import stats,delta
ALL_SEEDS=(2026092701,*d.SEEDS)

def label(arm,budget,seed):return ('AS' if arm=='B01' else 'NOAS')+str(budget)+'-s'+str(seed)
def logical_rows():
 out=[]
 for seed in ALL_SEEDS:
  old_run='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01' if seed==2026092701 else d.REFERENCE_RUN
  new_run=d.SEED01_RUN if seed==2026092701 else d.RUN
  for budget,run in [(30,old_run),(50,new_run)]:
   for arm in d.ARMS:out.append((label(arm,budget,seed),Path(d.PROJECT)/'runs'/run/(arm+'-s'+str(seed))))
 return out

def pairs_for(seed):
 a30,n30=label('B01',30,seed),label('B00',30,seed);a50,n50=label('B01',50,seed),label('B00',50,seed)
 return [('AS50_minus_NOAS50',a50,n50),('AS30_minus_NOAS30',a30,n30),('AS50_minus_AS30',a50,a30),('NOAS50_minus_NOAS30',n50,n30)]
def summary(values):
 a=np.asarray(values,dtype=float)
 return dict(values=list(map(float,a)),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),model_seed_count=len(a))

def source_decision():
 rows={};predictions={};base_ids=None;base_y=None;base_rx=None
 for arm,root in logical_rows():
  done=d.read(root/'source/completion.json');sc=d.read(root/'source/source_contract.json');frozen=d.read(root/'source/source_frozen.json');q=d.read(root/'source_qualification/complete.json')
  if sc['role_ids']!=d.read(d.SOURCE)['role_ids'] or done['target_access'] or frozen['target_access'] or q['target_access'] or done['final_source_metrics']['source_val_count']!=27000:raise ValueError('Exact frozen fullsource rows required')
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as a:
   ids=a['ids'];y=a['TX'];rx=a['RX'];clean=a['clean'];sat=a['satellite']
   if base_ids is None:base_ids=ids.copy();base_y=y.copy();base_rx=rx.copy()
   if len(ids)!=1800 or not np.array_equal(ids,base_ids) or not np.array_equal(y,base_y) or not np.array_equal(rx,base_rx) or clean.shape!=(1800,6) or sat.shape!=(2,3,1800,6) or not np.isfinite(clean).all() or not np.isfinite(sat).all():raise ValueError('Same fixed source panel required')
   preds=dict(clean=clean.argmax(-1))
   for i,s in enumerate(('practical_low_suburban','practical_mid_urban','practical_low_urban')):preds[s]=sat[:,i].argmax(-1).reshape(-1)
   panel={v:stats(y if v=='clean' else np.tile(y,2),p,rx if v=='clean' else np.tile(rx,2)) for v,p in preds.items()}
  rows[arm]=dict(full_V_clean=done['final_source_metrics']['source_val_accuracy'],full_V_RX_accuracy=done['final_source_metrics']['source_val_rx_accuracy'],panel=panel,mid_low_mean=float(np.mean([panel[s]['accuracy'] for s in ('practical_mid_urban','practical_low_urban')])))
  predictions[arm]=preds
 per_seed={}
 for seed in ALL_SEEDS:
  pairs={}
  for name,a,b in pairs_for(seed):
   pairs[name]={v:dict(delta(predictions[a][v],predictions[b][v],base_y if v=='clean' else np.tile(base_y,2)),macro_F1_pp=100*(rows[a]['panel'][v]['macro_F1']-rows[b]['panel'][v]['macro_F1']),worst_RX_pp=100*(rows[a]['panel'][v]['worst_RX']-rows[b]['panel'][v]['worst_RX'])) for v in predictions[a]}
   pairs[name]['fullV_clean_pp']=100*(rows[a]['full_V_clean']-rows[b]['full_V_clean'])
   pairs[name]['mid_low_mean_pp']=100*(rows[a]['mid_low_mean']-rows[b]['mid_low_mean'])
  interaction={v:pairs['AS50_minus_NOAS50'][v]['accuracy_pp']-pairs['AS30_minus_NOAS30'][v]['accuracy_pp'] for v in predictions[a]}
  per_seed[str(seed)]=dict(paired=pairs,source_clean_reference_within_0p1pp=pairs['AS50_minus_NOAS50']['fullV_clean_pp']>=-.1-1e-12,panel_accuracy_budget_interaction_pp=interaction,fullV_clean_budget_interaction_pp=pairs['AS50_minus_NOAS50']['fullV_clean_pp']-pairs['AS30_minus_NOAS30']['fullV_clean_pp'],mid_low_mean_budget_interaction_pp=pairs['AS50_minus_NOAS50']['mid_low_mean_pp']-pairs['AS30_minus_NOAS30']['mid_low_mean_pp'])
 paired_mean={name:{v:summary([per_seed[str(s)]['paired'][name][v]['accuracy_pp'] for s in ALL_SEEDS]) for v in predictions[a]} for name,_,_ in pairs_for(ALL_SEEDS[0])}
 value=dict(status='SOURCE_BUDGET_REPLICATION_READOUT_COMPLETE',rows=rows,per_seed=per_seed,paired_accuracy_pp_three_seed=paired_mean,source_high_urban_available=False,fullV_F1_or_error_recount_available=False,panel_draws=2,independent_physical_panel_samples=1800,target_scores_read=False,automatic_seed_extension=False,scientific_failure_skips_formal=False,repeatable_gain_proven=False,model_seeds=list(ALL_SEEDS),seed01_retrained=False,interpretation='Fixed recipe seed replication; source scientific negatives do not skip independent formal scoring; no target-driven selection')
 d.write(d.BASE/'source_matrix_decision.json',value);return value

def collect():
 result={};states=[]
 for arm,root in logical_rows():
  pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
  if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or pc['truth_read'] or pc['query_fit']:raise ValueError('All fixed seven-view predictions and independent scores required')
  scores=d.read(root/'scoring/scores.json')['results'];result[arm]={}
  for view in d.VIEWS:
   records=[v for v in scores if v['view']==view];overall=next(v for v in records if v['dimension']=='overall');rx=[v['accuracy'] for v in records if v['dimension']=='receiver'];result[arm][view]=dict(accuracy=overall['accuracy'],macro_F1=overall['macro_f1'],worst_RX=min(rx))
  result[arm]['urban_mean']=dict(accuracy=float(np.mean([result[arm][v]['accuracy'] for v in d.SCENES[3:]])))
  states.append(dict(arm=arm,root=str(root),reused=root.parent!=d.BASE))
 per_seed={}
 for seed in ALL_SEEDS:
  pairs={name:{v:100*(result[a][v]['accuracy']-result[b][v]['accuracy']) for v in (*d.VIEWS,'urban_mean')} for name,a,b in pairs_for(seed)}
  interaction={v:pairs['AS50_minus_NOAS50'][v]-pairs['AS30_minus_NOAS30'][v] for v in (*d.VIEWS,'urban_mean')}
  per_seed[str(seed)]=dict(paired_accuracy_pp=pairs,budget_interaction_pp=interaction,stage_clean_0p1pp=pairs['AS50_minus_NOAS50']['clean']>=-.1-1e-12)
 aggregate={name:{v:summary([per_seed[str(s)]['paired_accuracy_pp'][name][v] for s in ALL_SEEDS]) for v in (*d.VIEWS,'urban_mean')} for name,_,_ in pairs_for(ALL_SEEDS[0])}
 interactions={v:summary([per_seed[str(s)]['budget_interaction_pp'][v] for s in ALL_SEEDS]) for v in (*d.VIEWS,'urban_mean')}
 value=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS',rows=states,summary=result,per_seed=per_seed,paired_accuracy_pp_three_seed=aggregate,budget_interaction_pp_three_seed=interactions,model_seeds=list(ALL_SEEDS),single_seed=False,seed01_retrained=False,repeated_clean_gain_proven=False,source_decision=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').exists() else None,formal_feedback=False,goal_complete=False)
 d.write(d.BASE/'automatic_summary.json',value);return value
