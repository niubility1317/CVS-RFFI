"""Frozen source decision and independent same-row truth-last aggregate; no model fitting."""
from pathlib import Path
import numpy as np
from . import design as d
PAIRS=(('SIGNAL30','NRAW30'),('SIGNAL30','ARAW30'),('SIGNAL30','AS30'),('ARAW30','AS30'),('ARAW30','NRAW30'),('NRAW30','NOAS30'),('AS30','NOAS30'))
def logical_rows():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 return [('NOAS30',ref/'B00-s2026092701'),('AS30',ref/'B01-s2026092701'),('NRAW30',d.BASE/'N01-s2026092701'),('ARAW30',d.BASE/'C01-s2026092701'),('SIGNAL30',d.BASE/'S01-s2026092701')]
def stats(y,p,rx):
 cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(count=int(len(y)),correct=int((p==y).sum()),accuracy=float((p==y).mean()),macro_F1=float(f.mean()),worst_RX=float(min((p[rx==r]==y[rx==r]).mean() for r in np.unique(rx))),confusion=cm.tolist())
def delta(a,b,y):
 gooda=a==y;goodb=b==y;repair=int((gooda&~goodb).sum());new=int((~gooda&goodb).sum())
 return dict(repaired=repair,new_errors=new,net=repair-new,accuracy_pp=100*(repair-new)/len(y),count=len(y))
def source_decision():
 rows={};predictions={};base_ids=base_y=base_rx=None
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
 pairs={a+'_minus_'+b:{v:dict(delta(predictions[a][v],predictions[b][v],base_y if v=='clean' else np.tile(base_y,2)),macro_F1_pp=100*(rows[a]['panel'][v]['macro_F1']-rows[b]['panel'][v]['macro_F1']),worst_RX_pp=100*(rows[a]['panel'][v]['worst_RX']-rows[b]['panel'][v]['worst_RX'])) for v in predictions[a]} for a,b in PAIRS}
 fullV={a+'_minus_'+b:100*(rows[a]['full_V_clean']-rows[b]['full_V_clean']) for a,b in PAIRS}
 dc=float(fullV['SIGNAL30_minus_NRAW30']);ds=float(100*(rows['SIGNAL30']['mid_low_mean']-rows['ARAW30']['mid_low_mean']))
 dr=float(100*(rows['SIGNAL30']['mid_low_mean']-rows['AS30']['mid_low_mean']))
 promotion=dict(passed=bool(dc>=-.1-1e-12 and ds>=.5-1e-12 and dr>=-1e-12),candidate_minus_originalAs_midlow_pp=dr,candidate_minus_matchedNoAs_fullV_clean_pp=dc,candidate_minus_AsRaw_midlow_pp=ds,clean_min_pp=-.1,midlow_min_pp=.5,source_failure_skips_formal=False,automatic_extension=False,no_HQgap_weight_seed_rescue=True)
 value=dict(status='SOURCE_SIGNAL_BRIDGE_READOUT_COMPLETE',rows=rows,paired=pairs,fullV_clean_paired_pp=fullV,promotion=promotion,panel_draws=2,independent_model_seeds=1,independent_physical_panel_samples=1800,target_scores_read=False,scientific_failure_skips_formal=False,repeatable_gain_proven=False,source_high_urban_available=False,interpretation='One fixed source test;qualification concerns future E50 only,all5 logical rows complete all7 formal regardless')
 d.write(d.BASE/'source_matrix_decision.json',value);return value

def collect():
 for arm,root in logical_rows():
  pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
  if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or pc['truth_read'] or pc['query_fit']:raise ValueError('Truth closed until all5 all7 predictions and independent scores complete')
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 from .evaluate import observations,COUNT
 ids,_=observations();y,rx,days=truth_arrays(d.read(d.TRUTH),ids);result={};pred={};costs={}
 for arm,root in logical_rows():
  pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
  if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or pc['truth_read'] or pc['query_fit']:raise ValueError('All seven fixed predictions and independent scores required')
  saved=d.read(root/'scoring/scores.json')['results'];result[arm]={};pred[arm]={}
  with np.load(root/'prediction/predictions.npz',allow_pickle=False) as f:
   if not np.array_equal(f['ids'],ids):raise ValueError('Same official physical observations required')
   for view in d.VIEWS:
    p=f[view].copy();pred[arm][view]=p;value=stats(y,p,rx);strata={}
    for dim,x in [('receiver',rx),('day',days),('transmitter',y)]:
     strata[dim]={str(v):stats(y[x==v],p[x==v],rx[x==v]) for v in np.unique(x)}
    value['strata']=strata
    for record in [z for z in saved if z['view']==view]:
     check=value if record['dimension']=='overall' else strata[record['dimension']][record['stratum']]
     if abs(record['accuracy']-check['accuracy'])>1e-12 or abs(record['macro_f1']-check['macro_F1'])>1e-12:raise ValueError('Independent formal recount differs')
    result[arm][view]=value
  result[arm]['urban_mean']=dict(accuracy=float(np.mean([result[arm][v]['accuracy'] for v in d.SCENES[3:]])))
  result[arm]['suburban_gap_pp']={v:100*(result[arm]['clean']['accuracy']-result[arm][v]['accuracy']) for v in d.SCENES[:3]}
  costs[arm]=dict(reused=arm in ('NOAS30','AS30'),training=d.read(root/'source/result.json'),source_readout_NN=12600,formal_NN=pc['actual_NN'],formal_wall_seconds=pc['wall_seconds'],extra_model_forwards=(0 if arm in ('NOAS30','AS30') else 284160),new_training=arm not in ('NOAS30','AS30'))
 pairs={}
 for a,b in PAIRS:
  key=a+'_minus_'+b;pairs[key]={}
  for v in d.VIEWS:
   val=dict(delta(pred[a][v],pred[b][v],y),macro_F1_pp=100*(result[a][v]['macro_F1']-result[b][v]['macro_F1']),worst_RX_pp=100*(result[a][v]['worst_RX']-result[b][v]['worst_RX']))
   val['strata']={dim:{str(k):delta(pred[a][v][x==k],pred[b][v][x==k],y[x==k]) for k in np.unique(x)} for dim,x in [('receiver',rx),('day',days),('transmitter',y)]};pairs[key][v]=val
  pairs[key]['urban_mean_accuracy_pp']=100*(result[a]['urban_mean']['accuracy']-result[b]['urban_mean']['accuracy'])
 value=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS',summary=result,paired=pairs,costs=costs,stage_clean_0p1pp=bool(pairs['SIGNAL30_minus_NRAW30']['clean']['accuracy_pp']>=-.1-1e-12),single_seed=True,independent_model_seeds=1,repeated_clean_gain_proven=False,source_decision=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').exists() else None,formal_feedback=False,goal_complete=False,truth_last=True,independent_recount='VERIFIED')
 d.write(d.BASE/'automatic_summary.json',value);return value
