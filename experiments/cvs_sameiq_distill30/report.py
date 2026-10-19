"""Fixed source decision and seven-view same-row summary, after prediction freeze."""
import numpy as np
from pathlib import Path
from . import design as d
def logical_rows():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 return [('NOAS30',ref/'B00-s2026092701'),('AS30',ref/'B01-s2026092701'),('CE',d.BASE/'C01-s2026092701'),('CSIL',d.BASE/'D01-s2026092701')]
def source_decision():
 rows={};ids0=y0=None
 for label,root in logical_rows():
  done=d.read(root/'source/completion.json');q=d.read(root/'source_qualification/complete.json')
  if done['target_access'] or q['target_access'] or done['final_source_metrics']['source_val_count']!=27000:raise ValueError('Frozen legal source readouts required')
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as f:
   ids=f['ids'];y=f['TX'];sat=f['satellite']
   if ids0 is None:ids0=ids.copy();y0=y.copy()
   if not np.array_equal(ids,ids0) or not np.array_equal(y,y0) or sat.shape!=(2,3,1800,6):raise ValueError('Same source observations required')
   acc=(sat.argmax(-1)==y).mean(-1)
  rows[label]=dict(fullV_clean=done['final_source_metrics']['source_val_accuracy'],scene_by_draw=acc.tolist(),midlow=float(acc[:,1:].mean()),low=float(acc[:,2].mean()))
 clean=100*(rows['CSIL']['fullV_clean']-rows['NOAS30']['fullV_clean']);midlow=100*(rows['CSIL']['midlow']-rows['CE']['midlow']);low=100*(rows['CSIL']['low']-rows['AS30']['low'])
 value=dict(rows=rows,promotion=dict(passed=clean>=-.1-1e-12 and midlow>=.5-1e-12 and low>=-1e-12,clean_vs_NoAs_pp=clean,midlow_vs_CE_pp=midlow,low_vs_originalAs_pp=low,automatic_extension=False,source_failure_skips_formal=False),formal_scores_read=False,model_seed_count=1)
 d.write(d.BASE/'source_matrix_decision.json',value);return value
def collect():
 summary={};pred={};cost={}
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 from .evaluate import observations
 for label,root in logical_rows():
  pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
  if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or pc['truth_read']:raise ValueError('All four frozen rows/seven views required')
  records=d.read(root/'scoring/scores.json')['results'];summary[label]={}
  for view in d.VIEWS:
   overall=next(x for x in records if x['view']==view and x['dimension']=='overall');worst=min(x['accuracy'] for x in records if x['view']==view and x['dimension']=='receiver');summary[label][view]=dict(accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_RX=worst)
  with np.load(root/'prediction/predictions.npz',allow_pickle=False) as f:pred[label]={v:f[v].copy() for v in d.VIEWS};pred[label]['ids']=f['ids'].copy()
  if label in ('CE','CSIL'):cost[label]=d.read(root/'source/result.json')
 ids,_=observations();y,rx,days=truth_arrays(d.read(d.TRUTH),ids);pairs={}
 for a,b in [('CSIL','CE'),('CSIL','AS30'),('CSIL','NOAS30'),('CE','AS30')]:
  if not np.array_equal(pred[a]['ids'],ids) or not np.array_equal(pred[b]['ids'],ids):raise ValueError('Formal physical ID order differs')
  pairs[a+'_minus_'+b]={}
  for v in d.VIEWS:
   gooda=pred[a][v]==y;goodb=pred[b][v]==y;repair=int((gooda&~goodb).sum());damage=int((~gooda&goodb).sum());pairs[a+'_minus_'+b][v]=dict(repair=repair,damage=damage,delta_pp=(repair-damage)*100/len(y))
 result=dict(status='ARTIFACTS_COMPLETE',summary=summary,paired=pairs,cost=cost,source_decision=d.read(d.BASE/'source_matrix_decision.json'),single_model_seed=True,goal_complete=False,truth_last=True,formal_feedback=False)
 d.write(d.BASE/'automatic_summary.json',result);return result
