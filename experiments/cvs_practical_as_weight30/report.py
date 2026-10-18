"""Three frozen same-seed conditions; summaries never alter training or selection."""
from pathlib import Path
import numpy as np
from . import design as d

def logical_rows():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 return [('NOAS_ORIGINAL',ref/'B00-s2026092701'),('AS_ORIGINAL',ref/'B01-s2026092701'),('NOAS_WEIGHT',d.BASE/'B00-s2026092701'),('AS_WEIGHT',d.BASE/'B01-s2026092701')]
PAIRS=(('AS_WEIGHT','NOAS_WEIGHT'),('AS_ORIGINAL','NOAS_ORIGINAL'),('AS_WEIGHT','AS_ORIGINAL'),('NOAS_WEIGHT','NOAS_ORIGINAL'))
def stats(y,p,rx):
 cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(accuracy=float((p==y).mean()),macro_F1=float(f.mean()),worst_RX=float(min((p[rx==r]==y[rx==r]).mean() for r in np.unique(rx))))
def delta(a,b,y):
 gooda=a==y;goodb=b==y;repair=int((gooda&~goodb).sum());new=int((~gooda&goodb).sum())
 return dict(repaired=repair,new_errors=new,net=repair-new,accuracy_pp=100*(repair-new)/len(y),count=len(y))

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
 pairs={}
 for a,b in PAIRS:
  pairs[a+'_minus_'+b]={v:dict(delta(predictions[a][v],predictions[b][v],y if v=='clean' else np.tile(y,2)),macro_F1_pp=100*(rows[a]['panel'][v]['macro_F1']-rows[b]['panel'][v]['macro_F1']),worst_RX_pp=100*(rows[a]['panel'][v]['worst_RX']-rows[b]['panel'][v]['worst_RX'])) for v in predictions[a]}
 fullV={a+'_minus_'+b:100*(rows[a]['full_V_clean']-rows[b]['full_V_clean']) for a,b in PAIRS}
 increment_clean=float(fullV['AS_WEIGHT_minus_AS_ORIGINAL']);increment_midlow=float(100*(rows['AS_WEIGHT']['mid_low_mean']-rows['AS_ORIGINAL']['mid_low_mean']))
 promotion=dict(passed=bool(increment_clean>=-.1-1e-12 and increment_midlow>=.5-1e-12),newAs_minus_originalAs_fullV_clean_pp=increment_clean,newAs_minus_originalAs_midlow_pp=increment_midlow,clean_min_pp=-.1,midlow_min_pp=.5,controls_not_additional_threshold=True,future_E50_or_seed_expansion_only=True,source_failure_skips_formal=False,automatic_extension=False)
 interaction={v:float(pairs['AS_WEIGHT_minus_NOAS_WEIGHT'][v]['accuracy_pp']-pairs['AS_ORIGINAL_minus_NOAS_ORIGINAL'][v]['accuracy_pp']) for v in predictions['AS_WEIGHT']}
 value=dict(status='SOURCE_ASWEIGHT_READOUT_COMPLETE',rows=rows,paired=pairs,fullV_clean_paired_pp=fullV,promotion=promotion,panel_interaction_accuracy_pp=interaction,fullV_clean_interaction_pp=float(fullV['AS_WEIGHT_minus_NOAS_WEIGHT']-fullV['AS_ORIGINAL_minus_NOAS_ORIGINAL']),source_high_urban_available=False,fullV_F1_or_error_recount_available=False,panel_draws=2,independent_physical_panel_samples=1800,target_scores_read=False,automatic_seed_extension=False,scientific_failure_skips_formal=False,repeatable_gain_proven=False,interpretation='Fixed source comparison;only future budget/seed promotion uses two declared source criteria;all7 formal mandatory for both newrows;one seed')
 d.write(d.BASE/'source_matrix_decision.json',value);return value

def collect():
 result={};states=[]
 for arm,root in logical_rows():
  pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
  if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or pc['truth_read'] or pc['query_fit']:raise ValueError('All seven fixed predictions and independent scores required')
  scores=d.read(root/'scoring/scores.json')['results'];result[arm]={}
  for view in d.VIEWS:
   records=[v for v in scores if v['view']==view];overall=next(v for v in records if v['dimension']=='overall');rx=[v['accuracy'] for v in records if v['dimension']=='receiver'];result[arm][view]=dict(accuracy=overall['accuracy'],macro_F1=overall['macro_f1'],worst_RX=min(rx))
  result[arm]['urban_mean']=dict(accuracy=float(np.mean([result[arm][v]['accuracy'] for v in d.SCENES[3:]])))
  states.append(dict(arm=arm,root=str(root),reused=arm.endswith('ORIGINAL')))
 pairs={a+'_minus_'+b:{v:100*(result[a][v]['accuracy']-result[b][v]['accuracy']) for v in (*d.VIEWS,'urban_mean')} for a,b in PAIRS}
 value=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS',rows=states,summary=result,paired_accuracy_pp=pairs,stage_clean_0p1pp=bool(pairs['AS_WEIGHT_minus_NOAS_WEIGHT']['clean']>=-.1-1e-12),interaction_accuracy_pp={v:float(pairs['AS_WEIGHT_minus_NOAS_WEIGHT'][v]-pairs['AS_ORIGINAL_minus_NOAS_ORIGINAL'][v]) for v in (*d.VIEWS,'urban_mean')},single_seed=True,repeated_clean_gain_proven=False,source_decision=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').exists() else None,formal_feedback=False,goal_complete=False)
 d.write(d.BASE/'automatic_summary.json',value);return value
