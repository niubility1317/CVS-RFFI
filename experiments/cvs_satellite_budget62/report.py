"""Source-only fixed decision, followed by eight-row fixed-budget formal readout."""
from . import design as d

def logical_rows():return [(name,root) for name,root,_ in d.logical_rows()]
def stats(y,p,rx):
 import numpy as np
 cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(accuracy=float((p==y).mean()),macro_F1=float(f.mean()),worst_RX=float(min((p[rx==r]==y[rx==r]).mean() for r in np.unique(rx))))
def delta(a,b,y):
 gooda=a==y;goodb=b==y;repair=int((gooda&~goodb).sum());new=int((~gooda&goodb).sum())
 return dict(repaired=repair,new_errors=new,net=repair-new,accuracy_pp=100*(repair-new)/len(y),count=len(y))
def decision_from_rows(rows):
 clean=100*(float(rows['DET50']['full_V_clean'])-float(rows['NOAS50']['full_V_clean']))
 midlow=100*(float(rows['DET50']['mid_low_mean'])-float(rows['CE50']['mid_low_mean']))
 low=100*(float(rows['DET50']['panel']['practical_low_urban']['accuracy'])-float(rows['CE50']['panel']['practical_low_urban']['accuracy']))
 checks=dict(DET50_clean_vs_NoAs50_within_0p1pp=bool(clean>=-.1-1e-12),DET50_midlow_vs_CE50_at_least_0p5pp=bool(midlow>=.5-1e-12),DET50_low_not_below_CE50=bool(low>=-1e-12))
 return dict(checks=checks,source_qualified=bool(all(checks.values())),fullV_DET50_minus_NOAS50_clean_pp=float(clean),DET50_minus_CE50_midlow_pp=float(midlow),DET50_minus_CE50_low_pp=float(low))
def source_partial(error=None):
 missing=[]
 for name,root in logical_rows():
  for suffix in ('source/completion.json','source/source_contract.json','source/source_frozen.json','source_qualification/complete.json','source_qualification/predictions.npz'):
   if not (root/suffix).is_file():missing.append(dict(row=name,path=str(root/suffix)))
 value=dict(status='SOURCE_BUDGET_READOUT_PARTIAL',NN=0,model_seed_count=1,target_feedback=False,target_scores_read=False,formal_scores_read=False,source_qualified=False,rows={},missing=missing,error=error,automatic_seed_extension=False,scientific_failure_skips_formal=False,formal_evaluation_always_required=True,goal_complete=False)
 d.write(d.BASE/'source_matrix_decision.json',value);return value

def source_decision(allow_partial=False):
 import numpy as np
 if allow_partial:
  required=('source/completion.json','source/source_contract.json','source/source_frozen.json','source_qualification/complete.json','source_qualification/predictions.npz')
  if any(not (root/suffix).is_file() for _,root in logical_rows() for suffix in required):return source_partial()
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
  rows[arm]=dict(full_V_clean=float(done['final_source_metrics']['source_val_accuracy']),full_V_RX_accuracy=done['final_source_metrics']['source_val_rx_accuracy'],panel=panel,mid_low_mean=float(np.mean([panel[s]['accuracy'] for s in ('practical_mid_urban','practical_low_urban')])))
  predictions[arm]=preds
 pairs={}
 for a,b in [('DET50','NOAS50'),('CE50','NOAS50'),('DET50','CE50'),('DET50','RD50'),('CE50','RD50'),('DET50','DET30'),('CE50','CE30'),('RD50','RD30'),('NOAS50','NOAS30'),('DET30','CE30')]:
  pairs[a+'_minus_'+b]={v:dict(delta(predictions[a][v],predictions[b][v],y if v=='clean' else np.tile(y,2)),macro_F1_pp=100*(rows[a]['panel'][v]['macro_F1']-rows[b]['panel'][v]['macro_F1']),worst_RX_pp=100*(rows[a]['panel'][v]['worst_RX']-rows[b]['panel'][v]['worst_RX'])) for v in predictions[a]}
 interaction={v:pairs['DET50_minus_CE50'][v]['accuracy_pp']-pairs['DET30_minus_CE30'][v]['accuracy_pp'] for v in predictions['DET50']}
 value=dict(status='SOURCE_BUDGET_READOUT_COMPLETE',NN=0,model_seed_count=1,target_feedback=False,rows=rows,paired=pairs,dropout_by_budget_interaction_pp=interaction,**decision_from_rows(rows),source_high_urban_available=False,fullV_F1_or_error_recount_available=False,panel_draws=2,independent_physical_panel_samples=1800,target_scores_read=False,formal_scores_read=False,automatic_seed_extension=False,scientific_failure_skips_formal=False,repeatable_gain_proven=False,interpretation='Source only fixed decision;totalE50 extension,not a newmechanism or newseed;failure closes recipe without sweep/rescue')
 d.write(d.BASE/'source_matrix_decision.json',value);return value

def collect():
 result={};states=[]
 for arm,root,reused in d.logical_rows():
  pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
  if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or pc['truth_read'] or pc['query_fit']:raise ValueError('All seven fixed-view predictions/independent scores required')
  scores=d.read(root/'scoring/scores.json')['results'];result[arm]={}
  for view in d.VIEWS:
   records=[v for v in scores if v['view']==view];overall=next(v for v in records if v['dimension']=='overall');rx=[v['accuracy'] for v in records if v['dimension']=='receiver'];result[arm][view]=dict(accuracy=float(overall['accuracy']),macro_F1=float(overall['macro_f1']),worst_RX=float(min(rx)))
  result[arm]['urban_mean']=dict(accuracy=float(sum(result[arm][v]['accuracy'] for v in d.SCENES[3:])/3))
  states.append(dict(arm=arm,root=str(root),reused=reused))
 pairs={}
 for a,b in [('DET50','NOAS50'),('CE50','NOAS50'),('DET50','CE50'),('DET50','RD50'),('CE50','RD50'),('DET50','DET30'),('CE50','CE30'),('RD50','RD30'),('NOAS50','NOAS30'),('DET30','CE30')]:pairs[a+'_minus_'+b]={v:100*(result[a][v]['accuracy']-result[b][v]['accuracy']) for v in (*d.VIEWS,'urban_mean')}
 interaction={v:pairs['DET50_minus_CE50'][v]-pairs['DET30_minus_CE30'][v] for v in (*d.VIEWS,'urban_mean')}
 value=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS',rows=states,summary=result,paired_accuracy_pp=pairs,dropout_by_budget_interaction_pp=interaction,stage_clean_0p1pp=bool(pairs['DET50_minus_NOAS50']['clean']>=-.1-1e-12),single_seed=True,repeated_clean_gain_proven=False,source_decision=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').exists() else None,formal_feedback=False,goal_complete=False)
 d.write(d.BASE/'automatic_summary.json',value);return value
