"""Source-only development decision;formal summaries after all row predictions/score."""
from pathlib import Path
import numpy as np
from . import design as d

def logical_rows():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 return [('NOAS',ref/'B00-s2026092701'),('AS_RD',ref/'B01-s2026092701')]+[(r['arm'],d.BASE/r['row_id']) for r in d.rows()]

def source_decision():
 rows={};base_ids=None
 for arm,root in logical_rows():
  done=d.read(root/'source/completion.json');sc=d.read(root/'source/source_contract.json');frozen=d.read(root/'source/source_frozen.json');q=d.read(root/'source_qualification/complete.json')
  if sc['role_ids']!=d.read(d.SOURCE)['role_ids'] or done['target_access'] or frozen['target_access'] or q['target_access'] or done['final_source_metrics']['source_val_count']!=27000:raise ValueError('Exact frozen fullsource controls required')
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as a:
   ids=a['ids'];y=a['TX'];sat=a['satellite'];rx=a['RX'];day=a['day']
   if base_ids is None:base_ids=ids.copy()
   if len(ids)!=1800 or not np.array_equal(ids,base_ids) or sat.shape!=(2,3,1800,6) or not np.isfinite(sat).all():raise ValueError('Same fixed source panel required')
   acc=(sat.argmax(-1)==y).mean(-1)*100
  rows[arm]=dict(full_V_clean=done['final_source_metrics']['source_val_accuracy'],mid_low_by_draw_pp=acc[:,1:].tolist(),mid_low_mean_pp=float(acc[:,1:].mean()),source_RX_accuracy=done['final_source_metrics']['source_val_rx_accuracy'])
 det=rows['AS_DET'];old=rows['AS_RD'];ce=rows['AS_CE'];no=rows['NOAS'];clean=100*(det['full_V_clean']-no['full_V_clean']);clean_old=100*(det['full_V_clean']-old['full_V_clean']);clean_ce=100*(det['full_V_clean']-ce['full_V_clean']);urban=det['mid_low_mean_pp']-old['mid_low_mean_pp'];urban_ce=det['mid_low_mean_pp']-ce['mid_low_mean_pp']
 conditions=dict(stage_clean=clean>=-.1-1e-12,retained_As_gain=det['mid_low_mean_pp']-no['mid_low_mean_pp']>=5.,useful_vs_original=(urban>=.5 and clean_old>=-.1-1e-12) or (clean_old>=.1-1e-12 and urban>=-.1-1e-12),dropout_attribution=(urban_ce>=.5 and clean_ce>=-.1-1e-12) or (clean_ce>=.1-1e-12 and urban_ce>=-.1-1e-12))
 value=dict(status='SOURCE_DECISION_COMPLETE',conditions=conditions,passed=all(conditions.values()),rows=rows,paired=dict(DET_minus_NOAS_clean_pp=clean,DET_minus_RD_clean_pp=clean_old,DET_minus_CE_clean_pp=clean_ce,DET_minus_RD_midlow_pp=urban,DET_minus_CE_midlow_pp=urban_ce),target_scores_read=False,automatic_seed_extension=False,on_fail='Close fixed satellite-dropout bypass hypothesis;no rate/slot/window/seed/budget rescue',process_identity_preservation_gate=False,source_diagnostics_not_DG_claim=True)
 d.write(d.BASE/'source_matrix_decision.json',value);return value

def collect():
 # Full log/cost/paired-identity analysis follows;this is automatic artifact closure.
 result={};states=[]
 for arm,root in logical_rows():
  pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
  if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or pc['truth_read'] or pc['query_fit']:raise ValueError('All declared sevenview predictions/independent scores required')
  scores=d.read(root/'scoring/scores.json')['results'];result[arm]={}
  for view in d.VIEWS:
   rows=[v for v in scores if v['view']==view];overall=next(v for v in rows if v['dimension']=='overall');rx=[v['accuracy'] for v in rows if v['dimension']=='receiver'];result[arm][view]=dict(accuracy=overall['accuracy'],macro_F1=overall['macro_f1'],worst_RX=min(rx))
  states.append(dict(arm=arm,root=str(root),reused=arm in ('NOAS','AS_RD')))
 paired={}
 for arm in ('AS_RD','AS_CE','AS_DET'):
  paired[arm]={view:100*(result[arm][view]['accuracy']-result['NOAS'][view]['accuracy']) for view in d.VIEWS}
  paired[arm]['urban_mean_pp']=sum(paired[arm][s] for s in d.SCENES[3:])/3
 value=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS',rows=states,summary=result,paired_against_NoAs_pp=paired,stage_clean_0p1pp={arm:paired[arm]['clean']>=-.1-1e-12 for arm in paired},single_seed=True,repeated_clean_gain_proven=False,source_decision=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').exists() else None,formal_feedback=False,goal_complete=False)
 d.write(d.BASE/'automatic_summary.json',value);return value
