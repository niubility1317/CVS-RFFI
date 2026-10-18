"""Fixed source rules before final score collection;never changes live rows."""
import numpy as np
from . import design as d

def source_decision():
 if (d.BASE/'source_matrix_decision.json').exists():return d.read(d.BASE/'source_matrix_decision.json')
 cs={r['arm']:d.read(d.BASE/r['row_id']/'source/completion.json') for r in d.rows()};qs={r['arm']:d.read(d.BASE/r['row_id']/'source_qualification/complete.json') for r in d.rows()}
 reference=d.REFERENCE/('B01-s'+str(d.SEED));old=d.read(reference/'source/completion.json');oldq=d.read(reference/'source_qualification/complete.json');v={k:x['final_source_metrics'] for k,x in cs.items()};ov=old['final_source_metrics'];sat=np.asarray(qs['B01']['source_accuracies']);prev=np.asarray(oldq['source_accuracies'])
 if sat.shape!=(2,3) or prev.shape!=(2,3):raise ValueError('Fixed existing source two-draw panel changed')
 checks=dict(clean_not_below_matched_NoAs=v['B01']['source_val_accuracy']>=v['B00']['source_val_accuracy'],clean_vs_staged_As_within_quarter_pp=v['B01']['source_val_accuracy']>=ov['source_val_accuracy']-.0025,each_RX_vs_staged_As_within_half_pp=all(v['B01']['source_val_rx_accuracy'][str(r)]>=ov['source_val_rx_accuracy'][str(r)]-.005 for r in d.RX),both_draw_mid_low_vs_staged_As_gain_one_pp=bool(((sat-prev)[:,1:]>=.01).all()))
 q=dict(status='SOURCE_CRITERIA_PASS' if all(checks.values()) else 'SOURCE_CRITERIA_FAIL_FIXED_CONTINUOUS_ROUTE_CLOSED',criteria=checks,source_qualified=all(checks.values()),source_fullV=v,reference_staged_As_fullV=ov,source_panel_accuracies={k:x['source_accuracies'] for k,x in qs.items()},reference_panel_accuracies=oldq['source_accuracies'],mid_low_gains=(sat-prev)[:,1:].tolist(),reference_run=str(d.REFERENCE),model_seed=d.SEED,statistics_not_target=True,formal_scores_read=False,new_Ag=False,automatic_seed_extension=False,goal_achieved=False)
 d.write(d.BASE/'source_matrix_decision.json',q);return q

def collect():
 q=source_decision();results=[];cost=[]
 for r in d.rows():
  row=r['row_id'];p=d.BASE/row
  if d.read(p/'scoring/complete.json')['status']!='SCORED_COMPLETE' or d.read(p/'prediction/complete.json')['views']!=list(d.VIEWS):raise ValueError('Full7views required')
  results+=d.read(p/'scoring/scores.json')['results'];cost.append(d.read(p/'source/result.json'))
 d.write(d.BASE/'final_scores.json',dict(status='FIXED_TWO_ROW_SCORED',results=results,source_decision=q,cost=cost,new_model_epochs=60,new_updates=13320,new_Ag=False,all_rows_reported=True,formal_test_unchanged=True,target_feedback_forbidden=True,goal_achieved=False))
 return dict(status='MATRIX_SCORED',rows=[r['row_id'] for r in d.rows()],source_qualified=q['source_qualified'],goal_achieved=False)
