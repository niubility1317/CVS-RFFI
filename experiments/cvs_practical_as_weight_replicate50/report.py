"""Completion-only producer summary;24logical metric analysis is independent."""
from . import design as d

def collect():
 rows=[]
 for row in d.rows():
  for budget in (30,50):
   root=d.stage_root(row,budget);pred=d.read(root/'prediction/complete.json');score=d.read(root/'scoring/complete.json');done=d.read(root/'complete.json')
   if pred['status']!='PREDICTIONS_COMPLETE' or pred['views']!=list(d.VIEWS) or pred['truth_read'] or pred['query_fit'] or score['status']!='SCORED_COMPLETE' or done['status']!='TRAINED_AND_SCORED':raise ValueError('All eight fixed newstage sevenview closures required')
   rows.append(dict(row_id=row['row_id'],model_seed=row['model_seed'],target_budget=budget,root=str(root),source_readout_complete=(root/'source_qualification/complete.json').is_file(),source_readout_failure=(root/'source_qualification_failure.json').is_file(),all7_prediction_complete=True,independent_scoring_complete=True))
 value=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS',model_seeds=list(d.SEEDS),new_trajectories=4,new_stage_rows=8,new_model_epochs=120,new_updates=26640,rows=rows,scientific_continuation_unconditional=True,selection_from_intermediate_source_or_formal=False,seed01_or_original_rows_retrained=False,goal_complete=False)
 d.write(d.BASE/'automatic_summary.json',value);return value
