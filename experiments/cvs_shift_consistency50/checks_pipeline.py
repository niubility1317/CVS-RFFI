"""Synthetic closure and truth-order checks; never reads real query or truth."""
import json,tempfile,copy
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from . import score,predict
from .model import build

def reject(fn,kind=ValueError):
 try:fn()
 except kind:return
 raise AssertionError('Expected malformed protocol to fail')

def run():
 torch.set_num_threads(2)
 c=d.config(d.rows()[0]);bad=copy.deepcopy(c);bad['source_U']=True;reject(lambda:d.validate(bad))
 with patch('torch.load',side_effect=AssertionError('Training model must not load checkpoint')):build(c,'cpu')
 with patch.object(predict,'preflight',side_effect=ValueError('prediction incomplete')):
  with patch('experiments.cvs_report_ssl.evaluate.truth_arrays',side_effect=AssertionError('Truth was opened too early')):reject(lambda:score.score(c['row_id']))
 original=d.BASE
 with tempfile.TemporaryDirectory(prefix='sampling_closure_',dir=d.ROOT/'local_artifacts') as folder:
  d.BASE=Path(folder)
  try:
   for row in d.rows():
    c=d.config(row);p=d.BASE/row['row_id'];(p/'scores').mkdir(parents=True)
    rate={'CE50':.85,'SHIFT_CE50':.86,'SHIFT_STATS50':.87,'SHIFT_STATS_SSL50':.80}[row['arm']]
    correct=round(28000*rate);cm=[[0]*6 for _ in range(6)]
    for k in range(6):cm[k][k]=correct;cm[k][(k+1)%6]=28000-correct
    records=[dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],view='clean',dimension='overall' if i==0 else 'receiver',accuracy=rate,macro_f1=rate,query_count=168000,confusion=cm) for i in range(18)]
    d.write(p/'scores/metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
    d.write(p/'completion.json',dict(status='TRAINED_PREDICTED_SCORED',config=c,views=['clean'],count_per_view=168000,metric_records=18,truth_last=True,independent_recount='VERIFIED'))
   value=score.finish()
   assert value['models']==12 and value['prediction_count']==2016000 and not value['goal_84']
   assert len(value['summary'])==4 and len(value['paired'])==4
   last=d.BASE/d.rows()[-1]['row_id']/'completion.json';v=d.read(last);v['truth_last']=False;d.write(last,v);reject(score.finish)
  finally:d.BASE=original
 return dict(status='PASS',source_U_toggle_rejected=True,scratch_model_no_checkpoint=True,truth_closed_before_prediction=True,all12rows_three_seed_closure=True,primary_goal_not_granted_by_control=True,missing_truthlast_rejected=True,target_access=False)

if __name__=='__main__':print(json.dumps(run()))
