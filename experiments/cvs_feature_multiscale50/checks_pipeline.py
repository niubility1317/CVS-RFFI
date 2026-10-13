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
 c=d.config(d.rows()[0]);bad=copy.deepcopy(c);bad['features']=[];reject(lambda:d.validate(bad))
 args=vars(d.make_args(c,'cpu'));predict.validate_recipe_args(args,c)
 for key,value in (('use_ema_teacher',False),('pseudo_temporal_mode','epoch_bank'),('label_epochs',130),('output_dir','unrelated'),('run_id','wrong')):
  malformed=dict(args,**{key:value});reject(lambda:predict.validate_recipe_args(malformed,c))
 with patch('torch.load',side_effect=AssertionError('Training model must not load checkpoint')):build(c,'cpu')
 with patch.object(predict,'preflight',side_effect=ValueError('prediction incomplete')):
  with patch('experiments.cvs_report_ssl.evaluate.truth_arrays',side_effect=AssertionError('Truth was opened too early')):reject(lambda:score.score(c['row_id']))
 from .dispatch import eligible
 first=d.rows()[0];assert eligible(first)
 later=next(r for r in d.rows() if r['arm']==first['arm'] and r['model_seed']!=d.SEEDS[0])
 with patch('pathlib.Path.is_file',return_value=False):assert not eligible(later)
 with patch('pathlib.Path.is_file',side_effect=lambda p:p.parent.parent.name!=d.rows()[3]['row_id'],autospec=True):assert not eligible(later)
 with patch('pathlib.Path.is_file',return_value=True),patch.object(predict,'source_proof',return_value=Path('.')) as proof,patch.object(d,'read',return_value={'final_training_L_CE':.4}):
  assert eligible(later) and proof.call_count==4
 with patch('pathlib.Path.is_file',return_value=True),patch.object(predict,'source_proof',return_value=Path('.')),patch.object(d,'read',return_value={'final_training_L_CE':.6}):reject(lambda:eligible(later))
 original=d.BASE
 with tempfile.TemporaryDirectory(prefix='sampling_closure_',dir=d.ROOT/'local_artifacts') as folder:
  d.BASE=Path(folder)
  try:
   for row in d.rows():
    c=d.config(row);p=d.BASE/row['row_id'];(p/'scores').mkdir(parents=True)
    rate={'OLD_GATE_COS50':.85,'TIME_MS50':.86,'PA_MS50':.87,'JOINT_MS50':.80}[row['arm']]
    correct=round(28000*rate);cm=[[0]*6 for _ in range(6)]
    for k in range(6):cm[k][k]=correct;cm[k][(k+1)%6]=28000-correct
    records=[dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],view='clean',dimension='overall' if i==0 else 'receiver',accuracy=rate,macro_f1=rate,query_count=168000,confusion=cm) for i in range(18)]
    d.write(p/'scores/metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
    d.write(p/'completion.json',dict(status='TRAINED_PREDICTED_SCORED',config=c,views=['clean'],count_per_view=168000,metric_records=18,truth_last=True,independent_recount='VERIFIED'))
   value=score.finish()
   assert value['models']==12 and value['prediction_count']==2016000 and not value['goal_84']
   assert len(value['summary'])==4 and len(value['paired'])==5
   last=d.BASE/d.rows()[-1]['row_id']/'completion.json';v=d.read(last);v['truth_last']=False;d.write(last,v);reject(score.finish)
  finally:d.BASE=original
 from unittest.mock import patch as mock
 import numpy as np
 import dataset_wisig as ds
 previous=ds._TORCH_FROM_NUMPY_OK
 sample=np.linspace(-1,1,512,dtype=np.float32).reshape(2,256)
 expected=torch.tensor(sample.tolist(),dtype=torch.float32)
 real_as_tensor=torch.as_tensor
 def unavailable(value,*a,**kw):
  if isinstance(value,np.ndarray):raise RuntimeError('NumPy inference unavailable')
  return real_as_tensor(value,*a,**kw)
 try:
  ds._TORCH_FROM_NUMPY_OK=None
  with mock('torch.from_numpy',side_effect=RuntimeError('NumPy unavailable')),mock('torch.as_tensor',side_effect=unavailable):actual=ds._safe_to_torch_float_tensor(sample)
  assert torch.equal(actual,expected) and actual.dtype==torch.float32
 finally:ds._TORCH_FROM_NUMPY_OK=previous
 return dict(status='PASS',legacy_numpy_fallback_value_equivalence=True,wrong_recipe_andoutput_identity_rejected=True,all_four_firstseed_before_expansion=True,first_seed_source_falsification_without_V_or_target=True,legacy_gate_andEMA_removal_rejected=True,scratch_model_no_checkpoint=True,truth_closed_before_prediction=True,all12rows_three_seed_closure=True,primary_goal_not_granted_by_control=True,missing_truthlast_rejected=True,target_access=False)

if __name__=='__main__':print(json.dumps(run()))
