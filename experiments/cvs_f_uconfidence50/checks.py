import json,tempfile
from pathlib import Path
import numpy as np
from confidence import describe
from runner import source_scope,row_scope,reference_scope

def run():
 p=np.array([[.97,.03],[.04,.96],[.6,.4]]);r=describe(p,np.array([1,1,3]),np.array([1,2,1]),['A','B']);g=r['global_summary'];records=g['per_predicted_TX']
 assert [v['predicted_count'] for v in records]==[2,1] and abs(g['mean_confidence']-(.97+.96+.6)/3)<1e-12
 assert g['confidence_tail_count']=={'0.92':2,'0.95':2,'0.97':1} and records[0]['confidence_tail_count']['0.97']==1 and records[1]['confidence_tail_count']['0.97']==0
 assert r['by_RX']['3']['per_predicted_TX'][1]['mean_confidence'] is None
 rr=describe(p[:,::-1],np.array([1,1,3]),np.array([1,2,1]),['B','A']);assert rr['global_summary']['per_predicted_TX'][0]['predicted_count']==1
 assert abs(g['mean_entropy']-float(np.mean([-sum(v*np.log(v) for v in row) for row in p])))<1e-12
 negative=0
 for role in ('L_s','V','target','query'):
  try:source_scope(role)
  except ValueError:negative+=1
  else:raise AssertionError('Forbidden role accepted')
 try:row_scope('WEIGHT-s2026092701')
 except ValueError:negative+=1
 else:raise AssertionError('Unregistered snapshot accepted')
 for bad in (p*2,np.full_like(p,np.nan)):
  try:describe(bad,np.array([1,1,3]),np.array([1,2,1]),['A','B'])
  except ValueError:negative+=1
  else:raise AssertionError('Invalid posterior accepted')
 with tempfile.TemporaryDirectory() as tmp:
  try:reference_scope(Path(tmp),Path(tmp))
  except ValueError:negative+=1
  else:raise AssertionError('Wrong reference accepted')
 return dict(status='PASS',known_probability_oracle=True,empty_predicted_class_safe=True,class_column_permutation_equivariant=True,entropy_oracle=True,protocol_negative_cases=negative,IQ_reads=0,checkpoint_reads=0,U_truth_read=False,target_reads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
