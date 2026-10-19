"""Known-angle oracle and source-scope negative checks; no IQ/checkpoints."""
import json,tempfile
from pathlib import Path
import numpy as np
from geometry import describe
from runner import source_scope,reference_scope,row_scope

def run():
 b=.2;f=np.array([[1,b],[1,b],[1,-b],[1,-b],[-1,b],[-1,b],[-1,-b],[-1,-b]],dtype=np.float32);y=np.repeat([0,1],4);rx=np.tile([1,1,3,3],2);day=np.tile([1,2,1,2],2);saved=f.copy();r=describe(f,y,rx,day,['A','B']);expected=2*b*b/(1+b*b);margin=2/(1+b*b)
 assert all(abs(v['RX_centroid_spread']['mean_one_minus_cosine']-expected)<1e-7 and abs(v['out_RX_margin']['mean']-margin)<1e-7 for v in r['per_TX']);assert np.array_equal(f,saved)
 asym=f.copy();asym[4:,1]*=2;ar=describe(asym,y,rx,day,['A','B']);rr=describe(asym,1-y,rx,day,['B','A']);assert abs(ar['per_TX'][0]['RX_centroid_spread']['mean_one_minus_cosine']-rr['per_TX'][1]['RX_centroid_spread']['mean_one_minus_cosine'])<1e-12;assert abs(ar['per_TX'][1]['RX_centroid_spread']['mean_one_minus_cosine']-rr['per_TX'][0]['RX_centroid_spread']['mean_one_minus_cosine'])<1e-12;assert rr['per_TX'][0]['TX']=='B'
 bad=0
 for role in ('L_s','U_s','clean','query'):
  try:source_scope(role)
  except ValueError:bad+=1
  else:raise AssertionError('Forbidden role accepted')
 try:row_scope('BOTH-s2026092701')
 except ValueError:bad+=1
 else:raise AssertionError('Reference row selection changed')
 try:describe(f,np.full(8,-1),rx,day,['A','B'])
 except ValueError:bad+=1
 else:raise AssertionError('Hidden labels accepted')
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);(root/'release_commit.txt').write_text('wrong\n',encoding='utf-8')
  try:reference_scope(root,root)
  except ValueError:bad+=1
  else:raise AssertionError('Wrong reference accepted')
 return dict(status='PASS',known_angle_oracle=True,input_features_unchanged=True,class_permutation_equivariant=True,protocol_negative_cases=bad,IQ_reads=0,checkpoint_reads=0,target_reads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
