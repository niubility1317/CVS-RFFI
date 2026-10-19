"""Small protocol-negative and fixed-stream/source-decision checks, with zero NN."""
import json
from unittest.mock import patch
from . import design as d,report
from .bank import sequence
from .data import load

def run():
 import numpy as np
 a=sequence();assert a.shape==(2220,128) and np.array_equal(a,sequence()) and set(a.ravel().tolist())==set(range(6300))
 counts=np.bincount(a.ravel(),minlength=6300);assert counts.min()==45 and counts.max()==46
 assert all(len(set(a.ravel()[i:i+6300].tolist()))==6300 for i in range(0,45*6300,6300))
 negatives=0
 for role in ('U','target','query'):
  try:load(role)
  except ValueError:negatives+=1
  else:raise AssertionError('Non-L fitting interface accepted')
 values={}
 for row in d.rows():
  urban=.7 if row['As'] else .68
  if row['style']=='TEMP' and row['As']:urban=.71
  values[row['arm']]=dict(source_accuracies=[[urban]*3]*2,full_V_clean_accuracy=.975)
 def read(p):
  arm=next(r['arm'] for r in d.rows() if r['row_id'] in str(p));return values[arm]
 saved=[]
 with patch.object(d,'read',read),patch.object(d,'write',lambda p,v:saved.append(v)):
  result=report.source_decision();assert result['preferred']=='TEMP' and not result['formal_scores_read'] and len(saved)==1
  for v in values.values():v['source_accuracies']=[[.6]*3]*2
  result=report.source_decision();assert result['preferred'] is None and not result['eligible_source_order'] and not result['automatic_extension']
 return dict(status='PASS',shared_sequence_updates=2220,physical_L_slots=284160,full_L_cycles=45,source_order_and_failure_verified=True,protocol_negatives=negatives,NN=0,checkpoint_loads=0,real_IQ=0,query_read=False)

if __name__=='__main__':print(json.dumps(run()))
