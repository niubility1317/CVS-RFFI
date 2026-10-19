"""Focused synthetic mask/protocol negatives;no real IQ/checkpoint/NN."""
import copy
from . import design as d
from .protocol import cumulative,check_default_head
from .test_masking import run as mask_checks
def run():
 result=mask_checks();negative=0
 def bad(fn):
  nonlocal negative
  try:fn()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid protocol accepted')
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert c['As'] and c['Ag_mode']=='NONE' and c['parent_source']==d.reference_config()['parent_source'] and c['epochs']==10 and c['updates']==2220 and c['cumulative_updates']==6660
  assert c['crossview']['mode']=='ORD128' and c['masking']['mode']==row['arm'] and c['training_mechanism']['pair_weights']==[.5,.5,.25]
  for k,v in [('target_access',True),('As',False),('epochs',30),('updates',6660),('parent_source','P78'),('separate_head_training',True),('model_seed',2026092702)]:
   x=copy.deepcopy(c);x[k]=v;bad(lambda x=x:d.validate(x))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 bad(lambda:cumulative(1,223));bad(lambda:cumulative(10,2221));check_default_head((6,160));bad(lambda:check_default_head((6,64)))
 return dict(status='PASS',protocol_negatives=negative,masking=result,NN=0,checkpoint_loads=0,source_or_target_IQ=0)
