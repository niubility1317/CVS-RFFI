"""Focused fixed-scope and checkpoint protocol negatives; zero CVS forwards."""
import copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
from . import design as d
from .protocol import final_metadata,checkpoint

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid fixed satellite CE64 scope accepted')
 assert tuple(r['arm'] for r in d.rows())==('CE_HALF','CE_T2')
 for row in d.rows():
  c=d.config(row);d.validate(c)
  for k,v in [('target_access',True),('As',False),('new_Ag',True),('epochs',20),('updates',4440),('model_seed',2026092702),('parent_source','unknown'),('label_epochs',13),('Ag_mode','WHITE')]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  for key,val in [('temperature',3.),('weight',.75),('smoothing',0.),('T_squared',True)]:
   z=copy.deepcopy(c);z['ce_objective'][key]=val;bad(lambda z=z:d.validate(z))
  bad(lambda:d.config(row,20))
 c=d.config(d.rows()[0]);bad(lambda:d.config(dict(arm='other',row_id='x',model_seed=d.SEED)))
 assert d.config(d.rows()[0])['parent_source']==d.config(d.rows()[1])['parent_source']
 roles={'L':['toyL'],'U':['toyU'],'V':['toyV']};done=dict(config=c,epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0);frozen=dict(config=c,selection='final_only',target_access=False);init=dict(scratch_only=False,target_contact=False,checkpoint_sources=c['checkpoint_sources'],source_roles='EXACT_MATCH');contract=dict(role_ids=roles)
 final_metadata(c,done,frozen,init,contract,roles)
 for key,val in [('target_access',True),('epoch',20),('optimizer_steps',4440),('head_fit_updates',1)]:
  z=copy.deepcopy(done);z[key]=val;bad(lambda z=z:final_metadata(c,z,frozen,init,contract,roles))
 cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=d.RUN,candidate_id=c['row_id'],checkpoint_selection='final_only',ema_model={},args=dict(baseline_ckpt=c['parent_source']+'/final_ssdg.pth',teacher_ckpt=''))
 checkpoint(c,cp)
 for key,val in [('run_id',d.REFERENCE_RUN),('candidate_id','B01-s2026092701'),('epoch',20),('ema_model',None)]:
  z=copy.deepcopy(cp);z[key]=val;bad(lambda z=z:checkpoint(c,z))
 from .objective_checks import run as objective_checks
 objective=objective_checks()
 from . import report
 with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
  base=Path(tmp)/'new';ref=Path(tmp)/'runs'/d.REFERENCE_RUN
  no=dict(target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.9))
  old=dict(config=d.old_config(),target_access=False,formal_scores_read=False,training_updates=0,NN=12600,source_accuracies=[[.9,.74,.60]]*2)
  d.write(ref/'B00-s2026092701/source/completion.json',no);d.write(ref/'B01-s2026092701/source_qualification/complete.json',old)
  configs={}
  for row,rates in zip(d.rows(),[[[.9,.745,.60]]*2,[[.9,.76,.61]]*2]):
   c=d.config(row);yes=dict(config=c,**no);yes['final_source_metrics']=dict(source_val_count=27000,source_val_accuracy=.899)
   configs[row['row_id']]=c
   q=dict(config=c,target_access=False,formal_scores_read=False,training_updates=0,NN=12600,source_accuracies=rates)
   d.write(base/row['row_id']/'source/completion.json',yes);d.write(base/row['row_id']/'source_qualification/complete.json',q)
  with patch.object(d,'BASE',base),patch.object(d,'PROJECT',tmp),patch.object(d,'config',side_effect=lambda row:configs[row['row_id']]):
   v=report.source_decision();assert v['source_qualified'] is True and all(type(x) is bool for x in v['checks'].values());json.dumps(v,allow_nan=False)
   (base/d.rows()[1]['row_id']/'source_qualification/complete.json').unlink();v=report.source_decision();assert v['status']=='SOURCE_READOUT_PARTIAL' and v['source_qualified'] is False and v['formal_evaluation_always_required'] is True
 return dict(status='PASS',protocol_negatives=negative,objective_checks=objective,source_bool_JSON=True,both_rows_required=True,CVS_NN=0,IQ=0,CP=0,target=False,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
