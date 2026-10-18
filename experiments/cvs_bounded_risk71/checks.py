"""Fixed one-row scope/provenance/source rule, plus pure bounded-risk tests."""
import copy,json,math,tempfile
from pathlib import Path
from unittest.mock import patch
from . import design as d
from .protocol import final_metadata

def run(native_path=None):
 negative=0
 def bad(fn):
  nonlocal negative
  try:fn()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid bounded-risk contract accepted')
 assert len(d.rows())==1
 row=d.rows()[0];c=d.config(row);d.validate(c)
 assert c['parent_source']==d.old_config()['parent_source']
 assert c['parent_source']!=d.nominal_config()['output_root']
 expected=d.nominal.make_args(d.nominal_config(),'cpu');actual=d.make_args(c,'cpu')
 def equal(a,b):return a==b or (isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b))
 differences={k for k in set(vars(expected))|set(vars(actual)) if not equal(getattr(expected,k,None),getattr(actual,k,None))}
 assert differences<={'run_id','candidate_id','output_dir'}
 for k,v in [('target_access',True),('As',False),('epochs',20),('updates',4440),('model_seed',2026092702),('parent_source',d.nominal_config()['output_root']),('method','NOMINAL')]:
  changed=copy.deepcopy(c);changed[k]=v;bad(lambda changed=changed:d.validate(changed))
 for k,v in [('gamma',.5),('v',[.25]*4),('satellite_CE_coefficient',.375),('satellite_KL_coefficient',.375)]:
  changed=copy.deepcopy(c);changed['risk'][k]=v;bad(lambda changed=changed:d.validate(changed))
 for k,v in [('root','unknown'),('allocation_seed',1)]:
  changed=copy.deepcopy(c);changed['quality_pool'][k]=v;bad(lambda changed=changed:d.validate(changed))
 bad(lambda:d.config(row,20))
 roles={'L':['l'],'U':['u'],'V':['v']}
 done=dict(config=c,epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0)
 frozen=dict(config=c,selection='final_only',target_access=False)
 initial=dict(scratch_only=False,target_contact=False,checkpoint_sources=c['checkpoint_sources'],source_roles='EXACT_MATCH')
 final_metadata(c,done,frozen,initial,dict(role_ids=roles),roles)
 bad(lambda:final_metadata(c,dict(done,target_access=True),frozen,initial,dict(role_ids=roles),roles))
 from . import report
 with tempfile.TemporaryDirectory() as folder:
  base=Path(folder)/'new';names=('NOAS30','AS30','QUALITY_STRAT','BOUNDED_RISK');roots={name:Path(folder)/name for name in names};fixed={name:report.cfg(name) for name in names}
  for name,root in roots.items():
   clean=.899 if name=='BOUNDED_RISK' else .9;rate=.505 if name=='BOUNDED_RISK' else .5
   d.write(root/'source/completion.json',dict(config=fixed[name],target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=clean)))
   d.write(root/'source_qualification/complete.json',dict(config=fixed[name],target_access=False,formal_scores_read=False,training_updates=0,NN=12600,source_accuracies=[[.8,rate,rate]]*2))
  with patch.object(d,'BASE',base),patch.object(report,'roots',return_value=roots),patch.object(report,'cfg',side_effect=lambda name:fixed[name]):
   decision=report.source_decision();assert decision['source_qualified'] and len(decision['checks'])==2 and decision['candidate']=='BOUNDED_RISK'
   (roots['BOUNDED_RISK']/'source_qualification/complete.json').unlink();decision=report.source_decision();assert not decision['source_qualified'] and decision['formal_evaluation_always_required']
 from .risk_checks import run as risk_checks
 return dict(status='PASS',protocol_negatives=negative,risk_checks=risk_checks(),source_decision_conditions=2,
  native_args_unchanged_except_identity=True,reused_M2_not_retrained=True,real_CVS_NN=0,IQ=0,CP=0,optimizer_updates=0,smoke_NN=d.SMOKE_NN)
if __name__=='__main__':print(json.dumps(run()))
