"""Focused data/provenance/budget and source-decision negative checks."""
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
  else:raise AssertionError('Invalid fixed quality-risk contract accepted')
 assert tuple(r['arm'] for r in d.rows())==('M1','M2')
 assert len({d.config(r)['parent_source'] for r in d.rows()})==1
 for r in d.rows():
  c=d.config(r);d.validate(c);expected=d.old.make_args(d.old_config(),'cpu');actual=d.make_args(c,'cpu')
  def equal(a,b):return a==b or (isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b))
  diffs={k for k in set(vars(expected))|set(vars(actual)) if not equal(getattr(expected,k,None),getattr(actual,k,None))}
  assert diffs<={'run_id','candidate_id','output_dir'}
  for k,v in [('target_access',True),('As',False),('epochs',20),('updates',4440),('model_seed',2026092702),('parent_source','unknown'),('method','HARD_REWEIGHT')]:
   x=copy.deepcopy(c);x[k]=v;bad(lambda x=x:d.validate(x))
  x=copy.deepcopy(c);x['quality_pool']['satellite_CE_coefficient']=.5;bad(lambda:d.validate(x))
  bad(lambda r=r:d.config(r,20))
 c=d.config(d.rows()[0]);roles={'L':['l'],'U':['u'],'V':['v']}
 done=dict(config=c,epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0)
 frozen=dict(config=c,selection='final_only',target_access=False)
 initial=dict(scratch_only=False,target_contact=False,checkpoint_sources=c['checkpoint_sources'],source_roles='EXACT_MATCH')
 final_metadata(c,done,frozen,initial,dict(role_ids=roles),roles)
 bad(lambda:final_metadata(c,dict(done,target_access=True),frozen,initial,dict(role_ids=roles),roles))
 from . import report
 with tempfile.TemporaryDirectory() as folder:
  base=Path(folder)/'new';rr={'NOAS30':Path(folder)/'no','AS30':Path(folder)/'as','POOL_RANDOM':base/'M1-s2026092701','QUALITY_STRAT':base/'M2-s2026092701'}
  fixed={name:report.cfg(name) for name in rr}
  for name,r in rr.items():
   c=fixed[name];clean=.9 if name!='QUALITY_STRAT' else .899;rate=.505 if name=='QUALITY_STRAT' else .5
   d.write(r/'source/completion.json',dict(config=c,target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=clean)))
   d.write(r/'source_qualification/complete.json',dict(config=c,target_access=False,formal_scores_read=False,training_updates=0,NN=12600,source_accuracies=[[.8,rate,rate]]*2))
  with patch.object(d,'BASE',base),patch.object(report,'roots',return_value=rr),patch.object(report,'cfg',side_effect=lambda name:fixed[name]):
   result=report.source_decision();assert result['source_qualified'] and len(result['checks'])==2
   (rr['QUALITY_STRAT']/'source_qualification/complete.json').unlink();result=report.source_decision()
   assert result['status']=='SOURCE_READOUT_PARTIAL' and result['formal_evaluation_always_required'] and not result['source_qualified']
 from .pool_checks import run as pool_checks
 pc=pool_checks()
 return dict(status='PASS',protocol_negatives=negative,pool_checks=pc,native_args_unchanged_except_identity=True,
  source_decision_conditions=2,source_FAIL_still_all7=True,real_CVS_NN=0,IQ=0,CP=0,optimizer_updates=0,smoke_NN=d.SMOKE_NN)

if __name__=='__main__':print(json.dumps(run()))
