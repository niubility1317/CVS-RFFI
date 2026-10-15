"""Source-decision upload ordering, including a scientifically failed first seed."""
import json,os,tempfile
from pathlib import Path
from . import design as d,gate,evaluate

def run():
 saved=d.BASE;old_cuda=os.environ.get('CUDA_VISIBLE_DEVICES');cases=0
 try:
  os.environ['CUDA_VISIBLE_DEVICES']=''
  with tempfile.TemporaryDirectory() as tmp:
   d.BASE=Path(tmp)
   for passed in (False,True):
    q=dict(qualified=passed);p=d.BASE/'seed_expansion_release.json'
    if p.exists():p.unlink()
    assert not gate.decision_uploaded(q) and not gate.expansion_released(q);cases+=1
    d.write(d.BASE/'source_qualification.json',q)
    try:evaluate.finish()
    except ValueError as e:assert 'before any new target summary' in str(e);cases+=1
    else:raise AssertionError('Premature aggregate read accepted')
    v=dict(status='EXPANSION_RELEASED',source_only=True,new_target_summary_read=False,qualified=passed,first_seed=d.SEEDS[0],commit='fixed-code',remote_OID='fixed-code');d.write(p,v)
    assert gate.decision_uploaded(q) and gate.expansion_released(q)==passed;cases+=1
    v['new_target_summary_read']=True;d.write(p,v)
    try:gate.decision_uploaded(q)
    except ValueError:cases+=1
    else:raise AssertionError('Target-informed expansion accepted')
  return dict(status='PASS',cases=cases,PASS_and_FAIL_upload_required=True,failed_source_never_expands=True,aggregation_rejected_before_score_read=True)
 finally:
  d.BASE=saved
  if old_cuda is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=old_cuda
if __name__=='__main__':print(json.dumps(run()))
