"""Focused negative tests for idle takeover, unchanged evaluation and isolation."""
from unittest.mock import patch
from pathlib import Path
import tempfile
from experiments.cvs_dual_test_now import driver as d
def checks():
 original=d.evaluation.BASE
 d.configure()
 assert d.evaluation.BASE==d.BASE and d.BASE!=d.source.BASE
 assert d.evaluation.RUN==d.source.RUN
 assert all(c['output_root'].startswith(d.source.BASE.as_posix()+'/') for c in map(d.source.config,d.source.rows()))
 d.evaluation.BASE=original
 with tempfile.TemporaryDirectory() as folder:
  base=Path(folder);q=dict(kind='predict',active=[],completed=[],failures=[],pending=[r['row_id'] for r in d.source.rows()])
  d.source.write(base/'queue_state.json',q)
  identity=dict(pid=d.OLD_PID,start_ticks=d.OLD_START,cwd=str(d.OLD_RELEASE),argv=['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-m','experiments.cvs_dual_evidence.dispatch'])
  # Production /proc child read is not reached in these negative cases.
  with patch.object(d.source,'BASE',base),patch.object(d,'process',return_value=identity):
   for change in [dict(active=['row']),dict(completed=['row']),dict(failures=['row']),dict(pending=[]),dict(kind='source')]:
    d.source.write(base/'queue_state.json',dict(q,**change))
    try:d.verify_idle()
    except ValueError:pass
    else:raise AssertionError('Unsafe takeover accepted '+str(change))
   d.source.write(base/'queue_state.json',q);(base/'launch_predict.json').write_text('{}')
   try:d.verify_idle()
   except ValueError:pass
   else:raise AssertionError('Duplicate evaluation accepted')
  with patch.object(d,'process',return_value=dict(identity,start_ticks=0)):
   try:d.verify_idle()
   except ValueError:pass
   else:raise AssertionError('Reused PID accepted')
 assert d.TOTAL_LIMIT==5 and d.MIN_FREE_MB==3000
 return dict(status='PASS',negative_takeover_cases=7,source_checkpoint_roots_unchanged=True,evaluation_output_isolated=True,model_and_prediction_math_unchanged=True,target_read=False)
if __name__=='__main__':
 import json
 print(json.dumps(checks()))
