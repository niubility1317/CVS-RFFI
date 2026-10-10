"""Focused recovery routing checks; no target input is opened."""
import json,tempfile,copy
from pathlib import Path
from unittest.mock import patch
from . import recovery as r

def checks():
 r.configure();d=r.d
 assert len(r.ADOPT)==26 and len(d.rows())==48
 for row in d.rows():
  c=d.config(row,.05 if row['arm'] in d.SELECTED_ARMS else None);d.validate(c)
  assert c['run_id']==r.RUN and r.RUN in c['output_root']
  if row['row_id'] in r.ADOPT:
   old=copy.deepcopy(c['checkpoint_origin']['config']);expected=copy.deepcopy(old)
   expected.update(run_id=r.RUN,output_root=c['output_root'],checkpoint_origin=c['checkpoint_origin'])
   assert c==expected
  args=d.make_args(c,'cpu')
  assert not args.use_unlabeled and not args.use_ema_teacher and not args.use_concat_sat_channel_aug
  assert args.label_epochs==200 and args.pseudo_epochs==0 and args.from_scratch
 rid=next(iter(r.ADOPT));receipt=r.ADOPT[rid]['launch']
 with patch.object(r,'process',return_value=dict(receipt)):
  assert not r.old_ready(rid)
 with patch.object(r,'process',return_value=dict(receipt,cwd='/different')):
  try:r.old_ready(rid)
  except ValueError:pass
  else:raise AssertionError('PID identity mismatch accepted')
 with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OLD_BASE',Path(tmp)),patch.object(r,'process',return_value=None):
  d.write(Path(tmp)/rid/'failure.json',dict(error='other failure'))
  try:r.old_ready(rid)
  except ValueError:pass
  else:raise AssertionError('Other fault accepted')
  d.write(Path(tmp)/rid/'failure.json',dict(error=r.ERROR));assert r.old_ready(rid)
  (Path(tmp)/rid/'prediction').mkdir()
  try:r.old_ready(rid)
  except ValueError:pass
  else:raise AssertionError('Prior target access accepted')
 # Test evaluator routing without loading target or a checkpoint.
 with tempfile.TemporaryDirectory() as tmp,patch.object(d,'BASE',Path(tmp)),patch.object(r.e,'BASE',Path(tmp)):
  native=next(q for q in d.rows() if q['arm']=='native');c=d.config(native)
  d.write(d.BASE/native['row_id']/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,target_access=False))
  r.e.EVAL_ROW=native['row_id'];assert r.e.snapshot(d)==[c]
  r.e.EVAL_ROW='LTR_S_AF3-s'+str(d.SEEDS[0])
  try:r.e.snapshot(d)
  except FileNotFoundError:pass
  else:raise AssertionError('Candidate permitted before source selection')
  d.write(d.BASE/'source_selection.json',dict(status='SOURCE_SELECTION_FROZEN',selected_ratio=.05,target_scores_consumed=False))
  try:r.e.snapshot(d)
  except ValueError:pass
  else:raise AssertionError('Unselected candidate permitted')
  r.e.EVAL_ROW=None
 for ratio in (.03,.05,.1):assert len(d.test_rows(dict(status='SOURCE_SELECTION_FROZEN',selected_ratio=ratio,target_scores_consumed=False)))==36
 from .publish_recovery import REMOTE
 compile(REMOTE.replace('CONFIG','{}'),'recovery-publish','exec')
 return dict(status='PASS',models=48,reused=26,scratch=22,tested=36,live_workers_preserved=True,unknown_failure_rejected=True,source_selection_gate=True)

def preflight(output):
 import torch
 from .recover_source import smoke
 from .time_probe_checks import run
 torch.set_num_threads(2)
 result=checks();out=Path(output);out.mkdir(exist_ok=False)
 # Full original E200 evidence and actual trained checkpoint, zero IQ only.
 first=r.ADOPT['native-s2026092701']['config'];result['checkpoint']=smoke(first)
 # Probe regression uses the actual model class, with synthetic inputs only.
 result['probe']=run('cpu')
 r.d.write(out/'completion.json',result)
 print(json.dumps(result))

if __name__=='__main__':print(json.dumps(checks()))
