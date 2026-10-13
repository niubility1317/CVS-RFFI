import copy,math,tempfile
from pathlib import Path
from unittest.mock import patch
from . import design as d
from . import dispatch as q

def checks():
 second=next(r for r in d.training_rows() if r['model_seed']==d.SEEDS[1])
 with tempfile.TemporaryDirectory() as temp,patch.object(d,'BASE',Path(temp)),patch.object(q,'source_proof',lambda c:Path(c['output_root'])):
  assert not q.eligible(second)
  first=[d.config(r) for r in d.training_rows() if r['model_seed']==d.SEEDS[0]]
  for c in first:
   root=Path(c['output_root']);root.mkdir(parents=True)
   d.write(root/'completion.json',dict(final_training_L_CE=.25));d.write(root/'source_frozen.json',{})
  center=next(c for c in first if c['arm']=='CENTER_25K');path=Path(center['output_root'])/'source_Sinc_epoch_diagnostic.json'
  records=[dict(epoch=e,Sinc=dict(kernel_relative_L2=.001)) for e in range(1,51)];d.write(path,dict(records=records))
  assert q.eligible(second)
  negatives=0
  for movement in [0.,float('nan')]:
   bad=copy.deepcopy(records);bad[-1]['Sinc']['kernel_relative_L2']=movement
   # Deliberately malformed negative metadata is written outside production helper.
   import json
   path.write_text(json.dumps(dict(records=bad)),encoding='utf8')
   try:q.eligible(second)
   except ValueError:negatives+=1
   else:raise AssertionError('Source mechanism threshold failed open')
  d.write(path,dict(records=records))
  d.write(Path(first[0]['output_root'])/'completion.json',dict(final_training_L_CE=.51))
  try:q.eligible(second)
  except ValueError:negatives+=1
  else:raise AssertionError('Source loss health failed open')
 return dict(status='PASS',source_only=True,no_IQ_truth_or_target_score_read=True,negative_checks=negatives)
