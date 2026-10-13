"""Synthetic completed metrics exercise aggregation only; never read IQ/truth."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import design as d
from . import score as s
from experiments.cvs_crossrx_contrastive_clean import predict as parent
from experiments.cvs_ensemble_recovery import predict as ce_predict,design as ce

def records(c,readout):
 cm=[[28000 if i==j else 0 for j in range(6)] for i in range(6)]
 return [dict(arm=c['arm'],model_seed=c['model_seed'],readout=readout,dimension='overall' if i==0 else 'fixture'+str(i),accuracy=1.,macro_f1=1.,query_count=168000,confusion=cm,view='clean') for i in range(18)]

def run(expect_missing=False):
 ids=np.array(['opaque-a','opaque-b']);values={}
 for row in d.rows():
  c=d.config(row);values[(d.BASE/row['row_id']/'completion.json').as_posix()]=dict(status='TRAINED_PREDICTED_SCORED',config=c,views=['clean'],readouts=['ENSEMBLE'],count_per_view=168000,metric_records=18,truth_last=True,independent_recount='VERIFIED')
  values[(d.BASE/row['row_id']/'scores/metrics.json').as_posix()]=dict(records=records(c,'ENSEMBLE'),truth_last=True,independent_recount='VERIFIED')
  pc=d.parent_config(c);values[(Path(pc['output']).parent/'scores/metrics.json').as_posix()]=dict(records=records(pc,'FINAL'))
 for row in ce.rows():
  c=ce.config(row);values[(ce.BASE/row['row_id']/'scores/metrics.json').as_posix()]=dict(records=records(c,'ENSEMBLE_HEAD'))
 def read(path):return values[Path(path).as_posix()]
 def own(row):return d.config(next(r for r in d.rows() if r['row_id']==row)),ids
 artifacts=d.ROOT/'local_artifacts';artifacts.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='ensemble_aggregate_',dir=artifacts) as temp:
  target=Path(temp).resolve()/'result';assert artifacts.resolve() in target.parents
  written={}
  with patch.object(d,'read',side_effect=read),patch.object(d,'write',side_effect=lambda path,value:written.update({Path(path).as_posix():value})),patch.object(parent,'preflight',return_value=(None,ids)),patch.object(ce_predict,'preflight',return_value=(None,ids)):
   if expect_missing:
    try:s.finish()
    except NameError as exc:
     if 'preflight' not in str(exc):raise
     return dict(status='EXPECTED_FAILURE',layer='aggregation',error=str(exc),IQ_or_truth_read=False)
    raise AssertionError('Missing-import defect did not reproduce')
   with patch.object(s,'preflight',side_effect=own):
    result=s.finish(output=target,run_id='synthetic-aggregate')
    if len(result['summary'])!=2 or any(x['accuracy_mean']!=1 for x in result['summary']) or not all(target.as_posix() in k for k in written):raise AssertionError('Aggregate/outroot routing differs')
    try:s.finish(output=target,run_id='synthetic-aggregate')
    except FileExistsError:pass
    else:raise AssertionError('Existing recovery output accepted')
  return dict(status='PASS',synthetic_groups=6,paired_comparisons=len(result['paired']),new_output_only=True,overwrite_rejected=True,IQ_or_truth_read=False)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--expect-missing',action='store_true');a=p.parse_args();print(json.dumps(run(a.expect_missing)))
