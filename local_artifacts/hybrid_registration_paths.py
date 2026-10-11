"""Correct display/artifact path separators only; never change training config."""
import json
import sys
import time
from pathlib import Path

release=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_stage2_hybrid_20261011_r01')
sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_stage2_hybrid import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock

with lock(d.BASE/'registration.lock'):
 spec=d.read(d.BASE/'experiment.json')
 if spec['run_id']!=d.RUN or spec['execution']['launch_owner']!=d.OWNER:raise ValueError('Wrong registration owner')
 for row in spec['rows']:
  expected=next(x for x in d.rows() if x['row_id']==row['row_id'])
  if row['config']!=d.config(expected):raise ValueError('Immutable training config differs')
  row['output_root']=str(d.BASE/row['row_id'])
  row['resolved_config_ref']=str(d.BASE/row['row_id']/'source/resolved_config.json')
  row['log_path']=str(d.BASE/'logs'/f"train-{row['row_id']}.log")
 spec['code']['cwd']=str(release)
 spec['execution']['run_root']=str(d.BASE);spec['execution']['release_root']=str(release)
 d.write(d.BASE/'experiment.json',spec)
 event=dict(at=time.time(),status='METADATA_PATH_NORMALIZED',change='Windows-rendered report path references normalized to Linux POSIX; immutable configs, models, queue, processes unchanged',target_access=False)
 with (d.BASE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
print(json.dumps(dict(status='VERIFIED',run_id=d.RUN,metadata_only=True,configs_unchanged=True)))
