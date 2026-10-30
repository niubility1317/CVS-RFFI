"""One independent CPU-only scoring recovery of existing frozen predictions."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .star_representation_launch import _read,_write,_env
from . import star_binary_graph_run as producer
from . import star_binary_graph_score as scorer

RUN='20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01'
PREDICTION_COMMIT='bb9f5823f53bdbba6dd3e65f9f58dfe70bf86cdb'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','predictions_root','U_truth','Q_truth','targets_path'}


def validate(r):
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
        or any(not isinstance(r[k],str) or not r[k].strip() for k in FIELDS)
        or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None):
        raise ValueError('Only the fixed independent binary-graph scoring recovery is allowed')
    root=Path(r['run_root']);pred=Path(r['predictions_root'])
    if root.name!=RUN or pred!=root.parent/producer.RUN_ID/'predict':
        raise ValueError('New recovery root must refer to the original immutable predictions')
    if any(Path(r[k]).resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(Path(r[k]).resolve()) for k in ('predictions_root','U_truth','Q_truth','targets_path')):
        raise ValueError('Recovery outputs cannot replace private or original prediction inputs')
    m=_read(pred/'complete.json')
    if m.get('run_id')!=producer.RUN_ID or m.get('code_commit')!=PREDICTION_COMMIT or m.get('status')!=producer.STATUS:
        raise ValueError('Only the already-complete original four-state predictions may be scored')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Preserve the existing recovery; never overwrite or resubmit')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();_write(root/'configs/recovery.json',r)
    command=[r['python'],'-B','-u','-m',__package__+'.star_binary_graph_score_recovery','driver','--runtime',str(Path(runtime).resolve())]
    with (root/'logs/score.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Existing recovery state must be preserved')
    state=dict(run_id=RUN,code_commit=r['code_commit'],prediction_run_id=producer.RUN_ID,prediction_code_commit=PREDICTION_COMMIT,
        pid=os.getpid(),status='RUNNING',started_at=time.time(),GPU_workers=0,neural_updates=0,predictions_regenerated=False)
    _write(path,state)
    try:
        result=scorer.score(r['predictions_root'],r['U_truth'],r['Q_truth'],r['targets_path'],root/'score')
        if result['run_id']!=producer.RUN_ID or result['code_commit']!=PREDICTION_COMMIT or result['status']!=scorer.STATUS:
            raise ValueError('Recovered scores must retain original prediction provenance')
        state.update(status='ANALYZED_DEVELOPMENT',primary_state=result['primary_state'],primary_pass=result['primary_pass'],
            next_step=result['next_step'],truth_last=result['truth_last'],formal_target_reads=0,final_phase3_complete=False)
    except Exception as exc:
        state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(json.dumps(state),flush=True)


def main():
    print(json.dumps(dict(event='STAR_BINARY_GRAPH_SCORE_RECOVERY_PROCESS',pid=os.getpid(),cwd=os.getcwd())),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['start','driver']);p.add_argument('--runtime',required=True)
    a=p.parse_args();return dict(start=start,driver=driver)[a.mode](a.runtime)


if __name__=='__main__':main()
