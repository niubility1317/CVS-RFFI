"""Validate, but do not execute, the registered N607 source-V launch."""
import json
import posixpath
import re
import shlex
import sys
from pathlib import Path

PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
ENV_NAMES={'CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH'}


def validate(spec):
    execution=spec['execution'];row=spec['rows'][0];code=spec['code']
    if len(spec['rows'])!=1 or row['command']!=execution['launch_command']:
        raise ValueError('one row and identical registered commands required')
    runid=spec['run_id'];commit=code['commit']
    if not re.fullmatch(r'[A-Za-z0-9._-]+',runid) or not re.fullmatch(r'[0-9a-f]{40}',commit):
        raise ValueError('invalid registered run or commit')
    cwd=code['cwd'];run=execution['remote_run_root'];log=execution['remote_log_root']
    if not cwd.startswith(PROJECT+'/releases/') or cwd!=posixpath.normpath(cwd):
        raise ValueError('remote working directory must be a pinned N607 release')
    if run!=PROJECT+'/runs/'+runid or log!=PROJECT+'/logs/'+runid+'.log':
        raise ValueError('run and log must use the registered isolated paths')
    tokens=shlex.split(execution['launch_command']);environment={}
    while tokens and '=' in tokens[0] and not tokens[0].startswith('/'):
        key,value=tokens.pop(0).split('=',1)
        if key not in ENV_NAMES or key in environment:raise ValueError('unexpected environment setting')
        environment[key]=value
    if set(environment)!=ENV_NAMES or environment['CUDA_VISIBLE_DEVICES']!='':
        raise ValueError('registered CPU environment required')
    if any(not p.startswith(PROJECT+'/releases/') for p in environment['PYTHONPATH'].split(':')):
        raise ValueError('PYTHONPATH must contain pinned N607 releases')
    entry=PROJECT+'/releases/'+runid+'-release/experiments/phase1_v_restoration_diagnostic/phase1_source_v_pipeline.py'
    expected=[PYTHON,'-X','utf8','-u',entry,'--root',PROJECT,'--run-root',run,'--commit',commit]
    if tokens!=expected:raise ValueError('registered executable and arguments do not match the isolated N607 run')
    return dict(cwd=cwd,environment=environment,argv=tokens,run=run,log=log,commit=commit)


if __name__=='__main__':
    with Path(sys.argv[1]).open(encoding='utf8') as f:spec=json.load(f)
    print(json.dumps(validate(spec),ensure_ascii=False,indent=2))
