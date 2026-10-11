"""Immutable committed transport with CPU preflight and independent readback."""
import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport

INSPECT=r'''
import json,sys,time
from pathlib import Path
c=CONFIG;release=Path(c['project'])/'releases'/c['release'];base=Path(c['project'])/'runs'/c['run']
sys.path.insert(0,str(release))
from experiments.cvs_receiver_residual_capacity4.control import process
def read(p):return json.loads(p.read_text()) if p.is_file() else None
submit=read(release/'submit.json');launch=read(base/'launch.json') or {'rows':[]};done=read(base/'completion.json');failure=read(base/'failure.json')
workers=[dict(r,process=process(r['pid']),resolved=read(base/r['row_id']/'source/resolved_config.json'),source_completion=read(base/r['row_id']/'source/completion.json'),completion=read(base/r['row_id']/'completion.json')) for r in launch['rows']]
actual=process(submit['pid']) if submit else None
valid=bool(actual and actual['cwd']==str(release) and actual['argv']==submit['argv'])
if done:
 from experiments.cvs_ce_epoch21.dispatch import require_scored
 if done['status']!='EXPERIMENT_COMPLETE' or done['trained_models']!=3 or done['tested_models']!=3:raise ValueError('Incomplete matrix')
 require_scored(read(base/'scoring_complete.json'),3)
 valid=all(w['completion'] and w['completion']['status']=='TRAINED_AND_SCORED' for w in workers) and len(workers)==3
print(json.dumps(dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=actual,workers=workers,queue=read(base/'queue_state.json'),completion=done,failure=failure,preflight=read(release/'preflight.json'),read_at=time.time())))
'''


def configure():
    transport.d=d;transport.PATHS=[*transport.PATHS,'experiments/cvs_ce_epoch21']
    transport.REMOTE=transport.REMOTE.replace('experiments.cvs_report_ssl.control_checks','experiments.cvs_ce_epoch21.checks').replace('experiments.cvs_report_ssl.dispatch','experiments.cvs_ce_epoch21.dispatch')
    transport.INSPECT=INSPECT


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
    value=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
    print(json.dumps(dict(status=value['status'],complete=value.get('completion'),workers=len(value['workers']),failure=value['failure'])))
