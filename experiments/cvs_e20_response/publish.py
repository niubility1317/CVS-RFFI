"""Immutable E20+20 identity release with rowwise clean truth-last scoring."""
import argparse,json,subprocess,sys
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
workers=[dict(r,process=process(r['pid']),resolved=read(base/r['row_id']/'source/resolved_config.json'),completion=read(base/r['row_id']/'completion.json')) for r in launch['rows']]
actual=process(submit['pid']) if submit else None
valid=bool(actual and actual['cwd']==str(release) and actual['argv']==submit['argv'])
if done:
 if done['status']!='EXPERIMENT_COMPLETE' or len(done['rows'])!=21:raise ValueError('Diagnostic completion differs')
 valid=all(w['completion'] and w['completion']['status']=='TRAINED_AND_SCORED' for w in workers) and len(workers)==21
print(json.dumps(dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=actual,workers=workers,queue=read(base/'queue_state.json'),completion=done,failure=failure,preflight=read(release/'preflight.json'),read_at=time.time())))
'''

def configure():
    from experiments.cvs_finite_action.publish import configure as parent
    parent();transport.d=d
    transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_e20_response']))
    transport.REMOTE=transport.REMOTE.replace('experiments.cvs_finite_action.preflight','experiments.cvs_e20_response.preflight').replace('experiments.cvs_finite_action.dispatch','experiments.cvs_e20_response.dispatch')
    transport.INSPECT=INSPECT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
    value=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
    print(json.dumps(dict(status=value['status'],complete=value.get('completion'),workers=len(value['workers']),failure=value['failure'])))
