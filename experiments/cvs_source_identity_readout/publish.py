"""Reuse verified immutable transport; bind a distinct source-only release."""
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
workers=[dict(r,process=process(r['pid']),resolved=read(base/r['row_id']/'resolved_config.json'),completion=read(base/r['row_id']/'completion.json')) for r in launch['rows']]
actual=process(submit['pid']) if submit else None
valid=bool(actual and actual['cwd']==str(release) and actual['argv']==submit['argv'])
if done:
 if done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or len(done['rows'])!=3 or done['identity_updates'] or done['target_access']:raise ValueError('Diagnostic completion differs')
 valid=all(w['completion'] and w['completion']['status']=='SOURCE_DIAGNOSTIC_COMPLETE' for w in workers) and len(workers)==3
print(json.dumps(dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=actual,workers=workers,queue=read(base/'queue_state.json'),completion=done,failure=failure,preflight=read(release/'preflight.json'),read_at=time.time())))
'''

def configure():
    from experiments.cvs_scratch_energy_sensitivity.publish import configure as parent_configure
    parent_configure()
    transport.d=d
    transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_source_conditional_geometry','experiments/cvs_source_identity_readout']))
    transport.REMOTE=transport.REMOTE.replace('experiments.cvs_scratch_energy_sensitivity.preflight','experiments.cvs_source_identity_readout.preflight').replace('experiments.cvs_scratch_energy_sensitivity.dispatch','experiments.cvs_source_identity_readout.dispatch')
    transport.INSPECT=INSPECT


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
    value=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
    print(json.dumps(dict(status=value['status'],complete=value.get('completion'),workers=len(value['workers']),failure=value['failure'])))
