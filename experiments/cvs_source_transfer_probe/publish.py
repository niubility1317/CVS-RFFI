import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
INSPECT=r'''
import json,sys,time
from pathlib import Path
c=CONFIG;release=Path(c['project'])/'releases'/c['release'];base=Path(c['project'])/'runs'/c['run']
sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_source_transfer_probe import design as d
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_multi_state_action.dispatch import capacity
def read(p):return json.loads(p.read_text()) if p.is_file() else None
submit=read(release/'submit.json');actual=process(submit['pid']) if submit else None;queue=read(base/'queue_state.json');failure=read(base/'failure.json') or (dict(status='FAILED',rows=queue['failures']) if queue and queue.get('failures') else None);done=read(base/'completion.json');launch=read(base/'launch.json') or {'rows':[]};workers=[]
for r in launch['rows']:
 log=Path(r['log']);workers.append(dict(r,process=process(r['pid']),resolved=read(base/r['row_id']/'resolved_config.json'),completion=read(base/r['row_id']/'completion.json'),log_bytes=log.stat().st_size if log.is_file() else 0))
valid=bool(actual and actual['cwd']==str(release) and actual['argv']==submit['argv'])
if done:
 valid=done['status']=='SOURCE_DIAGNOSTIC_COMPLETE' and done['models']==9 and not done['target_access'] and all(w['completion'] and w['completion']['status']=='SOURCE_DIAGNOSTIC_COMPLETE' for w in workers) and len(workers)==9
print(json.dumps(dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=actual,workers=workers,queue=queue,completion=done,failure=failure,preflight=read(release/'preflight.json'),capacity={str(g):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()},read_at=time.time())))
'''

def configure():
 from experiments.cvs_nocfo_foundation.publish import configure as native
 native();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_frequency_context50','experiments/cvs_source_transfer_probe','experiments/cvs_transfer_features','experiments/cvs_legacy_no_sat','experiments/cvs_phase1_repair','experiments/cvs_e20_foundation','experiments/cvs_nocfo_engineering','experiments/cvs_finite_margin','experiments/cvs_finite_action','experiments/cvs_wave_geometry','experiments/cvs_scratch30_baseline','experiments/cvs_feature_challenge']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_nocfo_foundation.preflight','experiments.cvs_source_transfer_probe.preflight').replace('experiments.cvs_nocfo_foundation.dispatch','experiments.cvs_source_transfer_probe.dispatch');transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),completion=v['completion'],failure=v['failure'])))
