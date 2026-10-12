import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport

INSPECT=r'''
import json,sys,time
from pathlib import Path
c=CONFIG;release=Path(c['project'])/'releases'/c['release'];base=Path(c['project'])/'runs'/c['run']
sys.path.insert(0,str(release));sys.path.insert(0,str(release/'code'))
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_multi_state_action.dispatch import capacity
def read(p):return json.loads(p.read_text()) if p.is_file() else None
submit=read(release/'submit.json');actual=process(submit['pid']) if submit else None;failure=read(base/'failure.json');done=read(base/'completion.json');launch=read(base/'launch.json') or {'rows':[]}
workers=[]
for r in launch['rows']:
 p=base/r['row_id'];logs=p.parent/'logs'/(r['row_id']+'.log')
 epochs=[len((p/a/'epoch_metrics.jsonl').read_text().splitlines()) if (p/a/'epoch_metrics.jsonl').is_file() else 0 for a in ('IQ','RAW_REL','COMPLEX_REL','GL2_REL')]
 workers.append(dict(r,process=process(r['pid']),resolved=read(p/'resolved_config.json'),completion=read(p/'completion.json'),epoch_counts=epochs,log_bytes=logs.stat().st_size if logs.is_file() else 0))
valid=bool(actual and actual['cwd']==str(release) and actual['argv']==submit['argv'])
if done:
 from experiments.cvs_wave_geometry import design as d
 from experiments.cvs_wave_geometry.summarize import analyze
 verified=analyze(base)
 if done['head_fits']!=20 or done['head_updates']!=16000 or len(done['rows'])!=5 or done['target_access']:raise ValueError('Incomplete source pilot')
 valid=verified['status']=='VERIFIED'
print(json.dumps(dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=actual,workers=workers,queue=read(base/'queue_state.json'),completion=done,failure=failure,preflight=read(release/'preflight.json'),source_decision=read(base/'source_decision.json'),capacity={str(g):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()},read_at=time.time())))
'''

def configure():
 from experiments.cvs_nocfo_foundation.publish import configure as native
 native();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_wave_geometry','experiments/cvs_nocfo_engineering']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_nocfo_foundation.preflight','experiments.cvs_wave_geometry.preflight').replace('experiments.cvs_nocfo_foundation.dispatch','experiments.cvs_wave_geometry.dispatch');transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),epochs=[r['epoch_counts'] for r in v['workers']],completion=v['completion'],failure=v['failure'],decision=v['source_decision'])))
