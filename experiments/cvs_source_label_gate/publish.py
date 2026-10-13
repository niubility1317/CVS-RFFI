import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_time_factorial50 import publish as parent
from experiments.cvs_report_ssl import publish as transport
INSPECT=r'''
import json,sys,time
from pathlib import Path
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];base=p/'runs'/c['run'];sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_multi_state_action.dispatch import capacity
def read(path):return json.loads(path.read_text()) if path.is_file() else None
submit=read(release/'submit.json');result=dict(run_id=c['run'],read_at=time.time(),submit=submit,dispatcher_process=process(submit['pid']) if submit else None,preflight=read(release/'preflight.json'),failure=read(base/'failure.json'),completion=read(base/'completion.json'),workers=[])
for item in (read(base/'launch.json') or dict(rows=[]))['rows']:
 log=Path(item['log']);summary=read(base/item['row_id']/'summary.json');result['workers'].append(dict(item,process=process(item['pid']),summary=summary,log_size=log.stat().st_size if log.is_file() else 0,log_tail=log.read_text(errors='replace')[-2000:] if log.is_file() else ''))
result['capacity']={str(g):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()}
actual=result['dispatcher_process'];result['status']='VERIFIED' if actual and actual['cwd']==str(release) and actual['argv']==submit['argv'] and not result['failure'] else 'FAILED' if result['failure'] else 'UNKNOWN'
if result['completion'] and not result['failure']:
 done=result['completion']
 if done['status']!='SOURCE_DIAGNOSTICS_COMPLETE' or done['rows']!=3 or len(result['workers'])!=3 or done['source_L_packets']!=18900 or done['head_fit_updates'] or done['target_access'] or any(not w['summary'] or w['summary']['status']!='SOURCE_GATE_DIAGNOSTIC_COMPLETE' for w in result['workers']):raise ValueError('Incomplete source proxy')
 result['status']='VERIFIED'
print(json.dumps(result))
'''
def configure():
 parent.configure();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_source_label_gate']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_time_factorial50.preflight','experiments.cvs_source_label_gate.preflight').replace('experiments.cvs_time_factorial50.dispatch','experiments.cvs_source_label_gate.dispatch');transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output);print(json.dumps(dict(status=v['status'],completion=v['completion'],failure=v['failure'],workers=len(v['workers']))))
