"""One source-only release; root owns launch and immutable output."""
import argparse
import json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_source_pseudotarget_health.publish import REMOTE as HEALTH_REMOTE

REMOTE = HEALTH_REMOTE.replace('experiments.cvs_source_pseudotarget_health.run',
                               'experiments.cvs_six_direction_mechanisms.run').replace(
                               "'--remote-preflight'", "'--preflight'")

INSPECT = r'''
import sys,json,time
from pathlib import Path
c=CONFIG;release=Path(c['project'])/'releases'/c['release'];base=Path(c['project'])/'runs'/c['run'];sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_six_direction_mechanisms import design as d
def read(p):return json.loads(p.read_text()) if p.is_file() else None
submit=read(release/'submit.json');actual=process(submit['pid']) if submit else None;done=read(base/'completion.json');failure=read(base/'failure.json');log=release/'dispatcher.stdout.log';rows=[]
for row in d.rows():
 p=base/row['row_id'];rows.append(dict(row,model_contract=read(p/'model_contract.json'),raw_exists=(p/'raw_source.pt').is_file(),directions={k:read(p/(k+'.json')) for k in d.DIRECTIONS},completion=read(p/'completion.json')))
valid=bool(actual and actual['cwd']==str(release) and actual['argv']==submit['argv'] and actual['gpu']==str(submit['gpu']))
if done:
 if done.get('status')!='SOURCE_MECHANISMS_COMPLETE' or done.get('rows')!=3 or done.get('direction_cells')!=18 or done.get('identity_updates')!=0 or done.get('target_access') or done.get('U_truth_read') or done.get('V_updates') or not all(r['raw_exists'] and r['model_contract'] and r['completion'] and all(r['directions'].values()) for r in rows):raise ValueError('Six source mechanism matrix incomplete or unauthorized')
 valid=True
print(json.dumps(dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=actual,completion=done,failure=failure,rows=rows,analysis=read(base/'analysis.json'),log_bytes=log.stat().st_size if log.is_file() else 0,log_tail=log.read_text(errors='replace').splitlines()[-3:] if log.is_file() else [],preflight=read(release/'preflight.json'),capacity={str(g):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()},read_at=time.time())))
'''

def configure():
    from experiments.cvs_conditional_rx_adversary.publish import configure as parent
    parent();transport.d=d
    transport.PATHS=list(dict.fromkeys([*transport.PATHS,
        'experiments/cvs_repeated_rx_margin','experiments/cvs_source_pseudotarget_health',
        'experiments/cvs_conditional_rx_probe_health','experiments/cvs_six_direction_mechanisms']))
    transport.REMOTE=REMOTE;transport.INSPECT=INSPECT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
    print(json.dumps(dict(status=v['status'],completion=v['completion'],failure=v['failure'],
        completed_cells=sum(bool(value) for r in v['rows'] for value in r['directions'].values()),
        log_bytes=v['log_bytes'],log_tail=v['log_tail'])))
