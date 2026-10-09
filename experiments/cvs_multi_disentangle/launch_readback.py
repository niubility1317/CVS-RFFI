"""Independent post-state evidence for a partly queued fixed matrix."""
import json
from experiments.cvs_multi_disentangle import design as d
from experiments.cvs_multi_disentangle.monitor import ARTIFACT,REPORT,readback
from experiments.cvs_phase1_stack.publish import ssh


def capture():
    state=readback();release=d.PROJECT+'/releases/'+d.RELEASE
    inner='''
from pathlib import Path
import json,sys
sys.path[:0]=[RELEASE,RELEASE+'/code']
from experiments.cvs_multi_disentangle.dispatch import capacity
p=Path(BASE);rows=[]
for f in p.glob('*/source/epoch_metrics.jsonl'):
    lines=f.read_text().splitlines()
    if not lines:continue
    r=json.loads(lines[-1]);initial=json.loads((f.parent/'initialization.json').read_text())
    rows.append(dict(row_id=f.parents[1].name,initialization=initial,
        metrics={k:v for k,v in r.items() if k.startswith('multi_') or k in ('epoch','lr','epoch_time_s')}))
caps={g:dict(pids=sorted(c['pids']),free_mb=c['free_mb']) for g,c in capacity().items()}
leases={f.stem:json.loads(f.read_text()) for f in (p/'capacity_leases').glob('*.json')}
print(json.dumps(dict(rows=rows,capacity=caps,leases=leases)))
'''.replace('RELEASE',repr(release)).replace('BASE',repr(d.BASE.as_posix()))
    outer='import subprocess\nprint(subprocess.check_output('+repr(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',inner])+',text=True))'
    runtime=json.loads(ssh(outer));d.write(ARTIFACT/'runtime_telemetry_readback.json',runtime)
    configs={r['row_id']:d.config(r) for r in d.rows()}
    if state['failure'] or not state['controller'] or state['controller']['cwd']!=release:
        raise ValueError('Controller/launch evidence invalid')
    if state['controller']['state'] in ('Z','X'):
        raise ValueError('Controller not live')
    resolved=[];waiting=[];completed=[]
    for r in state['rows']:
        if r['row_id'] not in configs:raise ValueError('Unexpected launched row')
        actual=r['actual']
        if r.get('completion'):
            completed.append(r['row_id']);continue
        if not actual or actual['state'] in ('Z','X') or actual['cwd']!=release or actual['argv']!=r['argv'] or actual['start_ticks']!=r['start_ticks']:
            raise ValueError('Actual PID/cwd/argv differs')
        c=r['resolved']
        if c is not None:
            if any(c[k]!=v for k,v in configs[r['row_id']].items()) or c['commit']!=state['receipt']['commit'] or c['backend_flags']!=d.FULL_FP32_POLICY:
                raise ValueError('Actual runtime config differs')
            resolved.append(r['row_id'])
        else:
            lease=runtime['leases'].get(r['kind']+'-'+r['row_id'])
            if not lease or lease['status']!='WAITING_CAPACITY':raise ValueError('Unresolved worker lacks capacity wait evidence')
            waiting.append(r['row_id'])
    if any(len(c['pids'])>4 for c in runtime['capacity'].values()):raise ValueError('Capacity limit exceeded')
    for r in runtime['rows']:
        init=r['initialization']
        if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact']:
            raise ValueError('Initialization differs')
    evidence=dict(status='VERIFIED',source_commit=state['receipt']['commit'],controller=state['controller'],
        preregistered=len(configs),launched=len(state['rows']),resolved_rows=resolved,waiting_capacity_rows=waiting,
        queued_rows=(state.get('queue') or {}).get('pending',[]),completed_rows=completed,
        epoch_telemetry_rows=len(runtime['rows']),capacity=runtime['capacity'],
        source_training_complete=False,test_complete=False,performance_gain=None,
        auxiliary_mechanism_note='Warmup20; initial launch does not claim post-warmup mechanism execution.')
    d.write(REPORT/'evidence/launch_verified.json',evidence)
    d.write(REPORT/'evidence/launch_readback.json',state)
    d.write(REPORT/'evidence/runtime_telemetry_readback.json',runtime)
    return evidence


if __name__=='__main__':print(json.dumps(capture()))
