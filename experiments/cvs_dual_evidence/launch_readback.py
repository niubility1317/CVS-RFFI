"""Independent post-launch evidence; does not modify any remote process."""
import json
from experiments.cvs_dual_evidence import design as d
from experiments.cvs_dual_evidence.monitor import ARTIFACT,REPORT,readback
from experiments.cvs_phase1_stack.publish import ssh


def capture():
    state=readback();release=d.PROJECT+'/releases/'+d.RELEASE
    inner='''
from pathlib import Path
import json,sys
sys.path[:0]=[RELEASE,RELEASE+'/code']
from experiments.cvs_dual_evidence.dispatch import capacity
p=Path(BASE);rows=[]
for f in p.glob('*/source/epoch_metrics.jsonl'):
    lines=f.read_text().splitlines()
    if not lines:continue
    r=json.loads(lines[-1]);initial=json.loads((f.parent/'initialization.json').read_text())
    rows.append(dict(row_id=f.parents[1].name,initialization=initial,
        metrics={k:v for k,v in r.items() if k.startswith('dual_') or k in ('epoch','lr','epoch_time_s')}))
caps={g:dict(pids=sorted(c['pids']),free_mb=c['free_mb']) for g,c in capacity().items()}
leases={f.stem:json.loads(f.read_text()) for f in (p/'capacity_leases').glob('*.json')}
print(json.dumps(dict(rows=rows,capacity=caps,leases=leases)))
'''.replace('RELEASE',repr(release)).replace('BASE',repr(d.BASE.as_posix()))
    outer='import subprocess\nprint(subprocess.check_output('+repr(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',inner])+',text=True))'
    runtime=json.loads(ssh(outer));d.write(ARTIFACT/'runtime_telemetry_readback.json',runtime)
    configs={r['row_id']:d.config(r) for r in d.rows()}
    if state['failure'] or len(state['rows'])!=16 or not state['controller']:raise ValueError('Incomplete live launch evidence')
    resolved=[];waiting=[]
    for r in state['rows']:
        actual=r['actual']
        if not actual or actual['state'] in ('Z','X') or actual['cwd']!=release or actual['argv']!=r['argv'] or actual['start_ticks']!=r['start_ticks']:
            raise ValueError('Actual PID/cwd/argv differs')
        c=r['resolved']
        if c is not None:
            if any(c[k]!=v for k,v in configs[r['row_id']].items()) or c['commit']!=state['receipt']['commit'] or c['backend_flags']!=d.FULL_FP32_POLICY:
                raise ValueError('Actual runtime config differs')
            resolved.append(r['row_id'])
        else:
            lease=runtime['leases'].get('source-'+r['row_id'])
            if not lease or lease['status']!='WAITING_CAPACITY':raise ValueError('Unresolved worker not waiting on capacity')
            waiting.append(r['row_id'])
    if any(len(c['pids'])>4 for c in runtime['capacity'].values()):raise ValueError('Capacity limit exceeded')
    for r in runtime['rows']:
        init=r['initialization'];m=r['metrics']
        if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact']:raise ValueError('Initialization differs')
        if m['dual_arm']!='legacy_cosine' and (m['dual_main_gradient_norm']<=0 or m['dual_secondary_gradient_norm']<=0):raise ValueError('Active backbone gradient missing')
    evidence=dict(status='VERIFIED',source_commit=state['receipt']['commit'],controller=state['controller'],
        launched=16,resolved_rows=resolved,waiting_capacity_rows=waiting,epoch_telemetry_rows=len(runtime['rows']),
        capacity=runtime['capacity'],source_training_complete=False,test_complete=False,performance_gain=None)
    d.write(REPORT/'evidence/launch_verified.json',evidence)
    d.write(REPORT/'evidence/launch_readback.json',state)
    d.write(REPORT/'evidence/runtime_telemetry_readback.json',runtime)
    return evidence


if __name__=='__main__':print(json.dumps(capture()))
