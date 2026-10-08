from pathlib import Path
import json,sys
root=Path('E:/type10-7');wt=root/'code/snapshots/daot_practical_three_20260918_wt';sys.path.insert(0,str(wt))
from experiments.cvs_phase1_stack.publish import ssh
run='20261008-phase1-reference-repair-manysig-m32-r01'
script=r'''
from pathlib import Path
import json,time,subprocess
from datetime import datetime,timezone,timedelta
p=Path('/home/szu2070436088/2510044040/CV-SincNet');r=p/'runs/20261008-phase1-reference-repair-manysig-m32-r01'
control=p/'releases/cvs_reference_repair_direct_20261008_r01'
def read(f):return json.loads(f.read_text()) if f.is_file() else None
def proc(pid):
    path=Path('/proc')/str(pid)
    try:
        argv=[x for x in (path/'cmdline').read_bytes().decode().split('\0') if x]
        return dict(pid=pid,cwd=str((path/'cwd').resolve(strict=True)),argv=argv) if argv else None
    except (FileNotFoundError,ProcessLookupError):return None
receipt=read(control/'submit.json');rows=[]
for row in (read(r/'launch_source.json') or {}).get('rows',[]):
    out=r/row['row_id']/'source';log=Path(row['log']);ep=out/'epoch_metrics.jsonl';last=None
    if ep.is_file() and ep.stat().st_size:last=json.loads(ep.read_text().splitlines()[-1])
    rows.append(dict(**row,process=proc(row['pid']),lease=read(r/'capacity_leases'/('source-'+row['row_id']+'.json')),
        initialization=read(out/'initialization.json'),resolved=read(out/'resolved_config.json'),native_args=read(out/'resolved_native_args.json'),
        epoch=last,log_size=log.stat().st_size if log.is_file() else 0,log_tail=log.read_text(errors='replace').splitlines()[-5:] if log.is_file() else []))
print(json.dumps(dict(checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),receipt=receipt,
    controller=proc(receipt['pid']),previous_controller=proc(975107),queue=read(r/'queue_state.json'),failure=read(r/'failure.json'),
    dispatcher_active=read(r/'dispatcher_active.json'),rows=rows,control_log=(control/'controller.stdout.log').read_text(errors='replace')[-3000:],
    original_controllers=[proc(x) for x in (4170278,4170279,4176081)],
    original_r5=[proc(x) for x in (724060,724368,724689,724933)],
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))))
'''
v=json.loads(ssh(script));ev=root/'automation_reports/CV-SincNet'/run/'evidence'
(ev/'direct_start_readback.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(checked_at=v['checked_at'],controller=v['controller'],old_wait_owner=v['previous_controller'],queue=v['queue'],failure=v['failure'],
    original_controllers_alive=sum(x is not None for x in v['original_controllers']),original_r5_alive=sum(x is not None for x in v['original_r5']),
    rows=[dict(row_id=r['row_id'],gpu=r['gpu'],pid=r['pid'],alive=bool(r['process']),initialization=r['initialization'],
        resolved=bool(r['resolved']),epoch=(r['epoch'] or {}).get('epoch'),lease=r['lease'],log_size=r['log_size'],log_tail=r['log_tail']) for r in v['rows']],gpu=v['gpu'],control_log=v['control_log'] if v['failure'] else None),ensure_ascii=False))
