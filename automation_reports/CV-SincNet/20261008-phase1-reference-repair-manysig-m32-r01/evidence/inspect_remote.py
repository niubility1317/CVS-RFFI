from pathlib import Path
import json,sys
root=Path('E:/type10-7');wt=root/'code/snapshots/daot_practical_three_20260918_wt';sys.path.insert(0,str(wt))
from experiments.cvs_phase1_stack.publish import ssh
from experiments.cvs_phase1_repair import design as d
script=r'''
from pathlib import Path
import json,sys,subprocess,time
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
run=project/'runs/20261008-phase1-reference-repair-manysig-m32-r01'
release=project/'releases/cvs_reference_repair_20261008_r01'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def process(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,cwd=str((p/'cwd').resolve(strict=True)),argv=[x for x in (p/'cmdline').read_bytes().decode().split('\0') if x])
    except FileNotFoundError:return None
receipt=read(release/'submit.json')
configs=[read(p) for p in sorted((run/'configs').glob('*.json'))]
workers=[]
for phase in ['source','predict']:
    for row in (read(run/('launch_'+phase+'.json')) or {}).get('rows',[]):
        p=run/row['row_id']/('source' if phase=='source' else 'prediction');ep=p/'epoch_metrics.jsonl'
        last=json.loads(ep.read_text().splitlines()[-1]) if ep.is_file() and ep.stat().st_size else None
        log=Path(row['log'])
        workers.append(dict(**row,phase=phase,process=process(row['pid']),initialization=read(p/'initialization.json'),
            config=read(p/'resolved_config.json'),native_args=read(p/'resolved_native_args.json'),
            epoch=last,complete=read(p/('completion.json' if phase=='source' else 'complete.json')),
            log_bytes=log.stat().st_size if log.is_file() else 0,log_tail=log.read_text(errors='replace').splitlines()[-4:] if log.is_file() else []))
owners=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        argv=[x for x in (p/'cmdline').read_bytes().decode().split('\0') if x]
        if any(x.startswith('experiments.cvs_phase1_stack.') for x in argv):owners.append(process(int(p.name)))
    except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeError):continue
print(json.dumps(dict(read_at=time.time(),receipt=receipt,controller_process=process(receipt['pid']) if receipt else None,
    dispatcher=read(run/'dispatcher.json'),queue=read(run/'queue_state.json'),failure=read(run/'failure.json'),completion=read(run/'completion.json'),
    smoke=read(release/'smoke-remote/completion.json'),scoring=read(run/'scoring_complete.json'),configs=configs,workers=workers,
    older_stack_processes=owners,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),
    release_log=(release/'dispatcher.stdout.log').read_text(errors='replace')[-4000:] if (release/'dispatcher.stdout.log').is_file() else None,
    preflight_log=(release/'preflight.log').read_text(errors='replace')[-1000:] if (release/'preflight.log').is_file() else None)))
'''
v=json.loads(ssh(script))
out=root/'automation_reports/CV-SincNet'/d.RUN/'evidence';out.mkdir(parents=True,exist_ok=True)
(out/'remote_readback.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if v['receipt']:
    assert v['controller_process']['cwd']==v['receipt']['cwd']
    assert v['controller_process']['argv']==v['receipt']['argv']
    assert v['dispatcher']['commit']==v['receipt']['commit']
    assert v['smoke']['status']=='PASS' and len(v['smoke']['rows'])==8
    assert v['failure'] is None
    assert sorted(v['configs'],key=lambda x:x['row_id'])==sorted([d.config(r) for r in d.rows()],key=lambda x:x['row_id'])
print(json.dumps(dict(read_at=v['read_at'],controller=v['controller_process'],commit=(v['receipt'] or {}).get('commit'),
    smoke=v['smoke'],queue=v['queue'],failure=v['failure'],rows=len(v['configs']),workers=len(v['workers']),
    older_stack_processes=v['older_stack_processes'],preflight_log=v['preflight_log'] if not v['smoke'] else None),ensure_ascii=False))
