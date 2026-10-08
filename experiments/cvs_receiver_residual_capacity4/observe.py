"""Independent read-only post-state of owners, preserved workers and global GPU use."""
import json
from pathlib import Path
from experiments.cvs_receiver_residual_v2.publish import ssh
from experiments.cvs_receiver_residual_v2 import design as d

SCRIPT = r'''
import json,subprocess,time
from pathlib import Path
project=Path(PROJECT)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        if s[0] in ('Z','X','x'):return None
        return dict(pid=pid,start_ticks=int(s[19]),cwd=str((p/'cwd').resolve(strict=True)),
            argv=[x for x in (p/'cmdline').read_bytes().decode().split('\0') if x])
    except (FileNotFoundError,ProcessLookupError):return None
caps={};uuids={}
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
    g,u,used,free,util=[x.strip() for x in line.split(',')];g=int(g);uuids[u]=g
    caps[g]=dict(cuda_pids=set(),reserved_pids=set(),memory_used_mb=int(used),memory_free_mb=int(free),utilization_pct=int(util))
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
    if not line.strip():continue
    pid,u=[x.strip() for x in line.split(',')]
    if u in uuids:caps[uuids[u]]['cuda_pids'].add(int(pid))
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if not proc(int(p.name)):continue
        env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
        gpu=env.get('CUDA_VISIBLE_DEVICES','')
        if gpu.isdigit() and int(gpu) in caps:caps[int(gpu)]['reserved_pids'].add(int(p.name))
    except (OSError,UnicodeError):pass
for c in caps.values():
    c['total_pids']=sorted(c['cuda_pids']|c['reserved_pids']);c['cuda_pids']=sorted(c['cuda_pids']);c['reserved_pids']=sorted(c['reserved_pids'])
    c['count']=len(c['total_pids'])
    if c['count']>4:raise ValueError('Global GPU process limit exceeded')
runs=[]
for run in RUNS:
    base=project/'runs'/run;owner=read(base/'dispatcher_active.json');actual=proc(owner['pid'])
    if not actual or any(actual[k]!=owner[k] for k in ('pid','start_ticks','cwd','argv')):raise ValueError('Owner identity differs')
    queue=read(base/'queue_state.json')
    if queue['per_gpu_limit']!=4 or queue['failures'] or read(base/'failure.json'):raise ValueError('Queue not healthy at four/GPU')
    handoff=read(base/'capacity4_handoff_v2.json');preserved=[]
    for before in handoff['workers_before']:
        after=proc(before['pid'])
        if not after or any(after[k]!=before[k] for k in ('pid','start_ticks','cwd','argv')):raise ValueError('Worker not preserved')
        preserved.append(after)
    old=handoff['old_dispatcher']
    if proc(old['pid']):raise ValueError('Old scheduling owner remains alive')
    rows=[];ids=set()
    for r in read(base/'launch_source.json')['rows']:
        if r['row_id'] in ids:raise ValueError('Duplicate row receipt')
        ids.add(r['row_id']);source=base/r['row_id']/'source';live=proc(r['pid']);done=read(source/'completion.json')
        epochs=source/'epoch_metrics.jsonl';latest=None
        if epochs.exists():
            lines=epochs.read_text().splitlines()
            if lines:
                try:latest=json.loads(lines[-1])['epoch']
                except json.JSONDecodeError:pass
        if live and 'start_ticks' in r and live['start_ticks']!=r['start_ticks']:raise ValueError('Worker PID reused')
        rows.append(dict(row_id=r['row_id'],pid=r['pid'],gpu=r['gpu'],alive=bool(live),epoch=latest,
            source_complete=done and done['status']=='SOURCE_TRAINED',log_bytes=Path(r['log']).stat().st_size))
    runs.append(dict(run_id=run,owner=owner,owner_actual=actual,queue=queue,preserved_workers=preserved,
        source_complete=sum(bool(r['source_complete']) for r in rows),source_active=sum(r['alive'] for r in rows),rows=rows,
        source_frozen=(base/'source_matrix_frozen.json').exists(),scored=(base/'scoring_complete.json').exists()))
print(json.dumps(dict(status='VERIFIED',read_at=time.time(),gpu=caps,runs=runs,
    evidence='independent /proc PID-start-cwd-argv + nvidia-smi + completion artifacts; scheduling only')))
'''


def main():
    runs=['20261008-phase1-receiver-residual-manysig-m16-r01',d.RUN]
    code=SCRIPT.replace('PROJECT',repr(d.PROJECT)).replace('RUNS',repr(runs))
    data=json.loads(ssh(code));out=d.ROOT/'local_artifacts/cvs_receiver_residual_capacity4_20261008_r02'
    out.mkdir(exist_ok=True)
    (out/'verified_readback.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=data['status'],gpu_counts={g:c['count'] for g,c in data['gpu'].items()},
        gpu_free_mb={g:c['memory_free_mb'] for g,c in data['gpu'].items()},runs=[dict(run_id=r['run_id'],
        owner_pid=r['owner']['pid'],preserved=len(r['preserved_workers']),active=r['source_active'],
        complete=r['source_complete'],pending=len(r['queue']['pending']),epoch_range=[min(x['epoch'] for x in r['rows'] if x['alive'] and x['epoch']),
        max(x['epoch'] for x in r['rows'] if x['alive'] and x['epoch'])]) for r in data['runs']])))


if __name__=='__main__':main()
