"""Single owner of the source-only architecture matrix. Never opens target data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.cvs_identity_ce.source import write
from experiments.cvs_clean_design.model import VARIANTS, BASELINES


def occupancy(active):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    mapping={v[1].strip():(int(v[0]),int(v[2])) for v in (r.split(',') for r in raw.splitlines())}
    result={g:dict(pids=set(),free_mb=free) for g,free in mapping.values()}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        v=line.split(',')
        if len(v)==2 and v[0].strip() in mapping:result[mapping[v[0].strip()][0]]['pids'].add(int(v[1]))
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().decode().split('\0')
            file_worker=any(Path(a).name.startswith(('train','predict')) and a.endswith('.py') for a in args)
            module_worker='-m' in args and any(token in args[args.index('-m')+1] for token in ('.source','.train','.predict'))
            if not(file_worker or module_worker):continue
            env=dict(v.split('=',1) for v in (p/'environ').read_bytes().decode().split('\0') if '=' in v)
            device=env.get('CUDA_VISIBLE_DEVICES','')
            if device.isdigit() and int(device) in result:result[int(device)]['pids'].add(int(p.name))
        except (OSError,UnicodeError,IndexError):continue
    for job in active.values():
        if job['process'].poll() is None:result[job['gpu']]['pids'].add(job['process'].pid)
    return result


def choose_gpu(active):
    values=occupancy(active)
    candidates=[(len(v['pids']),-v['free_mb'],g) for g,v in values.items() if len(v['pids'])<2 and v['free_mb']>=12000]
    return min(candidates)[2] if candidates else None


def select_source_candidate(records):
    import statistics
    groups={}
    for r in records:groups.setdefault(r['variant'],[]).append(r)
    if set(groups)!=set(VARIANTS):
        raise ValueError('Incomplete registered source architectures')
    summary={}
    for name,rows in groups.items():
        if {r['seed'] for r in rows}!={2026092701,2026092702,2026092703,2026092704} or len(rows)!=4:
            raise ValueError('Incomplete source seeds')
        summary[name]=dict(score=statistics.mean(.5*r['accuracy']+.5*r['worst_rx'] for r in rows),
            source_accuracy=statistics.mean(r['accuracy'] for r in rows),worst_rx_accuracy=statistics.mean(r['worst_rx'] for r in rows),
            parameters=max(r['parameters'] for r in rows),conv_linear_macs=statistics.mean(r['macs'] for r in rows))
    eligible=[name for name in groups if name not in BASELINES and summary[name]['parameters']<=1.1*summary['native']['parameters']]
    if not eligible:raise ValueError('No eligible registered candidate')
    # Registered tie window 0.2 percentage points; then favor actual lower MACs/params.
    best=max(summary[name]['score'] for name in eligible)
    tied=[name for name in eligible if summary[name]['score']>=best-.002]
    chosen=min(tied,key=lambda name:(summary[name]['conv_linear_macs'],summary[name]['parameters'],name))
    return dict(status='SOURCE_SELECTION_FROZEN',selected_variant=chosen,source_summaries=summary,
        selection_rule='four-seed mean of 0.5 source-V accuracy + 0.5 worst-source-RX accuracy; within0.2pp tie lower conv/linear MACs then parameters',
        target_access=False,target_score_used=False,test_variants=[*BASELINES,chosen],test_view='clean')


def dispatch(spec_path):
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8'))
    if any(k.startswith('p1_') or k.startswith('target') for k in spec):raise ValueError('Source dispatcher rejects target paths')
    if len(spec['rows'])!=32:raise ValueError('Unexpected source matrix')
    run,logs=Path(spec['runtime_root']),Path(spec['log_root'])
    run.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    state=dict(status='SOURCE_TRAINING',pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,run_id=spec['run_id'],
        launch_owner=spec['launch_owner'],commit=(ROOT/'release_commit.txt').read_text().strip(),
        rows={r['row_id']:dict(status='QUEUED',variant=r['variant'],model_seed=r['model_seed']) for r in spec['rows']},target_access=False)
    write(run/'pipeline_state.json',state);pending=list(spec['rows']);active={}
    while pending or active:
        for rid,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[rid];entry=state['rows'][rid]
            entry.update(exit_code=code,finished=time.time(),status='FAILED' if code else 'SOURCE_TRAINED')
            if not code:
                p=Path(entry['source_output'])/'completion.json'
                if not p.exists() or json.loads(p.read_text())['epoch']!=200:entry['status']='FAILED'
            write(run/'pipeline_state.json',state)
        if pending:
            gpu=choose_gpu(active)
            if gpu is not None:
                row=pending.pop(0);rid=row['row_id']
                command=[sys.executable,'-u','-m','experiments.cvs_clean_design.source','--config',row['source_config']]
                logpath=logs/(rid+'.log');log=logpath.open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[rid]=dict(process=child,gpu=gpu,log=log)
                state['rows'][rid].update(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,log=str(logpath),source_output=row['source_output'],started=time.time())
                write(run/'pipeline_state.json',state);print('LAUNCHED '+json.dumps(state['rows'][rid]),flush=True)
        if pending or active:time.sleep(5)
    if any(r['status']!='SOURCE_TRAINED' for r in state['rows'].values()):
        state['status']='FAILED_PARTIAL';write(run/'pipeline_state.json',state);return 1
    records=[]
    for row in spec['rows']:
        folder=Path(row['source_output']);done=json.loads((folder/'completion.json').read_text());profile=json.loads((folder/'resource_profile.json').read_text())
        m=done['final_source_metrics']
        records.append(dict(variant=row['variant'],seed=row['model_seed'],accuracy=m['source_val_accuracy'],
            worst_rx=m['source_val_worst_rx'],parameters=profile['total_parameters'],macs=profile['conv_linear_macs_per_sample']))
    write(run/'source_selection.json',select_source_candidate(records))
    state['status']='SOURCE_MATRIX_COMPLETE_AWAITING_FROZEN_CLEAN_TEST';write(run/'pipeline_state.json',state)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();raise SystemExit(dispatch(a.spec))
