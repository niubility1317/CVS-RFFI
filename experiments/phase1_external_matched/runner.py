"""Read-only frozen inference; scoring is a separate truth-last process."""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

RUN = '20261031-phase1-external-matched-nodura-m40-r02'
PROFILES = ('protonet_cda', 'mrior_sda', 'dadda_sda', 'twostage',
            'feature_separation_hu2024', 'poster_ft', 'radionet_ada_knn',
            'orthogonal', 'issl', 'lora_triplet_knn')
SEEDS = tuple(range(2026092701, 2026092705))
TASKS = ('cross_rx', 'cross_both')
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write(path, value):
    from experiments.practical_full_2x2.cache import atomic
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, value)


def prepare(rt, root):
    from experiments.pamf_phase2_practical.group1 import source_scope, row
    phase1 = read(rt['phase1_runtime'])
    context = read(phase1['context'])
    h = context['handles']['F0']
    source = read(rt['external_runtime'])
    scoped = source_scope(source)
    if read(h['source_contract']) != read(scoped['source_contract']):
        raise ValueError('Source contracts differ')
    if h['classes'] != source['old_classes']:
        raise ValueError('Physical class order differs')
    if any(not row(profile)['active'] for profile in PROFILES):
        raise ValueError('External profile missing')
    jobs = []
    for profile in PROFILES:
        for seed in SEEDS:
            row = f'{profile}-s{seed}'
            base = root / 'readouts' / row
            pred = dict(run_id=RUN, row_id=row, profile=profile, model_seed=seed,
                        source_runtime=scoped, classes=h['classes'], source_days=[1, 2, 3],
                        readouts=['raw'], tests={t: {'views_root': h['tests'][t]['views_root']} for t in TASKS},
                        output_root=str(base / 'predictions'))
            score = dict(run_id=RUN, row_id=row, model_seed=seed, classes=h['classes'],
                         source_days=[1, 2, 3], readouts=['raw'],
                         tests={t: h['tests'][t] for t in TASKS},
                         prediction_output=pred['output_root'], output_root=str(base / 'scores'))
            for kind, cfg, expected in [('predict', pred, 'ALL_PREDICTIONS_COMPLETE'),
                                         ('score', score, 'SCORED_COMPLETE')]:
                path = root / 'configs' / (kind + '__' + row + '.json')
                write(path, cfg)
                jobs.append(dict(id=kind+'__'+row, kind=kind, config=str(path),
                                 completion=str(Path(cfg['output_root'])/'complete.json'),
                                 expected=expected, dependency=pred['output_root']+'/complete.json' if kind=='score' else None))
    write(root/'jobs.json', jobs)
    return jobs


def done(job):
    try:
        return read(job['completion']).get('status') == job['expected']
    except (FileNotFoundError, ValueError):
        return False


def is_score_job(argv):
    return any(arg == '--job' and i + 1 < len(argv) and argv[i+1].startswith('score-')
               for i, arg in enumerate(argv))


def gpu_eligible(device, own_active):
    # Count every CUDA and pending experiment PID, including other owners.
    return len(device['pids']) < 4 and device['free'] >= 12000 and not own_active


def phase2_capacity(state, gpu, alive, device_pids):
    """Wait for Phase2 to fill its two slots before admitting another job."""
    if state['status'] != 'RUNNING':
        return not alive(state['controller_pid'])
    if not alive(state['controller_pid']):
        return False
    workers = [v for v in state['active'].values()
               if v.get('kind') == 'predict' and v.get('gpu') == gpu and alive(v['pid'])]
    return len(workers) == 2 and all(v['pid'] in device_pids for v in workers)


def process_alive(pid):
    try:
        stat = Path('/proc', str(pid), 'stat').read_text()
        return stat.split(') ', 1)[1][0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def other_scorer(owned):
    from experiments.pamf_repl_completion.runner import foreign_CPU_score
    if foreign_CPU_score(owned):
        return True
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in owned:
            continue
        try:
            argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if is_score_job(argv):
                return True
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return False


def smoke(rt):
    import torch
    from experiments.pamf_phase2_practical.group1 import source_scope
    from experiments.pamf_phase2_practical.group1_source import load_source, source_data
    source = source_scope(read(rt['external_runtime']))
    from experiments.pamf_phase2_practical.io import activate
    activate(source)
    torch.set_num_threads(1)
    samples = source_data(source, 'L_s').x[:2]
    checks = []
    for profile in PROFILES:
        bundle = load_source(source, profile, SEEDS[0], 'cpu')
        with torch.inference_mode():
            logits = bundle['score_fn'](bundle['model'], samples)
        if tuple(logits.shape) != (2, 6) or not bool(torch.isfinite(logits).all()):
            raise ValueError('Invalid source-only output: '+profile)
        checks.append(profile)
    write(Path(rt['run_root'])/'smoke.json', dict(status='PASS', profiles=checks,
          real_checkpoints=True, source_samples=2, target_reads=0, truth_reads=0))


def predict(c):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.evaluation import handle, reject_truth
    from experiments.pamf_phase2_practical.io import activate
    from experiments.pamf_phase2_practical.group1_source import load_source
    from experiments.pamf_phase2_practical.features import tensor
    reject_truth(c)
    if tuple(c['tests']) != TASKS or c['readouts'] != ['raw']:
        raise ValueError('Expected two frozen raw-only populations')
    source = c['source_runtime']; activate(source)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle = load_source(source, c['profile'], c['model_seed'], 'cuda:0')
    model = bundle['model'].eval()
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for task in TASKS:
        manifest, ids, positions = handle(c['tests'][task]['views_root'], c['classes'])
        values = {}
        for view in VIEWS:
            path = manifest['clean_ref'] if view == 'clean' else str(Path(manifest.get('observations_root', c['tests'][task]['views_root'])) / (view+'.npy'))
            iq = np.load(path, mmap_mode='r', allow_pickle=False)
            if iq.ndim != 3 or iq.shape[1:] != (2, 256):
                raise ValueError('Invalid target IQ shape')
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(ids), 128):
                    indices = slice(start, start+128) if positions is None else positions[start:start+128]
                    x = tensor(np.ascontiguousarray(iq[indices], dtype=np.float32), 'cuda:0')
                    scores = bundle['score_fn'](model, x)
                    if scores.shape != (len(x), 6) or not bool(torch.isfinite(scores).all()):
                        raise ValueError('Invalid native six-class readout')
                    predictions.extend(scores.argmax(1).cpu().tolist())
            values['raw__'+view] = np.asarray(predictions, dtype=np.int64)
            print('PREDICTED', c['row_id'], task, view, len(ids), flush=True)
        folder = out/task; folder.mkdir()
        np.savez(folder/'predictions.npz', ids=ids, **values)
        counts[task] = dict(count=len(ids), views=list(VIEWS), readouts=['raw'])
    write(out/'complete.json', dict(status='ALL_PREDICTIONS_COMPLETE', run_id=c['run_id'],
          row_id=c['row_id'], tests=counts, views=list(VIEWS), readouts=['raw'], truth_read=False, query_fit=False))


def score(c):
    from experiments.pamf_repl_view_shuffle.evaluation import score as original
    return original(c)


def analyze(rt, jobs):
    import numpy as np
    root = Path(rt['run_root']); rows = []; scored = []
    for job in jobs:
        if job['kind'] != 'score':
            continue
        cfg = read(job['config']); result = read(Path(cfg['output_root'])/'scores.json')
        scored.append((cfg['row_id'].rsplit('-s', 1)[0], cfg['model_seed'], result))
    for seed in SEEDS:
        original = Path(rt['pamf_run_root'])/'readouts'/f'main_split-repl-shuffled_time-3view-s{seed}'/'scores'
        if read(original/'complete.json').get('status') != 'SCORED_COMPLETE':
            raise ValueError('Original PAMF scoring incomplete')
        scored.append(('pamf_tri_sig_time_ssl', seed, read(original/'scores.json')))
    for profile, seed, result in scored:
        if not result['truth_last'] or not result['target_feedback_forbidden']:
            raise ValueError('Invalid scorer provenance')
        for view in VIEWS:
            for dimension in ('overall', 'receiver'):
                groups = defaultdict(list)
                for r in result['results']:
                    if r['view'] == view and r['readout'] == 'raw' and r['dimension'] == dimension:
                        groups[r['stratum']].append(np.asarray(r['confusion'], dtype=np.int64))
                for name, matrices in groups.items():
                    cm = sum(matrices); count = int(cm.sum())
                    if dimension == 'overall' and (len(matrices) != 2 or count != 168000):
                        raise ValueError('Incomplete target populations')
                    denom = cm.sum(0)+cm.sum(1)
                    f1 = np.divide(2*cm.diagonal(), denom, out=np.zeros(6), where=denom != 0).mean()
                    rows.append(dict(method=profile, seed=seed, view=view, dimension=dimension,
                                     receiver=name, count=count, accuracy=100*float(cm.trace()/count), macro_f1=100*float(f1)))
    out = root/'analysis'; out.mkdir(exist_ok=False)
    with (out/'per_seed.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups=defaultdict(list)
    for r in rows:
        groups[(r['method'],r['view'],r['dimension'],r['receiver'])].append(r)
    summary=[]
    for key, values in sorted(groups.items()):
        if len(values)!=4 or {v['seed'] for v in values}!=set(SEEDS):
            raise ValueError('Four seed results required')
        summary.append(dict(method=key[0],view=key[1],dimension=key[2],receiver=key[3],
                            count_per_seed=values[0]['count'],accuracy_mean=statistics.mean(v['accuracy'] for v in values),
                            accuracy_sd=statistics.stdev(v['accuracy'] for v in values),
                            macro_f1_mean=statistics.mean(v['macro_f1'] for v in values),
                            macro_f1_sd=statistics.stdev(v['macro_f1'] for v in values)))
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    write(out/'complete.json',dict(status='ANALYZED',methods=len(PROFILES),seeds=4,truth_last=True, DURA=False))


def dispatch(rt):
    from experiments.pamf_repl_view_shuffle.runner import resources
    root=Path(rt['run_root']); root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir(); jobs=prepare(rt,root); state={j['id']:'QUEUED' for j in jobs}; active={}; failed=[]
    def launch(kind, config, name, gpu=None):
        cmd=[rt['python'],'-u','-m','experiments.phase1_external_matched.runner','worker','--kind',kind,'--config',config]
        env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=root/'logs'/(name+'.log')
        with log.open('x') as f:
            proc=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        return proc,dict(pid=proc.pid,gpu=gpu,log=str(log),kind=kind,started_at=time.time())
    def save(status):
        write(root/'pipeline_state.json',dict(status=status,controller_pid=os.getpid(),updated_at=time.time(),
              counts=dict(Counter(state.values())),active={k:item for k,(_,item) in active.items()},failed=failed))
    save('SMOKE_PENDING')
    proc,item=launch('smoke',rt['runtime_file'],'smoke'); active['smoke']=(proc,item);save('SMOKE_RUNNING')
    code=proc.wait(); del active['smoke']
    if code or read(root/'smoke.json').get('status')!='PASS':
        failed.append('smoke');save('FAILED_NO_QUERY_SMOKE');return
    while True:
        for name,(proc,item) in list(active.items()):
            if proc.poll() is None:continue
            job=next(j for j in jobs if j['id']==name)
            state[name]='COMPLETE' if proc.returncode==0 and done(job) else 'FAILED'
            if state[name]=='FAILED':failed.append(name)
            del active[name]
        pending=[j for j in jobs if state[j['id']]=='QUEUED']
        scores=[j for j in pending if j['kind']=='score' and j['dependency'] and Path(j['dependency']).exists()]
        if scores and not any(x[1]['kind']=='score' for x in active.values()) and not other_scorer({p.pid for p,_ in active.values()}):
            j=scores[0];active[j['id']]=launch('score',j['config'],j['id']);state[j['id']]='RUNNING'
        if not failed:
            devices=resources()
            phase2=read(rt['phase2_state'])
            for gpu in rt['allowed_gpus']:
                dev=devices[gpu]
                # One new worker per GPU, including other users in the total limit.
                if not gpu_eligible(dev, any(x[1]['gpu']==gpu for x in active.values())) or not phase2_capacity(phase2,gpu,process_alive,dev['pids']):continue
                j=next((v for v in pending if v['kind']=='predict' and state[v['id']]=='QUEUED'),None)
                if not j:break
                active[j['id']]=launch('predict',j['config'],j['id'],gpu);state[j['id']]='RUNNING'
        save('RUNNING' if not failed else 'PARTIAL_TECHNICAL_FAILURE')
        if not active and (failed or all(v=='COMPLETE' for v in state.values())):break
        time.sleep(15)
    if not failed:
        analyze(rt,jobs);save('ANALYZED')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args(); c=read(a.config)
    if a.action=='dispatch':dispatch(c)
    elif a.kind=='smoke':smoke(c)
    elif a.kind=='predict':predict(c)
    elif a.kind=='score':score(c)
    else:raise ValueError('Unknown worker kind')
