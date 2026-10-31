"""Complete the failed first G's seven-view evaluation without retraining."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from . import contextual_inverse as candidate
from . import runner as base

RUN = '20261101-phase1-contextual-inverse-eval-s2026092701-r02'
ORIGINAL_RUN = candidate.RUN
MODEL_COMMIT = '9959956a0572de08fbcac3258fd30a5c0769763a'
MODULE = 'experiments.pamf_dura_direct_train.contextual_recovery'


def paths(root):
    p = base.paths(root)
    p['original_output'] = root/'runs'/ORIGINAL_RUN
    p['output'] = root/'runs'/RUN
    return p


def model_args(execution_commit):
    return SimpleNamespace(arm=candidate.ARM, commit=execution_commit, checkpoint_commit=MODEL_COMMIT)


def verify_parent(p):
    row = p['original_output']/candidate.ARM
    completed = base.read_json(row/'training_complete.json')
    failure = base.read_json(row/'failure.json')
    profile = base.read_json(row/'matched_profile.json')
    if (completed.get('status'),completed.get('epoch'),completed.get('updates'),
        completed.get('training_F_access'),completed.get('target_reads'),completed.get('OE_reads')) != ('SOURCE_TRAINED',30,7410,False,0,0):
        raise ValueError('Only the exact completed F-free source parent is allowed')
    if "Inference tensors do not track version counter" not in failure.get('error',''):
        raise ValueError('Expected preserved inference-version-counter failure required')
    if profile.get('status') != 'PROFILE_COMPLETE':
        raise ValueError('Preserved same-G historical-F50 profile required')
    model = candidate.load_model(row/'final.pt','cpu',arm=candidate.ARM,commit=MODEL_COMMIT)
    return row, model


def smoke(p, model):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    clean, roles = base.source_training(p)
    indices = np.flatnonzero(roles == 'L_s')[:2]
    source = np.load(p['cache']/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    x = base.tensor(source[indices],'cpu')
    # Both the reader and its buffers are created before no_grad; they retain
    # version counters required by the frozen Sinc cache.
    reader = load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    with torch.no_grad():
        iq = model(x)
        scores = reader(iq)
    if iq.shape != (2,2,256) or scores.shape != (2,6) or not torch.isfinite(scores).all():
        raise ValueError('Actual fixed G/F checkpoint source smoke failed')
    return dict(status='PASS', real_Ls=2, source_G_checkpoint_reused=True,
                fixed_F50_checkpoint=True, version_counter_reader=True,
                new_optimizer_updates=0, target_reads=0, TX_labels_read=0)


def command(root, execution_commit, mode, role=None):
    args=[sys.executable,'-u','-m',MODULE,mode,'--root',str(root),'--execution-commit',execution_commit]
    if role is not None:
        args.extend(['--role',role])
    return args


def goal_worker_tokens():
    """Use the pinned original dispatcher's existing owner inventory."""
    tree = ast.parse(inspect.getsource(base.dispatch))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tokens' for t in node.targets):
            tokens = ast.literal_eval(node.value)
            if not isinstance(tokens, tuple) or not tokens or not all(isinstance(x,str) for x in tokens):
                raise ValueError('Original restoration worker inventory is malformed')
            return tokens + ('experiments.pamf_dura_direct_train.contextual_inverse worker', MODULE+' worker')
    raise ValueError('Original restoration worker inventory missing')


def active_goal_workers(commands, tokens):
    workers = [line for line in commands if any(token in line for token in tokens)]
    used = set()
    for line in workers:
        pid = line.split(None,1)[0]
        try:
            environment = (Path('/proc')/pid/'environ').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        for item in environment:
            if item.startswith(b'CUDA_VISIBLE_DEVICES='):
                used.update(int(v) for v in item.split(b'=',1)[1].split(b',') if v.isdigit())
    return len(workers), used


def run(root, execution_commit):
    import fcntl
    p = paths(root)
    if p['output'].exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('New output and CPU-only controller required')
    parent, model = verify_parent(p)
    checks = smoke(p,model)
    p['output'].mkdir(); (p['output']/'logs').mkdir()
    row = p['output']/candidate.ARM; row.mkdir()
    original = parent/'final.pt'; copied = row/'final.pt'
    with original.open('rb') as source, copied.open('xb') as dest:
        shutil.copyfileobj(source,dest)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(original) != sha(copied):
        raise ValueError('Frozen checkpoint copy differs')
    shutil.copyfile(parent/'training_complete.json', row/'training_complete.json')
    shutil.copyfile(parent/'matched_profile.json', row/'matched_profile.json')
    base.write_json(p['output']/'smoke.json',checks)
    base.write_json(p['output']/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        code_commit=execution_commit,model_commit=MODEL_COMMIT,parent_run=ORIGINAL_RUN,
        candidate_checkpoint_sha256=sha(copied),new_optimizer_updates=0,
        profile='Exact existing completed original same-G/same-F50 matched profile copied, no remeasurement'))
    process=None
    while process is None:
        with (root/'runs/receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=base.capacity()
            commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
            workers, used=active_goal_workers(commands,goal_worker_tokens())
            free=base.eligible_gpus(state,used,3,[4,6,7])
            if workers >= 2:
                free=[]
            if free:
                gpu=free[0]
                with (p['output']/'logs/worker.log').open('xb') as log:
                    process=subprocess.Popen(command(root,execution_commit,'worker'),cwd=Path.cwd(),
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                base.write_json(p['output']/'submitted.json',dict(pid=process.pid,GPU=gpu,
                    free_memory_MB=state[gpu]['free_mb'],existing_compute_PIDs=len(state[gpu]['pids'])))
        if process is None:
            print('WAITING_SAFE_GPU_CAPACITY',flush=True)
            time.sleep(10)
    code=process.wait()
    success=code == 0 and (row/'complete.json').is_file()
    base.write_json(p['output']/'complete.json',dict(status='ANALYZED' if success else 'FAILED_TECHNICAL',
        exit_code=code, source_G_reused=True, new_optimizer_updates=0, target_feedback=False,
        phase3_test_complete=False))


def worker(root, execution_commit):
    import fcntl
    p=paths(root);args=model_args(execution_commit)
    row=p['output']/candidate.ARM
    try:
        for role in ('source','target'):
            candidate.predict(args,p,role)
            with (p['output']/'score.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                subprocess.run(command(root,execution_commit,'score',role),
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        base.write_json(row/'complete.json',dict(status='ANALYZED',source_G_reused=True,
            new_optimizer_updates=0,all_seven_predictions_before_truth=True,target_feedback=False,
            phase3_test_complete=False))
    except BaseException as error:
        base.write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
            traceback=traceback.format_exc(),outputs_preserved=True))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('run','worker','score'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--execution-commit',required=True)
    parser.add_argument('--role',choices=('source','target'))
    args=parser.parse_args();root=args.root.resolve()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.execution_commit:
        raise ValueError('Pinned recovery release commit required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    candidate.configure()
    if args.mode == 'run':
        run(root,args.execution_commit)
    elif args.mode == 'worker':
        worker(root,args.execution_commit)
    else:
        if args.role is None:parser.error('--role required for scoring')
        from .evaluation import score
        score(model_args(args.execution_commit),paths(root),args.role)


if __name__ == '__main__':
    main()
