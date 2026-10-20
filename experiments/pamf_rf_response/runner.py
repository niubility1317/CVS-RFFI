import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from . import design as d
from .runtime import ROOT, load_model, source_inputs, validate_args
from .models import directions, oracle_response, response_metrics
from .evaluation import predict, score
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.data import IQSource, write_json
from experiments.pamf_restoration.runner import array_tensor, code_commit
from experiments.pamf_restoration.models import pair
from experiments.pamf_restoration.dispatcher import capacity

PATH_ARGS = ('source_cache', 'parent_run', 'parent_action', 'reference_run', 'reader_folder', 'output')


def command(args, mode):
    result = [sys.executable, '-u', '-m', 'experiments.pamf_rf_response.runner', mode]
    for key in (*PATH_ARGS, 'expected_commit'):
        result += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    return result


def smoke(args):
    source_inputs(args)
    model, parent = load_model(args, 'cpu')
    x = array_tensor(IQSource(args.source_cache, 'L_s').iq[:2], 'cpu')
    main = torch.ones((2, 256), dtype=torch.complex64)
    identity = Action(torch.ones((2, 1), dtype=torch.complex64), main, torch.zeros_like(main))
    variance = torch.full((2,), .0001)
    with torch.no_grad():
        base = model(x)
        for delta, valid in directions(x, [0, 1], .01).values():
            change = identity(delta)
            plus, minus = model(x+pair(change)), model(x-pair(change))
            oracle, residual = oracle_response(identity, change, variance)
            values = response_metrics(plus, minus, base, delta, oracle, change, variance, valid, residual)
            if any(not torch.isfinite(value).all() for value in values.values()):
                raise ValueError('Actual checkpoint finite-response smoke failed')
    reader = load_reader(args.reader_folder, args.source_cache, 'cpu').requires_grad_(False).eval()
    with torch.no_grad():
        if not torch.isfinite(reader(base)).all():
            raise ValueError('Original F50/default-head no-query smoke failed')
    write_json(args.output/'smoke.json', dict(status='PASS', real_Ls_samples=2, fixed_full_parent=parent,
        original_F50_default_head=True, F_counterfactual_inputs=0, optimizer_updates=0,
        TX_truth_reads=0, target_reads=0, no_query=True))


def run(args):
    import fcntl
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('New immutable output and CPU controller required')
    args.output.mkdir()
    write_json(args.output/'launch.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv,
        commit=args.expected_commit, owner='codex/root/RF-response', started=time.time()))
    write_json(args.output/'resolved_config.json', d.configuration())
    try:
        smoke(args)
        process = None
        while process is None:
            with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = capacity()
                cards = [g for g in d.GPUS if len(state[g]['pids']) < 2 and state[g]['free_mb'] >= 12000]
                cards.sort(key=lambda g: (len(state[g]['pids']), -state[g]['free_mb'], list(d.GPUS).index(g)))
                if cards:
                    gpu = cards[0]
                    cmd, log = command(args, 'predict'), args.output/'predict.log'
                    with log.open('x', encoding='utf-8') as stream:
                        process = subprocess.Popen(cmd, cwd=ROOT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    write_json(args.output/'worker.json', dict(pid=process.pid, GPU=gpu, cwd=str(ROOT),
                        argv=cmd, log=str(log), started=time.time()))
                    deadline = time.monotonic()+45
                    while process.poll() is None and time.monotonic() < deadline:
                        if process.pid in capacity()[gpu]['pids']:
                            break
                        time.sleep(.5)
                    if process.poll() is None and process.pid not in capacity()[gpu]['pids']:
                        raise RuntimeError('CUDA PID not visible within45s;preserve process;no automatic retry')
            if process is None:
                time.sleep(2)
        if process.wait() != 0:
            raise RuntimeError('Source counterfactual prediction failed;preserve output;no automatic retry')
        subprocess.run(command(args, 'score'), cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
        write_json(args.output/'complete.json', dict(status='ANALYZED', source_only=True, target_reads=0,
            epochs=0, optimizer_updates=0, source_V_state_updates=0, all_counterfactuals_before_truth=True,
            counterfactual_class_predictions=0, target_promotion=False, goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error),
            traceback=traceback.format_exc(), outputs_preserved=True, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description='Frozen source RF response;no training or target mode')
    parser.add_argument('mode', choices=('run', 'predict', 'score'))
    for key in PATH_ARGS:
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    for key in PATH_ARGS:
        setattr(args, key, getattr(args, key).resolve())
    validate_args(args)
    if code_commit() != args.expected_commit:
        raise ValueError('Fixed source diagnostic code commit required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {'run': run, 'predict': predict, 'score': score}[args.mode](args)


if __name__ == '__main__':
    main()
