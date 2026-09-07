"""Bounded four-row source-only queue; never stop or restart existing jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_phase1_adv3b02_ecrs_v1r_v2 import build_plan

ROWS = ['B3a', 'B3b', 'B3c', 'B4']
FIRST_DISPATCH = 'run_ecrs_v1r_first_batch_20260908.py'


def occupied_slots(gpu_uuids, compute, ignored_pids=(), reservations=()):
    """Count unique compute PIDs and not-yet-visible children, per GPU."""
    sets = {gpu: set() for gpu in gpu_uuids}
    reverse = {uuid: gpu for gpu, uuid in gpu_uuids.items()}
    for uuid, pid in compute:
        if uuid in reverse and pid not in ignored_pids:
            sets[reverse[uuid]].add(pid)
    for gpu, pid in reservations:
        sets[gpu].add(pid)
    return {gpu: len(pids) for gpu, pids in sets.items()}


def inventory(project, active):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'], text=True)
    uuids = {int(a.strip()): b.strip() for a, b in (s.split(',') for s in output.splitlines())}
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    compute = [(a.strip(), int(b.strip())) for a, b in
               (s.split(',') for s in output.splitlines() if ',' in s)]
    ignored = set()
    for _, pid in compute:
        proc = Path('/proc') / str(pid)
        try:
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
            cwd = (proc / 'cwd').resolve()
            scripts = [Path(x) for x in argv if x.endswith('.py')]
            # The exact first-batch supervisor retains a CUDA context but does no training.
            if len(scripts) == 1 and scripts[0].name == FIRST_DISPATCH and (
                    project / 'releases') in cwd.parents and scripts[0] == cwd / 'code/scripts' / FIRST_DISPATCH:
                ignored.add(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass  # Unknown processes continue to count as occupied slots.
    reservations = [(gpu, child.pid) for child, row, gpu in active if child.poll() is None]
    return occupied_slots(uuids, compute, ignored, reservations)


def smoke(checkpoint):
    import torch
    from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
    from model_dual_cvsincnet import build_dual_model
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model, audit = build_exact_ssdg_model_from_checkpoint(payload, input_len=256, device=device)
    model.eval()
    with torch.no_grad():
        logits = model.forward_identity(torch.randn(2, 2, 256, device=device))['tx_logits']
    assert torch.isfinite(logits).all()
    del model, payload, logits
    # Exercise the four new response paths on this torch/CUDA environment, no data/query access.
    variants = [('B3a', 'legacy28_old_reference', 'real8'),
                ('B3b', 'legacy28_estimated_reference', 'real8'),
                ('B3c', 'compact8', 'real8'), ('B4', 'compact8', 'complex24')]
    for row, variant, anchor in variants:
        model = build_dual_model(3, 2, model_size='M', dataset='wisig', input_len=256,
            model_variant='lite_d', branch_ablation='no_dac', domain_branch_ablation='no_stats',
            use_ecrs=True, ecrs_config={'version': 'v2', 'estimator_variant': variant,
                                      'anchor_mode': anchor}).to(device)
        out = model.forward_response(torch.randn(3, 2, 256, device=device))
        loss = torch.nn.functional.cross_entropy(out['resp_tx_logits'], torch.arange(3, device=device))
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.response_encoder().parameters())
        assert all(p.grad is None for p in model.ecrs.physical.parameters())
        print(json.dumps({'response_smoke': row, 'status': 'PASS', 'query_access': False}), flush=True)
        del model, out, loss
    print(json.dumps({'checkpoint_smoke': 'PASS', 'audit': audit}, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke-checkpoint', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.smoke_only:
        smoke(args.smoke_checkpoint)
        return 0
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('run-id must be a single directory name')
    root = Path(__file__).resolve().parents[2]
    project = args.project_root.resolve()
    run_root, logs = project / 'runs' / args.run_id, project / 'logs' / args.run_id
    argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--project-root', str(project),
            '--run-id', args.run_id, '--smoke-checkpoint', str(args.smoke_checkpoint)]
    if args.detach:
        if run_root.exists():
            raise FileExistsError(run_root)
        logs.mkdir(parents=True, exist_ok=False)
        with (logs / 'dispatch.log').open('xb') as f:
            child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=f,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'dispatch_pid': child.pid, 'log_root': str(logs)}))
        return 0
    with (logs / 'dispatch_owner.json').open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'cwd': str(root)}, f)
    data = project / 'Dataset_WigSig/ManySig.pkl'
    if not data.is_file() or not args.smoke_checkpoint.is_file():
        raise FileNotFoundError('dataset or checkpoint missing')
    plan = build_plan(root, run_root, data, ROWS)
    for entry in plan['commands']:
        entry['config']['log_dir'] = str(logs / entry['row'])
        entry['argv'][entry['argv'].index('--log_dir') + 1] = entry['config']['log_dir']
    (logs / 'launch_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    pending, active, failed = list(plan['commands']), [], False
    deadline = time.monotonic() + 24 * 3600
    smoke_done = False
    while pending or active:
        for child, row, gpu in list(active):
            code = child.poll()
            if code is not None:
                failed |= code != 0
                with (logs / 'exit_status.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'row': row, 'pid': child.pid, 'exit_code': code}) + '\n')
                active.remove((child, row, gpu))
        if pending and ((logs / 'STOP_LAUNCHING').exists() or time.monotonic() >= deadline):
            (logs / 'unstarted.json').write_text(json.dumps([e['row'] for e in pending]), encoding='utf-8')
            pending.clear()
            failed = True
        slots = inventory(project, active) if pending else {}
        available = sorted((g for g, count in slots.items() if count < 2), key=lambda g: (slots[g], g))
        if pending and available:
            gpu = available[0]
            if not smoke_done:
                subprocess.run(argv + ['--smoke-only'], cwd=root, stdin=subprocess.DEVNULL,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUTF8='1'), check=True)
                smoke_done = True
                run_root.mkdir(parents=True, exist_ok=False)
                continue  # Re-read capacity after the smoke subprocess has exited.
            entry = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUTF8='1',
                PYTHONPATH=str(root / 'code'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            path = logs / (entry['row'] + '.stdout.log')
            with path.open('xb') as f:
                child = subprocess.Popen(entry['argv'], cwd=root, stdin=subprocess.DEVNULL,
                    stdout=f, stderr=subprocess.STDOUT, env=env)
            active.append((child, entry['row'], gpu))
            with (logs / 'children.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'row': entry['row'], 'pid': child.pid, 'gpu': gpu,
                    'cwd': str(root), 'argv': entry['argv'], 'stdout': str(path)}) + '\n')
            print(f"RUNNING row={entry['row']} pid={child.pid} gpu={gpu}", flush=True)
        (logs / 'queue_status.json').write_text(json.dumps({'pending': [e['row'] for e in pending],
            'active': [{'row': r, 'pid': p.pid, 'gpu': g} for p, r, g in active],
            'smoke_done': smoke_done, 'remaining_launch_window_s': max(0, deadline-time.monotonic())}), encoding='utf-8')
        if pending or active:
            time.sleep(30)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
