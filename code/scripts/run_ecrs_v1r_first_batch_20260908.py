"""Launch the published B0/B2-V1/B2 source screen on GPUs 0/1/2."""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_phase1_adv3b02_ecrs_v1r_v2 import build_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke-checkpoint', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('run-id must be a single directory name')
    root = Path(__file__).resolve().parents[2]
    run_root = args.project_root / 'runs' / args.run_id
    log_root = args.project_root / 'logs' / args.run_id
    if args.detach:
        if run_root.exists():
            raise FileExistsError(run_root)
        log_root.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--project-root',
                str(args.project_root), '--run-id', args.run_id, '--smoke-checkpoint',
                str(args.smoke_checkpoint)]
        with (log_root / 'dispatch.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'dispatch_pid': child.pid, 'log_root': str(log_root), 'run_root': str(run_root)}))
        return 0
    if not log_root.is_dir():
        raise FileNotFoundError('launch via --detach to reserve a unique log root')
    # Atomic ownership prevents a duplicate direct invocation of this dispatch.
    with (log_root / 'dispatch_owner.json').open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'cwd': str(root)}, f)
    dataset = args.project_root / 'Dataset_WigSig/ManySig.pkl'
    if not dataset.is_file() or not args.smoke_checkpoint.is_file():
        raise FileNotFoundError('dataset or real smoke checkpoint missing')
    plan = build_plan(root, run_root, dataset, ['B0', 'B2-V1', 'B2'])
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    gpu_uuid = {int(a.strip()): b.strip() for a, b in (line.split(',') for line in gpu_lines)}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    counts = collections.Counter(line.split(',')[0].strip() for line in set(apps) if ',' in line)
    for gpu in (0, 1, 2):
        if gpu not in gpu_uuid or counts[gpu_uuid[gpu]] >= 2:
            raise RuntimeError(f'GPU {gpu} has no allowed training slot')
    # First executable check: real checkpoint reconstruction; no dataset/query load.
    import torch
    from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
    torch.set_num_threads(2)
    payload = torch.load(args.smoke_checkpoint, map_location='cpu', weights_only=False)
    device = torch.device('cuda:0')
    model, audit = build_exact_ssdg_model_from_checkpoint(payload, input_len=256, device=device)
    model.eval()
    with torch.no_grad():
        logits = model.forward_identity(torch.randn(2, 2, 256, device=device))['tx_logits']
    if not torch.isfinite(logits).all():
        raise RuntimeError('nonfinite checkpoint smoke output')
    print(json.dumps({'smoke': 'PASS', 'checkpoint': str(args.smoke_checkpoint),
                      'query_access': False, 'audit': audit}, default=str), flush=True)
    del model, payload, logits
    torch.cuda.empty_cache()
    run_root.mkdir(parents=True, exist_ok=False)
    for entry in plan['commands']:
        log_dir = str(log_root / entry['row'])
        entry['config']['log_dir'] = log_dir
        entry['argv'][entry['argv'].index('--log_dir') + 1] = log_dir
    (log_root / 'launch_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    children = []
    for gpu, entry in enumerate(plan['commands']):
        log_path = log_root / (entry['row'] + '.stdout.log')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUTF8='1',
                   PYTHONPATH=str(root / 'code'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with log_path.open('xb') as log:
            child = subprocess.Popen(entry['argv'], cwd=root, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append((child, entry['row'], gpu))
        with (log_root / 'children.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'row': entry['row'], 'pid': child.pid, 'gpu': gpu,
                'cwd': str(root), 'argv': entry['argv'], 'stdout': str(log_path)}) + '\n')
        print(f"RUNNING row={entry['row']} pid={child.pid} gpu={gpu}", flush=True)
    pending = list(children)
    failed = False
    while pending:
        for child, row, gpu in list(pending):
            status = child.poll()
            if status is not None:
                failed |= status != 0
                with (log_root / 'exit_status.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'row': row, 'pid': child.pid, 'exit_code': status}) + '\n')
                print(f'EXIT row={row} code={status}', flush=True)
                pending.remove((child, row, gpu))
        if pending:
            time.sleep(30)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
