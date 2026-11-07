"""Queue a source-only N607 comparison of flow training guard paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.phase1_dura_orthogonal.gpu_capacity_queue import capacity


def idle_uuid(rows, apps, gpu):
    ready, _, count = capacity(rows, apps, gpu)
    if not ready or count:
        return None
    return next(row.split(',')[1].strip() for row in rows.splitlines()
                if row.split(',')[0].strip() == str(gpu))


def wait_for_idle(out, gpu):
    print('N607_FLOW_SYNC_WAITING', gpu, flush=True)
    while True:
        if out.exists():
            raise FileExistsError('Diagnostic output already exists')
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
        uuid = idle_uuid(rows, apps, gpu)
        if uuid:
            print('N607_FLOW_SYNC_IDLE_SLOT', gpu, uuid, flush=True)
            return uuid
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if not args.cache.is_file():
        raise FileNotFoundError('Fixed source context cache required')
    uuid = wait_for_idle(args.out, args.gpu)
    if 'torch' in sys.modules:
        raise RuntimeError('GPU selection must occur before torch import')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
    from experiments.phase1_dura_orthogonal.benchmark_flow_sync import exact, train
    from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
    from experiments.phase1_dura_orthogonal.train_source import WIDE_FIT_SIZE, extract_context, load_source_training

    if not torch.cuda.is_available():
        raise ValueError('Selected CUDA device unavailable')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ids, _, _, _, _, _ = source_inputs(args.root)
    _, _, _, fit = load_source_training(args.root, ids, fit_size=WIDE_FIT_SIZE)
    data, _ = load_cache(args.cache, fit, 32, 'cuda:0', deterministic=True)
    small = tuple(torch.cat([v[i:i+64] for i in (0, 32768, 65536)]).cuda() for v in data)
    selected = tuple(torch.cat([v[i:i+512] for i in (0, 32768, 65536)]).cuda() for v in data)
    del data
    dura = frozen_dura(args.root, 'cuda:0')
    with torch.no_grad():
        context = extract_context(dura, selected[0][:8])
        if not all(torch.isfinite(v).all() for v in context) or any(p.requires_grad for p in dura.parameters()):
            raise ValueError('Real frozen DURA source smoke required')
    del dura
    print('N607_FLOW_SYNC_REAL_DURA_SMOKE_PASS', flush=True)
    results = {}
    for mode in ('conditional', 'mixed', 'direct'):
        for fast in (False, True):
            warm = train(small, mode, fast, 1)
            del warm
        pairs = []
        for order in ((False, True), (True, False)):
            outcomes = {fast: train(selected, mode, fast, 5) for fast in order}
            original, fast = outcomes[False], outcomes[True]
            match = exact(original, fast)
            pairs.append(dict(original_seconds=original[5], fast_seconds=fast[5], steps=len(original[2]),
                              losses_parameters_optimizer_rng_exact=match,
                              max_loss_absolute_difference=float((original[2]-fast[2]).abs().max())))
            if not match:
                break
            del outcomes, original, fast
        results[mode] = pairs
        if not all(p['losses_parameters_optimizer_rng_exact'] for p in pairs):
            break
    ok = len(results) == 3 and all(len(pairs) == 2 and all(p['losses_parameters_optimizer_rng_exact']
             for p in pairs) for pairs in results.values())
    args.out.mkdir(parents=True)
    result = dict(status='N607_FLOW_SYNC_PARITY_COMPLETE' if ok else 'N607_FLOW_SYNC_PARITY_FAILED',
                  commit=args.commit, source_ids=512, views_per_id=3, epochs=5, batch=32,
                  results=results, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  physical_gpu=args.gpu, physical_gpu_uuid=uuid, TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0, weights_saved=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
