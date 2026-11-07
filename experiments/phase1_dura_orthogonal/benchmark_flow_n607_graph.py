"""Queue and measure source-only CUDA-graph parity on one N607 GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.phase1_dura_orthogonal.gpu_capacity_queue import capacity


def wait_for_slot(out, gpu):
    print('N607_FLOW_GRAPH_QUEUE_WAITING', gpu, flush=True)
    while True:
        if out.exists():
            raise FileExistsError('Diagnostic output already exists')
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
        ready, free, count = capacity(rows, apps, gpu)
        if ready:
            selected = next(row.split(',')[1].strip() for row in rows.splitlines()
                            if row.split(',')[0].strip() == str(gpu))
            print('N607_FLOW_GRAPH_QUEUE_SLOT', gpu, selected, 'free_mib', free, 'other_apps', count, flush=True)
            return selected
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
    selected_uuid = wait_for_slot(args.out, args.gpu)
    if 'torch' in sys.modules:
        raise RuntimeError('GPU selection must occur before torch import')
    os.environ.update(CUDA_VISIBLE_DEVICES=selected_uuid, OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
    from experiments.phase1_dura_orthogonal.benchmark_flow_cuda_graph import graph_train
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
    print('N607_FLOW_GRAPH_REAL_DURA_SMOKE_PASS', flush=True)
    for graph in (False, True):
        warm = graph_train(small, 1, epoch_times=True) if graph else train(small, 'conditional', True, 1)
        del warm
    pairs = []
    for order in ((False, True), (True, False)):
        outcomes = {graph: graph_train(selected, 5, epoch_times=True) if graph
                    else train(selected, 'conditional', True, 5) for graph in order}
        eager, captured = outcomes[False], outcomes[True]
        match = exact(eager, captured)
        pairs.append(dict(eager_seconds=eager[5], graph_epoch_seconds=captured[5], steps=len(eager[2]),
                          losses_parameters_optimizer_rng_exact=match,
                          max_loss_absolute_difference=float((eager[2]-captured[2]).abs().max())))
        if not match:
            break
        del outcomes, eager, captured
    args.out.mkdir(parents=True)
    result = dict(status='N607_FLOW_GRAPH_PARITY_COMPLETE' if all(p['losses_parameters_optimizer_rng_exact'] for p in pairs)
                  else 'N607_FLOW_GRAPH_PARITY_FAILED', commit=args.commit, source_ids=512,
                  views_per_id=3, epochs=5, batch=32, pairs=pairs, graph_capture_included_in_timing=True,
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(), physical_gpu=args.gpu,
                  physical_gpu_uuid=selected_uuid,
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, weights_saved=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
