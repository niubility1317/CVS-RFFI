"""Queue a source-only N607 comparison of batch versus epoch flow-time transfer."""
import argparse
import json
import os
from pathlib import Path
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_graph import wait_for_slot


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
    uuid = wait_for_slot(args.out, args.gpu)
    if 'torch' in sys.modules:
        raise RuntimeError('GPU selection must occur before torch import')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
    from experiments.phase1_dura_orthogonal.benchmark_flow_epoch_times import epoch_train
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
    print('N607_EPOCH_TIMES_REAL_DURA_SMOKE_PASS', flush=True)
    for epoch_mode in (False, True):
        warm = epoch_train(small, 'conditional', 1) if epoch_mode else train(small, 'conditional', True, 1)
        del warm
    pairs = []
    for order in ((False, True), (True, False)):
        outcomes = {epoch_mode: epoch_train(selected, 'conditional', 5) if epoch_mode
                    else train(selected, 'conditional', True, 5) for epoch_mode in order}
        batch, epoch = outcomes[False], outcomes[True]
        match = exact(batch, epoch)
        pairs.append(dict(batch_seconds=batch[5], epoch_seconds=epoch[5], steps=len(batch[2]),
                          losses_parameters_optimizer_rng_exact=match,
                          max_loss_absolute_difference=float((batch[2]-epoch[2]).abs().max())))
        if not match:
            break
        del outcomes, batch, epoch
    args.out.mkdir(parents=True)
    result = dict(status='N607_EPOCH_TIMES_PARITY_COMPLETE' if all(p['losses_parameters_optimizer_rng_exact'] for p in pairs)
                  else 'N607_EPOCH_TIMES_PARITY_FAILED', commit=args.commit, source_ids=512,
                  views_per_id=3, epochs=5, batch=32, pairs=pairs, torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(), physical_gpu=args.gpu, physical_gpu_uuid=uuid,
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, weights_saved=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
