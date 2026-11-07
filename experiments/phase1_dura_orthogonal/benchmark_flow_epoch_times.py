"""Source-only exact parity and timing for once-per-epoch flow-time transfer."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import experiments
import torch

from experiments.phase1_dura_orthogonal.benchmark_flow_sync import exact, train
from experiments.phase1_dura_orthogonal.benchmark_masked_training_full import load_inputs
from experiments.phase1_dura_orthogonal.conditional_iq_flow import (
    ConditionalIQFlow, DirectIQFlow, flow_matching_loss, sample_epoch_flow_times)
from experiments.phase1_dura_orthogonal.fused_gradient_guard import checked_backward_and_clip
from experiments.phase1_dura_orthogonal.train_source import extract_context


def epoch_train(fields, mode, epochs):
    torch.manual_seed(2026110612)
    model = (DirectIQFlow if mode == 'direct' else ConditionalIQFlow)(width=32, steps=4).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_gen = torch.Generator().manual_seed(2026110613)
    time_gen = torch.Generator().manual_seed(2026110713)
    losses = []
    torch.cuda.synchronize()
    started = time.monotonic()
    for epoch in range(epochs):
        order = torch.randperm(len(fields[0]), generator=order_gen).cuda()
        times = sample_epoch_flow_times(len(order), time_gen, origin_fraction=.5 if mode == 'mixed' else 0.).cuda()
        for start in range(0, len(order), 32):
            x, anchor, direction, variance, clean = (v[order[start:start+32]] for v in fields)
            if mode == 'direct':
                anchor, direction, variance = model.direct_context(x)
            loss = flow_matching_loss(model, x, anchor, direction, variance, clean, times[start:start+32], validate_time=False)
            optimizer.zero_grad(set_to_none=True)
            checked_backward_and_clip(loss, model, (model.out.weight,))
            optimizer.step()
            losses.append(loss.detach())
        print('FLOW_EPOCH_TIMES_EPOCH', mode, epoch+1, flush=True)
    torch.cuda.synchronize()
    return model, optimizer, torch.stack(losses).cpu(), order_gen.get_state(), time_gen.get_state(), time.monotonic()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--initial-commit', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-count', type=int, choices=(512, 32768), default=512)
    parser.add_argument('--pairs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--modes', nargs='+', choices=('conditional', 'mixed', 'direct'),
                        default=('conditional', 'mixed', 'direct'))
    args = parser.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    experiments.__path__ = [str(args.root/'releases/pamf_stage2_inference_20261021_r01/experiments'), *experiments.__path__]
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    data, prior, initial = load_inputs(args.cache, args.teacher, args.initial_state, args.initial_commit)
    del prior, initial
    small = tuple(torch.cat([v[i:i+64] for i in (0, 32768, 65536)]) for v in data)
    selected = tuple(torch.cat([v[i:i+args.source_count] for i in (0, 32768, 65536)]) for v in data)
    dura = frozen_dura(args.root, 'cuda:0')
    with torch.no_grad():
        context = extract_context(dura, selected[0][:8])
        if not all(torch.isfinite(v).all() for v in context) or any(p.requires_grad for p in dura.parameters()):
            raise ValueError('Frozen real DURA source smoke required')
    del dura, data
    print('FLOW_EPOCH_TIMES_REAL_DURA_SMOKE_PASS', flush=True)
    results = {}
    failed = False
    for mode in args.modes:
        for epoch_mode in (False, True):
            warm = epoch_train(small, mode, 1) if epoch_mode else train(small, mode, True, 1)
            del warm
        results[mode] = []
        for order in ((False, True), (True, False))[:args.pairs]:
            outcomes = {epoch_mode: epoch_train(selected, mode, 5) if epoch_mode else train(selected, mode, True, 5)
                        for epoch_mode in order}
            batch, epoch = outcomes[False], outcomes[True]
            match = exact(batch, epoch)
            results[mode].append(dict(batch_seconds=batch[5], epoch_seconds=epoch[5], steps=len(batch[2]),
                                      losses_parameters_optimizer_rng_exact=match,
                                      max_loss_absolute_difference=float((batch[2]-epoch[2]).abs().max())))
            if not match:
                failed = True
                break
            del outcomes, batch, epoch
        if failed:
            break
    args.out.mkdir(parents=True)
    result = dict(status='FLOW_EPOCH_TIMES_PARITY_FAILED' if failed else 'FLOW_EPOCH_TIMES_COMPLETE',
                  commit=args.commit, source_ids=args.source_count, views_per_id=3, epochs=5,
                  batch=32, results=results, gpu=torch.cuda.get_device_name(), TX_labels_read=0,
                  F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, weights_saved=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
