"""Source-only, paired flow training synchronization benchmark; saves no model."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import experiments

from experiments.phase1_dura_orthogonal.benchmark_masked_training_full import load_inputs
from experiments.phase1_dura_orthogonal.conditional_iq_flow import (
    ConditionalIQFlow, DirectIQFlow, flow_matching_loss, sample_flow_times)
from experiments.phase1_dura_orthogonal.fused_gradient_guard import checked_backward_and_clip
from experiments.phase1_dura_orthogonal.train_source import extract_context


def train(fields, mode, fast, epochs):
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
        for start in range(0, len(order), 32):
            x, anchor, direction, variance, clean = (v[order[start:start+32]] for v in fields)
            times = (sample_flow_times(len(x), time_gen, origin_fraction=.5)
                     if mode == 'mixed' else torch.rand((len(x),), generator=time_gen)).cuda()
            if mode == 'direct':
                anchor, direction, variance = model.direct_context(x)
            loss = flow_matching_loss(model, x, anchor, direction, variance, clean, times,
                                      validate_time=not fast)
            if not fast and not torch.isfinite(loss):
                raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True)
            if fast:
                checked_backward_and_clip(loss, model, (model.out.weight,))
            else:
                loss.backward()
                if model.out.weight.grad is None or not torch.isfinite(model.out.weight.grad).all():
                    raise ValueError('Finite output gradient required')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        print('FLOW_SYNC_EPOCH', mode, 'fast' if fast else 'original', epoch+1, flush=True)
    torch.cuda.synchronize()
    return model, optimizer, torch.stack(losses).cpu(), order_gen.get_state(), time_gen.get_state(), time.monotonic()-started


def exact(a, b):
    if not torch.equal(a[2], b[2]) or not torch.equal(a[3], b[3]) or not torch.equal(a[4], b[4]):
        return False
    if any(not torch.equal(a[0].state_dict()[k], b[0].state_dict()[k]) for k in a[0].state_dict()):
        return False
    left, right = a[1].state_dict(), b[1].state_dict()
    if left['param_groups'] != right['param_groups'] or left['state'].keys() != right['state'].keys():
        return False
    for key in left['state']:
        if left['state'][key].keys() != right['state'][key].keys():
            return False
        for name, value in left['state'][key].items():
            other = right['state'][key][name]
            if not (torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other):
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--initial-commit', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
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
    selected = tuple(torch.cat([v[i:i+512] for i in (0, 32768, 65536)]) for v in data)
    small = tuple(torch.cat([v[i:i+64] for i in (0, 32768, 65536)]) for v in data)
    dura = frozen_dura(args.root, 'cuda:0')
    with torch.no_grad():
        x = selected[0][:8]
        anchor, direction, variance = extract_context(dura, x)
        model = ConditionalIQFlow(width=32, steps=4).cuda()
        if not all(torch.isfinite(v).all() for v in (anchor, direction, variance)) or not torch.equal(model(x, anchor, direction, variance), anchor):
            raise ValueError('Real DURA source smoke failed')
    if any(p.requires_grad or p.grad is not None for p in dura.parameters()):
        raise ValueError('DURA must remain frozen')
    del dura, model, data
    print('FLOW_SYNC_REAL_DURA_SMOKE_PASS', flush=True)
    results = {}
    for mode in ('conditional', 'mixed', 'direct'):
        for fast in (False, True):
            del_result = train(small, mode, fast, 1)
            del del_result
        results[mode] = []
        for order in ((False, True), (True, False)):
            outcomes = {fast: train(selected, mode, fast, 5) for fast in order}
            a, b = outcomes[False], outcomes[True]
            if not exact(a, b):
                raise ValueError('Flow training trajectories differ')
            results[mode].append(dict(original_seconds=a[5], fast_seconds=b[5], steps=len(a[2]),
                                      losses_parameters_optimizer_rng_exact=True))
            del outcomes, a, b
    args.out.mkdir(parents=True)
    result = dict(status='FLOW_SYNC_BENCHMARK_COMPLETE', commit=args.commit, source_ids=512,
                  views_per_id=3, epochs=5, batch=32, results=results,
                  gpu=torch.cuda.get_device_name(), TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0, weights_saved=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('FLOW_SYNC_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
