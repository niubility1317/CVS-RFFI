"""Paired source-only CUDA graph and eager flow training time/parity diagnostic."""
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
from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow, sample_epoch_flow_times
from experiments.phase1_dura_orthogonal.flow_cuda_graph import FlowGraphStep
from experiments.phase1_dura_orthogonal.train_source import extract_context


def graph_train(fields, epochs, *, epoch_times=False):
    torch.manual_seed(2026110612)
    model = ConditionalIQFlow(width=32, steps=4).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_gen = torch.Generator().manual_seed(2026110613)
    time_gen = torch.Generator().manual_seed(2026110713)
    count = (len(fields[0])//32)*epochs
    history = torch.empty(count, device='cuda:0')
    torch.cuda.synchronize()
    started = time.monotonic()
    runner = FlowGraphStep(model, optimizer, [*(v[:32] for v in fields), torch.zeros(32, device='cuda:0')])
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(len(fields[0]), generator=order_gen).cuda()
        all_times = (sample_epoch_flow_times(len(order), time_gen, origin_fraction=0.).cuda()
                     if epoch_times else None)
        for start in range(0, len(order), 32):
            index = order[start:start+32]
            batch = [v[index] for v in fields]
            times = (all_times[start:start+len(index)] if all_times is not None
                     else torch.rand((len(index),), generator=time_gen).cuda())
            history[step].copy_(runner.step([*batch, times]))
            step += 1
        print('FLOW_GRAPH_EPOCH', epoch+1, flush=True)
    torch.cuda.synchronize()
    return model, optimizer, history.cpu(), order_gen.get_state(), time_gen.get_state(), time.monotonic()-started


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
    parser.add_argument('--epoch-times', action='store_true')
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
        anchor, direction, variance = extract_context(dura, selected[0][:8])
        if not all(torch.isfinite(v).all() for v in (anchor, direction, variance)) or any(p.requires_grad for p in dura.parameters()):
            raise ValueError('Frozen real DURA source smoke required')
    del dura, data
    print('FLOW_GRAPH_REAL_DURA_SMOKE_PASS', flush=True)
    for graph in (False, True):
        warm = graph_train(small, 1, epoch_times=args.epoch_times) if graph else train(small, 'conditional', True, 1)
        del warm
    pairs = []
    for order in ((False, True), (True, False))[:args.pairs]:
        outcomes = {graph: graph_train(selected, 5, epoch_times=args.epoch_times) if graph else train(selected, 'conditional', True, 5) for graph in order}
        eager, captured = outcomes[False], outcomes[True]
        match = exact(eager, captured)
        pairs.append(dict(eager_seconds=eager[5], graph_seconds=captured[5], steps=len(eager[2]),
                          losses_parameters_optimizer_rng_exact=match,
                          max_loss_absolute_difference=float((eager[2]-captured[2]).abs().max())))
        if not match:
            break
        del outcomes, eager, captured
    args.out.mkdir(parents=True)
    result = dict(status='FLOW_GRAPH_BENCHMARK_COMPLETE' if all(p['losses_parameters_optimizer_rng_exact'] for p in pairs)
                  else 'FLOW_GRAPH_PARITY_FAILED', commit=args.commit, source_ids=args.source_count,
                  views_per_id=3, epochs=5, batch=32, pairs=pairs, graph_capture_included_in_timing=True,
                  graph_time_transfer='epoch' if args.epoch_times else 'batch',
                  gpu=torch.cuda.get_device_name(), TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0, weights_saved=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
