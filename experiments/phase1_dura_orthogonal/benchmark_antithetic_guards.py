"""Source-only GPU parity and timing for fused antithetic-flow guards."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch

from experiments.phase1_dura_orthogonal.antithetic_iq_flow import AntitheticIQFlow, TRAIN_LATENT_SEED, stochastic_flow_loss
from experiments.phase1_dura_orthogonal.conditional_iq_flow import sample_epoch_flow_times
from experiments.phase1_dura_orthogonal.fused_gradient_guard import checked_backward_and_clip
from experiments.phase1_dura_orthogonal.source_context_cache import _contract, _validate_data

INITIAL_COMMIT = '114a169dd94c358031c0aa9237df884d41f2678e'
FIT_SIZE = 32768


def load_context(cache_path, identity_path, source_count):
    initial = torch.load(identity_path, map_location='cpu', weights_only=False)
    fit = np.asarray(initial.get('source_fit_ids', []))
    if (initial.get('commit') != INITIAL_COMMIT or initial.get('seed') != 2026110612
            or initial.get('source_dura') != 'dura_v1_R65_E20_4940' or fit.shape != (FIT_SIZE,)
            or len(np.unique(fit)) != FIT_SIZE or any(initial.get(k) != 0 for k in
                ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads'))):
        raise ValueError('Fixed source fit identity required')
    payload = torch.load(cache_path, map_location='cpu', weights_only=True)
    if payload.get('contract') != _contract(fit, 32, 'cuda:0', deterministic=True):
        raise ValueError('Matching legal frozen DURA source context required')
    data = payload['data']
    _validate_data(data, payload['counts'], 3*FIT_SIZE)
    return tuple(torch.cat([v[i:i+source_count] for i in (0, FIT_SIZE, 2*FIT_SIZE)]).to('cuda:0') for v in data)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def train(fields, optimized, epochs):
    torch.manual_seed(2026110612)
    model = AntitheticIQFlow(width=32).to('cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_gen = torch.Generator().manual_seed(2026110613)
    time_gen = torch.Generator().manual_seed(2026110713)
    latent_gen = torch.Generator().manual_seed(TRAIN_LATENT_SEED)
    losses = []
    torch.cuda.synchronize()
    started = time.monotonic()
    for epoch in range(epochs):
        order = torch.randperm(len(fields[0]), generator=order_gen).to('cuda:0')
        times = sample_epoch_flow_times(len(order), time_gen, origin_fraction=0.).to('cuda:0')
        latents = torch.randn((len(order), 2, 256), generator=latent_gen).to('cuda:0')
        for start in range(0, len(order), 32):
            index = order[start:start+32]
            x, anchor, direction, variance, clean = (v[index] for v in fields)
            loss = stochastic_flow_loss(model, x, anchor, direction, variance, clean,
                times[start:start+len(x)], latents[start:start+len(x)], validate_time=not optimized)
            optimizer.zero_grad(set_to_none=True)
            if optimized:
                checked_backward_and_clip(loss, model, (model.out.weight,))
            else:
                if not torch.isfinite(loss):
                    raise FloatingPointError('Finite source loss required')
                loss.backward()
                if model.out.weight.grad is None or not torch.isfinite(model.out.weight.grad).all():
                    raise ValueError('Finite flow output gradient required')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        print('ANTITHETIC_GUARD_EPOCH', 'optimized' if optimized else 'original', epoch+1, flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(), order_gen.get_state(), time_gen.get_state(), latent_gen.get_state())
    return model, optimizer.state_dict(), torch.stack(losses).cpu(), rng, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--source-count', type=int, choices=(512, 32768), default=512)
    p.add_argument('--pairs', type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and local GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data = load_context(args.cache, args.identity, args.source_count)
    torch.manual_seed(2026110612)
    smoke = AntitheticIQFlow(width=32).to('cuda:0').eval()
    with torch.no_grad():
        if not torch.equal(smoke(*(v[:8] for v in data[:4])), data[1][:8]):
            raise ValueError('Real frozen source context zero-head smoke required')
    del smoke
    print('ANTITHETIC_GUARD_REAL_SOURCE_SMOKE_PASS', flush=True)
    small = tuple(v[:96] for v in data)
    for optimized in (False, True):
        warm = train(small, optimized, 1)
        del warm
    results = []
    for order in ((False, True), (True, False))[:args.pairs]:
        outcomes = {optimized: train(data, optimized, 5) for optimized in order}
        a, b = outcomes[False], outcomes[True]
        exact = dict(step_losses=equal_tree(a[2], b[2]), parameters=equal_tree(a[0].state_dict(), b[0].state_dict()),
                     optimizer=equal_tree(a[1], b[1]), rng=equal_tree(a[3], b[3]))
        with torch.no_grad():
            exact['source_outputs'] = torch.equal(a[0](*(v[:8] for v in data[:4])), b[0](*(v[:8] for v in data[:4])))
        if not all(exact.values()):
            raise ValueError('Source optimization trajectories differ: '+repr(exact))
        results.append(dict(optimized_first=order[0], original_seconds=a[4], optimized_seconds=b[4],
                            steps=len(a[2]), exact=exact))
        del outcomes, a, b
    args.out.mkdir(parents=True)
    result = dict(status='ANTITHETIC_GUARD_BENCHMARK_COMPLETE', commit=args.commit,
        source_ids=args.source_count, views_per_id=3, epochs=5, batch=32, results=results,
        gpu=torch.cuda.get_device_name(), TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, weights_saved=False,
        caveat='Fixed local initialization; no claim of N607 training parity or target-wide prediction equality.')
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('ANTITHETIC_GUARD_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
