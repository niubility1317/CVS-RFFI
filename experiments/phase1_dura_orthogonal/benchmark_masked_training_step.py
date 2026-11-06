"""Compare two short, deterministic source-only training trajectories."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch

from experiments.phase1_dura_orthogonal.benchmark_masked_prior_loss import original
from experiments.phase1_dura_orthogonal.masked_prior_constraint import masked_prior_loss, SOURCE_COMMIT, WEIGHT, MASK_SEED
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior
from experiments.phase1_dura_orthogonal.train_source import source_loss


def extract(cache_path, teacher_path, out):
    if out.exists():
        raise FileExistsError('Unique source sample output required')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    contract = cache['contract']
    if (contract.get('schema') != 'phase1_fixed_dura_source_context_v1' or
            contract.get('source_epochs') != [21, 22, 23] or
            contract.get('deterministic_algorithms') is not True or
            any(contract.get(k) != 0 for k in ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads')) or
            teacher.get('commit') != SOURCE_COMMIT or len(contract['fit_ids']) != 32768 or
            not np.array_equal(np.asarray(contract['fit_ids']), np.asarray(teacher['fit_ids']))):
        raise ValueError('Matching permitted source context and source prior required')
    data = cache['data']
    if len(data) != 5 or any(len(x) != 98304 for x in data):
        raise ValueError('Full three-view source context required')
    subsets = tuple(torch.cat([v[i:i+64] for i in (0, 32768, 65536)]).contiguous() for v in data)
    if any(not torch.isfinite(v).all() for v in subsets) or (subsets[3] <= 0).any():
        raise ValueError('Finite source IQ context required')
    out.mkdir(parents=True)
    torch.save(dict(data=subsets, source_fit_ids=contract['fit_ids'][:64], views=[21, 22, 23],
                    source_prior_commit=SOURCE_COMMIT, TX_labels_read=0, F_weights_read=0,
                    F_outputs_read=0, V_reads=0, target_reads=0), out/'sample.pt')
    print('SOURCE_CONTEXT_SAMPLE_COMPLETE', len(subsets[0]), flush=True)


def train(prior, data, loss_fn):
    torch.manual_seed(2026110612)
    model = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_gen = torch.Generator().manual_seed(2026110613)
    mask_gen = torch.Generator().manual_seed(MASK_SEED)
    losses = []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for epoch in range(2):
        order = torch.randperm(len(data[0]), generator=order_gen).to('cuda:0')
        for offset in range(0, len(order), 32):
            indices = order[offset:offset+32]
            x, anchor, direction, variance, clean = (v[indices] for v in data)
            prediction = model(x, anchor, direction, variance, validate=False)
            starts = torch.randint(1, 224, (len(prediction),), generator=mask_gen).to('cuda:0')
            loss = source_loss(prediction, anchor, clean, objective='dura_guard') + WEIGHT*loss_fn(prior, prediction, clean, starts)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source IQ training loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-begin
    return model, losses, elapsed


def compare(sample_path, teacher_path, out, commit):
    if out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sample = torch.load(sample_path, map_location='cpu', weights_only=False)
    state = torch.load(teacher_path, map_location='cpu', weights_only=False)
    if (sample.get('source_prior_commit') != SOURCE_COMMIT or state.get('commit') != SOURCE_COMMIT or
            any(sample.get(k) != 0 for k in ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads')) or
            not np.array_equal(np.asarray(sample['source_fit_ids']), np.asarray(state['fit_ids'])[:64])):
        raise ValueError('Fixed source-only sample and prior provenance required')
    data = tuple(x.to('cuda:0') for x in sample['data'])
    if len(data) != 5 or any(len(x) != 192 for x in data):
        raise ValueError('192 source view records required')
    prior = MaskedIQPrior().to('cuda:0').eval().requires_grad_(False)
    prior.load_state_dict(state['model'], strict=True)
    # Real prior, source context, and recovery initialization are exercised before paired runs.
    smoke = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0').eval()
    with torch.no_grad():
        probe = smoke(*(x[:8] for x in data[:4]))
    if not torch.equal(probe, data[1][:8]):
        raise ValueError('Real source-context zero-head smoke required')
    del smoke
    print('MASKED_TRAINING_REAL_SOURCE_SMOKE_PASS', flush=True)
    warm, _, _ = train(prior, data, original)
    del warm
    warm, _, _ = train(prior, data, masked_prior_loss)
    del warm
    old_model, old_losses, old_time = train(prior, data, original)
    new_model, new_losses, new_time = train(prior, data, masked_prior_loss)
    old_state, new_state = old_model.state_dict(), new_model.state_dict()
    different = [key for key in old_state if not torch.equal(old_state[key], new_state[key])]
    maximum = max((float((old_state[key]-new_state[key]).abs().max().cpu()) for key in different), default=0.)
    outputs_equal = True
    with torch.no_grad():
        for start in range(0, len(data[0]), 32):
            a = old_model(*(x[start:start+32] for x in data[:4]), validate=False)
            b = new_model(*(x[start:start+32] for x in data[:4]), validate=False)
            outputs_equal &= torch.equal(a, b)
    out.mkdir(parents=True)
    result = dict(status='MASKED_TRAINING_SOURCE_PARITY_COMPLETE', commit=commit,
        source_prior_commit=SOURCE_COMMIT, source_ids=64, source_views=192, epochs=2, batch=32,
        warmup_trajectories_per_path=1,
        losses_exact=old_losses == new_losses, parameters_exact=not different,
        different_parameters=different, max_parameter_abs_difference=maximum,
        source_outputs_exact=outputs_equal, source_output_count=len(data[0]),
        elapsed_seconds={'original':old_time, 'optimized':new_time},
        device=torch.cuda.get_device_name(), TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, caveat='Short local real-source training check; excludes preparation, not full training or target prediction equality.')
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('MASKED_TRAINING_SOURCE_PARITY_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('extract', 'compare'))
    p.add_argument('--cache', type=Path)
    p.add_argument('--sample', type=Path)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit')
    args = p.parse_args()
    if args.mode == 'extract':
        if args.cache is None:
            p.error('--cache is required for extract')
        extract(args.cache, args.teacher, args.out)
    else:
        if args.sample is None or args.commit is None:
            p.error('--sample and --commit are required for compare')
        compare(args.sample, args.teacher, args.out, args.commit)


if __name__ == '__main__':
    main()
