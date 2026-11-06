"""Full-budget, source-only paired training parity on one local GPU."""
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
from experiments.phase1_dura_orthogonal.train_source import source_loss, load_source_training, WIDE_FIT_SIZE


def capture_initial(root, out, commit):
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
    if out.exists():
        raise FileExistsError('Unique source initialization output required')
    torch.set_num_threads(2)
    torch.manual_seed(2026110612)
    ids, _, _, _, _, _ = source_inputs(root)
    _, _, _, fit = load_source_training(root, ids, WIDE_FIT_SIZE)
    dura = frozen_dura(root, 'cpu')
    model = DuraMultiscaleSpectral(width=32, taps=9)
    out.mkdir(parents=True)
    with (out/'initial.pt').open('xb') as stream:
        torch.save(dict(model=model.state_dict(), source_fit_ids=fit, seed=2026110612,
            source_dura='dura_v1_R65_E20_4940', commit=commit, TX_labels_read=0,
            F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0), stream)
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Original DURA must remain frozen')
    print('MASKED_FULL_INITIALIZATION_CAPTURED', len(fit), flush=True)


def load_inputs(cache_path, prior_path, initial_path, commit):
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    prior_state = torch.load(prior_path, map_location='cpu', weights_only=False)
    initial = torch.load(initial_path, map_location='cpu', weights_only=False)
    contract = cache['contract']
    if (contract.get('schema') != 'phase1_fixed_dura_source_context_v1' or
            contract.get('source_epochs') != [21, 22, 23] or
            contract.get('batch_size') != 32 or contract.get('precision') != 'FP32_TF32off' or
            contract.get('dura') != 'dura_v1_R65_E20_4940' or
            contract.get('deterministic_algorithms') is not True or
            any(contract.get(k) != 0 for k in ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads')) or
            prior_state.get('commit') != SOURCE_COMMIT or len(contract['fit_ids']) != 32768 or
            not np.array_equal(np.asarray(contract['fit_ids']), np.asarray(prior_state['fit_ids'])) or
            initial.get('commit') != commit or initial.get('seed') != 2026110612 or
            initial.get('source_dura') != 'dura_v1_R65_E20_4940' or
            any(initial.get(k) != 0 for k in ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads')) or
            not np.array_equal(np.asarray(contract['fit_ids']), np.asarray(initial['source_fit_ids']))):
        raise ValueError('Matching full source-only context and source prior required')
    data = cache['data']
    expected_shapes = ((98304, 2, 256), (98304, 2, 256), (98304, 2, 256), (98304,), (98304, 2, 256))
    if (len(data) != 5 or any(v.dtype != torch.float32 or tuple(v.shape) != shape for v, shape in zip(data, expected_shapes)) or
            sum(cache['counts'].values()) != 98304):
        raise ValueError('Complete source context tensors required')
    prior = MaskedIQPrior().to('cuda:0').eval().requires_grad_(False)
    prior.load_state_dict(prior_state['model'], strict=True)
    return tuple(v.to('cuda:0') for v in data), prior, initial['model']


def train(label, fn, prior, data, initial, epochs):
    torch.manual_seed(2026110612)
    model = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0')
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_generator = torch.Generator().manual_seed(2026110613)
    mask_generator = torch.Generator().manual_seed(MASK_SEED)
    step_losses, epoch_losses = [], []
    torch.cuda.synchronize()
    started = time.monotonic()
    for epoch in range(1, epochs+1):
        order = torch.randperm(len(data[0]), generator=order_generator).to('cuda:0')
        total = torch.zeros((), device='cuda:0')
        for start in range(0, len(order), 32):
            index = order[start:start+32]
            x, anchor, innovation, variance, clean = (v[index] for v in data)
            output = model(x, anchor, innovation, variance, validate=False)
            masks = torch.randint(1, 224, (len(output),), generator=mask_generator).to('cuda:0')
            loss = source_loss(output, anchor, clean, objective='dura_guard') + WEIGHT*fn(prior, output, clean, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if (model.wave.weight.grad is None or model.filter_head.weight.grad is None or
                    not torch.isfinite(model.wave.weight.grad).all() or not torch.isfinite(model.filter_head.weight.grad).all()):
                raise ValueError('Finite correction gradients required')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()*len(index)
            step_losses.append(loss.detach())
        epoch_losses.append(float(total.cpu())/len(order))
        print('MASKED_FULL_EPOCH', label, epoch, 'loss', epoch_losses[-1],
              'elapsed', round(time.monotonic()-started, 2), flush=True)
    torch.cuda.synchronize()
    return model, torch.stack(step_losses).cpu(), epoch_losses, time.monotonic()-started


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--initial-state', type=Path)
    p.add_argument('--capture-root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.capture_root:
        capture_initial(args.capture_root, args.out, args.commit)
        return
    if args.cache is None or args.teacher is None or args.initial_state is None:
        p.error('--cache, --teacher and --initial-state are required for comparison')
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data, prior, initial = load_inputs(args.cache, args.teacher, args.initial_state, args.commit)
    torch.manual_seed(2026110612)
    smoke = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0').eval()
    with torch.no_grad():
        if not torch.equal(smoke(*(x[:8] for x in data[:4])), data[1][:8]):
            raise ValueError('Real source-context zero-head smoke required')
    del smoke
    print('MASKED_FULL_REAL_SOURCE_SMOKE_PASS', flush=True)
    args.out.mkdir(parents=True)
    small = tuple(torch.cat([v[i:i+64] for i in (0, 32768, 65536)]) for v in data)
    warm, _, _, _ = train('warm_original', original, prior, small, initial, 1)
    del warm
    warm, _, _, _ = train('warm_optimized', masked_prior_loss, prior, small, initial, 1)
    del warm, small
    old, old_steps, old_epochs, old_seconds = train('original', original, prior, data, initial, 5)
    new, new_steps, new_epochs, new_seconds = train('optimized', masked_prior_loss, prior, data, initial, 5)
    old_state, new_state = old.state_dict(), new.state_dict()
    different = [key for key in old_state if not torch.equal(old_state[key], new_state[key])]
    maximum = max((float((old_state[key]-new_state[key]).abs().max().cpu()) for key in different), default=0.)
    old.eval(); new.eval()
    outputs_equal = True
    with torch.no_grad():
        for offset in (0, 32768, 65536):
            for start in (offset, offset+32):
                batch = tuple(v[start:start+32] for v in data[:4])
                outputs_equal &= torch.equal(old(*batch, validate=False), new(*batch, validate=False))
    with (args.out/'original.pt').open('xb') as f:
        torch.save(old_state, f)
    with (args.out/'optimized.pt').open('xb') as f:
        torch.save(new_state, f)
    result = dict(status='MASKED_FULL_SOURCE_TRAINING_PARITY_COMPLETE', commit=args.commit,
        source_prior_commit=SOURCE_COMMIT, source_ids=32768, source_views=98304, epochs=5, batch=32,
        warmup_source_views=192,
        steps=len(old_steps), step_losses_exact=torch.equal(old_steps, new_steps),
        epoch_losses_exact=old_epochs == new_epochs, parameters_exact=not different,
        different_parameters=different, max_parameter_abs_difference=maximum,
        source_outputs_exact=outputs_equal, source_output_count=192,
        elapsed_seconds={'original':old_seconds, 'optimized':new_seconds},
        device=torch.cuda.get_device_name(), TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0,
        caveat='Recreated N607 initialization, full source-fit budget on one local GPU; fixed order, no target predictions, not N607 GPU parity.')
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('MASKED_FULL_SOURCE_TRAINING_PARITY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
