"""F-free source training with simulator-action perturbation response preservation."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.counterfactual_response import choose_perturbation, response_loss, DIRECTIONS, EPSILON
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
from experiments.phase1_dura_orthogonal.train_source import load_source_training, extract_context
from experiments.phase1_joint_particle.train import _data, _loss
from experiments.phase1_cumulant_context.model import CumulantContextRecovery, cumulant_loss, CUMULANT_WEIGHT, LAGS
from experiments.phase1_cumulant_context.train import EPOCHS, BATCH, WIDTH, LAYERS, MODEL_SEED, ORDER_SEED

SCHEMA = 'phase1_cumulant_true_response_source_v1'
RESPONSE_WEIGHT = .01
SOURCE_EPOCHS = (21, 22, 23)


@torch.no_grad()
def prepare_true_counterfactual(root, dura, values, fit, smoke):
    """Match source simulator action to each cached observation; inference never uses it."""
    pilot, _, _, _, _, _ = source_inputs(root)
    source_cache, fresh, clean, expected_fit = load_source_training(root, pilot, fit_size=32768)
    if not np.array_equal(fit, expected_fit):
        raise ValueError('Same frozen source fit IDs required')
    if smoke:
        roles = np.load(source_cache / 'train/roles.npy', allow_pickle=False)
        role_fit = roles[fit]
        positions = np.concatenate((np.flatnonzero(role_fit == 'L_s')[:32],
                                    np.flatnonzero(role_fit == 'U_s')[:32]))
        if len(positions) != 64:
            raise ValueError('Mixed legal source roles required for smoke')
        fit = fit[positions]
    count = len(fit)
    if len(values[0]) != count * (1 if smoke else len(SOURCE_EPOCHS)):
        raise ValueError('Matching source views required')
    parts = []
    for view, epoch in enumerate(SOURCE_EPOCHS[:1] if smoke else SOURCE_EPOCHS):
        folder = fresh / 'epochs' / f'{epoch:02d}'
        metadata = core.read(folder / 'complete.json')
        if (metadata.get('count') != core.COUNT or
                metadata.get('namespace') != f'learned-action/train/epoch{epoch:02d}' or
                any(metadata.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
            raise ValueError('Matching legal source simulation cache required')
        received = np.load(folder / 'received.npy', mmap_mode='r', allow_pickle=False)
        kernel = np.load(folder / 'kernel.npy', mmap_mode='r', allow_pickle=False)
        coefficients = np.load(folder / 'coefficients.npy', mmap_mode='r', allow_pickle=False)
        if (received.shape != (core.COUNT, 2, 256) or kernel.shape != (core.COUNT, 2, 64)
                or coefficients.shape != (core.COUNT, 4, 256) or
                any(v.dtype != np.float32 for v in (received, kernel, coefficients))):
            raise ValueError('Complete matching simulator IQ and action arrays required')
        for start in range(0, count, BATCH):
            ids = fit[start:start+BATCH]
            offset = view * count + start
            x = values[0][offset:offset+len(ids)]
            y = values[4][offset:offset+len(ids)]
            if (not torch.equal(x, core.tensor(received[ids], x.device)) or
                    not torch.equal(y, core.tensor(clean[ids], x.device))):
                raise ValueError('Cached DURA context must match source physical view and clean IQ')
            h = core.tensor(kernel[ids], x.device)
            c = core.tensor(coefficients[ids], x.device)
            estimated = dura.estimate_context(x)[1]
            simulated = true_action(type(estimated), h, c)
            delta = choose_perturbation(y, offset)
            effect = simulated(delta)
            changed = x + torch.stack((effect.real, effect.imag), dim=1)
            anchor, direction, variance = extract_context(dura, changed)
            delta_iq = torch.stack((delta.real, delta.imag), dim=1)
            fields = (changed, anchor, direction, variance, delta_iq)
            if not all(torch.isfinite(v).all() for v in fields):
                raise FloatingPointError('Finite source counterfactual context required')
            parts.append(tuple(v.detach() for v in fields))
        print('TRUE_RESPONSE_SOURCE_PREP', epoch, count, flush=True)
    return tuple(torch.cat(parts, 0) for parts in zip(*parts))


def run(root, out, commit, cache, smoke=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    training_environments('cuda:0', os.environ)
    if out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique source output and selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(MODEL_SEED)
    values, fit, counts = _data(root, 'cuda:0', smoke, cache)
    dura = frozen_dura(root, 'cuda:0')
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen DURA required')
    changed = prepare_true_counterfactual(root, dura, values, fit, smoke)
    # Loading the frozen DURA constructs modules and consumes torch RNG; reset
    # immediately before creating the matched independently initialized model.
    torch.manual_seed(MODEL_SEED)
    model = CumulantContextRecovery(width=WIDTH, layers=LAYERS).to('cuda:0').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    rng = np.random.default_rng(ORDER_SEED)
    out.mkdir(parents=True)
    started = time.monotonic()
    losses, responses = [], []
    for epoch in range(1, 2 if smoke else EPOCHS+1):
        order = rng.permutation(len(values[0]))
        if smoke:
            order = np.tile(order, 3)
        total = torch.zeros((), device='cuda:0')
        response_total = torch.zeros((), device='cuda:0')
        for start in range(0, len(order), BATCH):
            idx = torch.tensor(order[start:start+BATCH].tolist(), dtype=torch.long, device='cuda:0')
            received, anchor, direction, variance, clean = (v[idx] for v in values)
            aug_received, aug_anchor, aug_direction, aug_variance, delta = (v[idx] for v in changed)
            output = model(received, anchor, direction, variance)
            augmented = model(aug_received, aug_anchor, aug_direction, aug_variance)
            response = response_loss(output, augmented, delta)
            characteristic = cumulant_loss(output, clean)
            loss = _loss(output, anchor, clean) + CUMULANT_WEIGHT*characteristic + RESPONSE_WEIGHT*response
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if start == 0 and (model.out.weight.grad is None or model.out.weight.grad.abs().sum() <= 0):
                raise ValueError('Finite nonzero source gradient required')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()*len(idx)
            response_total += response.detach()*len(idx)
        torch.cuda.synchronize()
        losses.append(float(total.cpu())/len(order))
        responses.append(float(response_total.cpu())/len(order))
        print('TRUE_RESPONSE_SOURCE_EPOCH', epoch, 'loss', losses[-1], 'response', responses[-1],
              'elapsed', round(time.monotonic()-started, 1), flush=True)
    common = dict(schema=SCHEMA, commit=commit, epochs=1 if smoke else EPOCHS, fit_ids=len(fit),
        actual_source_view_count=len(values[0]), views_per_record=3, width=WIDTH, layers=LAYERS,
        model_seed=MODEL_SEED, order_seed=ORDER_SEED, lags=list(LAGS), cumulant_weight=CUMULANT_WEIGHT,
        response_weight=RESPONSE_WEIGHT, response_epsilon=EPSILON, response_directions=list(DIRECTIONS),
        response_source_action='matched_source_simulator', simulator_action_inference_reads=0,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        source_loss='normalized IQ + .02 DURA change + .1 bounded moment + .01 true-action response',
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        DURA_frozen=True, source_counts=counts, source_counts_scope='complete_cached_three_views',
        elapsed_seconds=time.monotonic()-started, losses=losses, response_losses=responses)
    with (out/('smoke.pt' if smoke else 'final.pt')).open('xb') as f:
        torch.save(dict(**common, model=model.eval().cpu().state_dict()), f)
    core.write(out/('smoke.json' if smoke else 'source_summary.json'), dict(
        status='SOURCE_TRAINING_SMOKE_PASS' if smoke else 'SOURCE_TRAINED', **common))
    print('TRUE_RESPONSE_SOURCE_DONE', commit, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--device', choices=('cuda:0',), default='cuda:0')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(a.root, a.out, a.commit, a.cache, a.smoke)


if __name__ == '__main__':
    main()
