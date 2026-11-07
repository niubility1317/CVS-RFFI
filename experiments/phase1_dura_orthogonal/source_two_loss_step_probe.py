"""Source-only finite-step AdamW versus two-loss projected-update diagnostic."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
import experiments
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_orthogonal.counterfactual_response import response_loss
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.source_context_cache import _contract, _validate_data
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import perturbations, EPSILON
from experiments.phase1_dura_orthogonal.train_source import extract_context, source_loss
from experiments.phase1_dura_orthogonal.two_loss_projection import project_two_descent_halfspaces

RESPONSE_WEIGHT = .01
LEARNING_RATES = (.0003, .00003)


@torch.no_grad()
def prepare(dura, data, fit_count, start, count):
    """Re-estimate both contexts for fixed, disjoint source-ID slices."""
    batches = []
    for offset in (0, fit_count, 2*fit_count):
        x = data[0][offset+start:offset+start+count].to('cuda:0')
        clean = data[4][offset+start:offset+start+count].to('cuda:0')
        anchor, direction, variance = extract_context(dura, x)
        delta = perturbations(clean)[1]
        received_delta = dura.estimate_context(x)[1](delta)
        changed = x + torch.stack((received_delta.real, received_delta.imag), 1)
        changed_anchor, changed_direction, changed_variance = extract_context(dura, changed)
        delta_iq = torch.stack((delta.real, delta.imag), 1)
        batches.append((x, anchor, direction, variance, clean, changed,
                        changed_anchor, changed_direction, changed_variance, delta_iq))
    return batches


def losses(model, batches):
    base, response = [], []
    for x, anchor, direction, variance, clean, changed, changed_anchor, changed_direction, changed_variance, delta in batches:
        output = model(x, anchor, direction, variance, validate=False)
        changed_output = model(changed, changed_anchor, changed_direction, changed_variance, validate=False)
        base.append(source_loss(output, anchor, clean, objective='dura_guard'))
        response.append(response_loss(output, changed_output, delta))
    return torch.stack(base).mean(), torch.stack(response).mean()


def gradient_vector(loss, params):
    parts = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, parts)]).detach()


def as_pair(values):
    return dict(base=float(values[0]), response=float(values[1]))


def directional_cosine(gradient, direction):
    denom = gradient.double().norm()*direction.double().norm()
    return float(gradient.double()@direction.double()/denom) if denom > 0 else 0.


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--initial-state', type=Path, required=True)
    p.add_argument('--initial-commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    payload = torch.load(args.cache, map_location='cpu', weights_only=True)
    fit = payload['contract']['fit_ids']
    if len(fit) != 32768 or payload['contract'] != _contract(fit, 32, 'cuda:0', deterministic=True):
        raise ValueError('Matching fixed source context required')
    _validate_data(payload['data'], payload['counts'], 3*len(fit))
    state = torch.load(args.initial_state, map_location='cpu', weights_only=False)
    if (state.get('commit') != args.initial_commit or state.get('seed') != 2026110612
            or state.get('source_dura') != 'dura_v1_R65_E20_4940'
            or not np.array_equal(np.asarray(state.get('source_fit_ids', [])), np.asarray(fit))
            or any(state.get(k) != 0 for k in ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads'))):
        raise ValueError('Matching source-only initialization required')
    experiments.__path__ = [str(args.root/'releases/pamf_stage2_inference_20261021_r01/experiments'), *experiments.__path__]
    dura = frozen_dura(args.root, 'cuda:0')
    count = 2 if args.smoke else 32
    started = time.monotonic()
    train = prepare(dura, payload['data'], len(fit), 0, count)
    heldout = prepare(dura, payload['data'], len(fit), 32, count)
    del payload
    model = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0')
    model.load_state_dict(state['model'], strict=True)
    model.train()
    params = tuple(model.parameters())
    initial = parameters_to_vector(params).detach().clone()
    base, response = losses(model, train)
    a, b = gradient_vector(base, params), gradient_vector(response, params)
    if not all(torch.isfinite(x).all() for x in (base, response, a, b)):
        raise FloatingPointError('Finite source losses and gradients required')
    with torch.no_grad():
        before = dict(train=as_pair((base, response)), heldout=as_pair(losses(model, heldout)))
    del base, response
    results = []
    for lr in LEARNING_RATES:
        vector_to_parameters(initial.clone(), params)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=.001)
        optimizer.zero_grad(set_to_none=True)
        base, response = losses(model, train)
        (base + RESPONSE_WEIGHT*response).backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        adam_params = parameters_to_vector(params).detach().clone()
        direction = initial-adam_params
        projected = project_two_descent_halfspaces(direction, a, b)
        with torch.no_grad():
            adam = dict(train=as_pair(losses(model, train)), heldout=as_pair(losses(model, heldout)))
            vector_to_parameters(initial-projected, params)
            realized = initial-parameters_to_vector(params).detach()
            constrained = dict(train=as_pair(losses(model, train)), heldout=as_pair(losses(model, heldout)))
        results.append(dict(lr=lr, adam=adam, projected=constrained,
            gradient_cosines=dict(adam_base=directional_cosine(a, direction),
                adam_response=directional_cosine(b, direction),
                projected_base=directional_cosine(a, projected),
                projected_response=directional_cosine(b, projected),
                realized_base=directional_cosine(a, realized),
                realized_response=directional_cosine(b, realized)),
            projected_fraction_of_adam_norm=float(projected.double().norm()/direction.double().norm().clamp_min(1e-30))))
    if any(p.grad is not None or p.requires_grad for p in dura.parameters()):
        raise ValueError('Original DURA must remain frozen')
    torch.cuda.synchronize()
    result = dict(status='SOURCE_TWO_LOSS_STEP_SMOKE_PASS' if args.smoke else 'SOURCE_TWO_LOSS_STEP_COMPLETE',
        commit=args.commit, source_train_ids=count, source_heldout_ids=count, views_per_id=3,
        perturbation='amplitude_cubic', epsilon_relative_rms=EPSILON, response_weight=RESPONSE_WEIGHT,
        source_initialization_commit=args.initial_commit, before=before, results=results,
        elapsed_seconds=time.monotonic()-started, device=torch.cuda.get_device_name(),
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        optimizer_steps_in_memory=len(LEARNING_RATES), saved_model=False,
        caveat='Single fresh AdamW step and synthetic source response only; projection is first order, not a finite-step or heldout guarantee, and not a trained recovery version.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
