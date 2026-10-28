"""Frozen six-class source geometry and explicitly supervised source-OE readouts.

No IO, V inputs, thresholds, encoder or identity-head updates live here. The
binary readouts use source membership supervision, not OE transmitter labels.
"""
from __future__ import annotations

import numpy as np

from .branch_export import DIMENSIONS


CLASS_COUNT = 6
DESCRIPTOR_DIM = 100
METHODS = ('linear_oe', 'mlp16_oe')
GEOMETRY_SCHEMA = 'phase3_source_open_geometry_v1'
READOUT_SCHEMA = 'phase3_source_oe_readouts_v1'
STEPS = 200
OPTIMIZER = dict(name='AdamW', lr=.001, weight_decay=.0001, betas=[.9, .999], eps=1e-8)


def _matrix(value, name, columns, *, fp32=False):
    array = np.asarray(value)
    if (array.ndim != 2 or not len(array) or array.shape[1] != columns
            or array.dtype.kind != 'f' or (fp32 and array.dtype != np.float32)
            or not np.isfinite(array).all()):
        raise ValueError('Nonempty finite floating matrix required: ' + name)
    return array.astype(np.float64, copy=False)


def _branches(branches):
    if not isinstance(branches, dict) or set(branches) != set(DIMENSIONS):
        raise ValueError('Exactly the eight declared raw branches are required')
    values = {name: _matrix(branches[name], name, dimension, fp32=True)
              for name, dimension in DIMENSIONS.items()}
    if len({len(value) for value in values.values()}) != 1:
        raise ValueError('Raw branch row counts differ')
    if np.any(values['reference29'][:, 27] != 0):
        raise ValueError('NoExplicitCFO reference coordinate 27 must remain zero')
    return values


def _unit(value):
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def _array(value, shape, name, *, fp32=False):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind != 'f'
            or (fp32 and array.dtype != np.float32) or not np.isfinite(array).all()):
        raise ValueError('Finite aggregate shape/dtype differs: ' + name)
    return array


def fit_geometry(branches, y):
    """Fit only L statistics; all six source classes must be present in L."""
    values = _branches(branches)
    count = len(values['fused160'])
    labels = np.asarray(y)
    if (labels.shape != (count,) or labels.dtype.kind not in 'iu' or count <= CLASS_COUNT
            or set(np.unique(labels).tolist()) != set(range(CLASS_COUNT))):
        raise ValueError('Visible source L labels must contain exactly all six classes with n>6')
    fitted = {}
    for name, dimension in DIMENSIONS.items():
        raw = values[name]
        mean = np.zeros(dimension) if name == 'fused160' else raw.mean(axis=0)
        scale = np.ones(dimension) if name == 'fused160' else raw.std(axis=0, ddof=0)
        scale = np.where(scale <= 1e-8, 1., scale)
        unit = _unit((raw - mean) / scale)
        # Zero means/prototypes are valid for a completely masked or constant branch.
        prototypes = _unit(np.asarray([unit[labels == label].mean(axis=0) for label in range(CLASS_COUNT)]))
        residual = unit - prototypes[labels]
        covariance = residual.T @ residual / (count - CLASS_COUNT)
        trace_scale = max(float(np.trace(covariance) / dimension), 1e-8)
        regularized = .8 * covariance + .2 * trace_scale * np.eye(dimension)
        try:
            lower = np.linalg.cholesky(regularized)
            whitener = np.linalg.solve(lower, np.eye(dimension))
        except np.linalg.LinAlgError as exc:
            raise ValueError('Source shared covariance is not numerically positive definite: ' + name) from exc
        if not np.isfinite(whitener).all():
            raise ValueError('Nonfinite source whitener: ' + name)
        fitted[name] = dict(mean=mean, scale=scale, prototypes=prototypes, whitener=whitener)
    return dict(schema=GEOMETRY_SCHEMA, class_count=CLASS_COUNT, branches=fitted)


def _geometry(bundle):
    if (not isinstance(bundle, dict) or set(bundle) != {'schema', 'class_count', 'branches'}
            or bundle['schema'] != GEOMETRY_SCHEMA or type(bundle['class_count']) is not int
            or bundle['class_count'] != CLASS_COUNT or not isinstance(bundle['branches'], dict)
            or set(bundle['branches']) != set(DIMENSIONS)):
        raise ValueError('Frozen six-class geometry schema differs')
    for name, dimension in DIMENSIONS.items():
        branch = bundle['branches'][name]
        if not isinstance(branch, dict) or set(branch) != {'mean', 'scale', 'prototypes', 'whitener'}:
            raise ValueError('Only four source aggregate arrays per branch are accepted')
        _array(branch['mean'], (dimension,), name + ' mean')
        scale = _array(branch['scale'], (dimension,), name + ' scale')
        _array(branch['prototypes'], (CLASS_COUNT, dimension), name + ' prototypes')
        _array(branch['whitener'], (dimension, dimension), name + ' whitener')
        if np.any(scale <= 0):
            raise ValueError('Source normalization scales must be positive')
    reference = bundle['branches']['reference29']
    if (reference['mean'][27] != 0 or reference['scale'][27] != 1
            or np.any(reference['prototypes'][:, 27] != 0)):
        raise ValueError('Frozen NoExplicitCFO coordinate differs')
    return bundle['branches']


def _fp32(value, name):
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite FP32 result: ' + name)
    return result


def descriptor(branches, logits6, bundle):
    """Return class-order-invariant, row-local [N,100] knownness descriptors."""
    values = _branches(branches)
    fitted = _geometry(bundle)
    logits = _matrix(logits6, 'six-class complete head logits', CLASS_COUNT, fp32=True)
    if len(logits) != len(values['fused160']):
        raise ValueError('Raw branch and six-class logit row counts differ')
    columns = []
    for name in DIMENSIONS:
        branch = fitted[name]
        unit = _unit((values[name] - branch['mean']) / branch['scale'])
        centers = branch['prototypes']
        cosine = np.clip(unit @ centers.T, -1., 1.)
        whitened = unit @ branch['whitener'].T
        whitened_centers = centers @ branch['whitener'].T
        distances = (whitened * whitened).sum(axis=1)[:, None] + (whitened_centers * whitened_centers).sum(axis=1)[None, :] - 2 * whitened @ whitened_centers.T
        negative_distance = -np.maximum(distances, 0.)
        columns.extend((np.sort(cosine, axis=1)[:, ::-1], np.sort(negative_distance, axis=1)[:, ::-1]))
    top = logits.max(axis=1)
    exponential_sum = np.exp(logits - top[:, None]).sum(axis=1)
    pair = np.sort(logits, axis=1)[:, -2:]
    columns.append(np.column_stack((1. / exponential_sum, top, pair[:, 1] - pair[:, 0], top + np.log(exponential_sum))))
    result = _fp32(np.concatenate(columns, axis=1), 'source descriptor')
    if result.shape != (len(logits), DESCRIPTOR_DIM):
        raise ValueError('Source descriptor dimension differs')
    return result


def fit_readouts(known_descriptors, oe_descriptors, model_seed=2026092701):
    """Train two CPU binary membership readers, exactly 200 full-batch steps each.

    Only the caller's legal source L and source OE training descriptors enter.
    Membership supervision is explicit; no OE TX labels or V argument is accepted.
    """
    known = _matrix(known_descriptors, 'known source train descriptors', DESCRIPTOR_DIM)
    oe = _matrix(oe_descriptors, 'source OE train descriptors', DESCRIPTOR_DIM)
    if type(model_seed) is not int or model_seed <= 0:
        raise ValueError('Positive integer CPU readout seed required')
    train = np.concatenate((known, oe), axis=0)
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale = np.where(scale <= 1e-8, 1., scale)
    normalized = _fp32((train - mean) / scale, 'training descriptors')
    import torch

    # Explicit list bridges avoid the PyTorch 2.1/NumPy 2 ndarray ABI boundary.
    x = torch.tensor(normalized.tolist(), dtype=torch.float32, device='cpu')
    models, losses = {}, {}
    counts = {}
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU generator; no CUDA generator/state is consulted.
        torch.random.default_generator.manual_seed(model_seed)
        for name in METHODS:
            if name == 'linear_oe':
                parameters = dict(weight=torch.nn.Parameter(torch.zeros((1, DESCRIPTOR_DIM), dtype=torch.float32, device='cpu')),
                                  bias=torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cpu')))
            else:
                parameters = dict(weight1=torch.nn.Parameter(torch.empty((16, DESCRIPTOR_DIM), dtype=torch.float32, device='cpu')),
                                  bias1=torch.nn.Parameter(torch.zeros(16, dtype=torch.float32, device='cpu')),
                                  weight2=torch.nn.Parameter(torch.empty((1, 16), dtype=torch.float32, device='cpu')),
                                  bias2=torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cpu')))
                torch.nn.init.xavier_uniform_(parameters['weight1'])
                torch.nn.init.xavier_uniform_(parameters['weight2'])
            optimizer = torch.optim.AdamW(list(parameters.values()), lr=.001, weight_decay=.0001,
                                           betas=(.9, .999), eps=1e-8, foreach=False)
            trace = []
            for _ in range(STEPS):
                optimizer.zero_grad(set_to_none=True)
                if name == 'linear_oe':
                    g = (x @ parameters['weight'].T + parameters['bias']).reshape(-1)
                else:
                    hidden = torch.relu(x @ parameters['weight1'].T + parameters['bias1'])
                    g = (hidden @ parameters['weight2'].T + parameters['bias2']).reshape(-1)
                loss = .5 * torch.nn.functional.softplus(-g[:len(known)]).mean() + .5 * torch.nn.functional.softplus(g[len(known):]).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite source OE training loss')
                trace.append(float(loss.detach().item()))
                loss.backward()
                optimizer.step()
            models[name] = {key: _fp32(value.detach().cpu().tolist(), name + ' ' + key)
                            for key, value in parameters.items()}
            counts[name] = sum(value.size for value in models[name].values())
            losses[name] = trace
    return dict(schema=READOUT_SCHEMA, descriptor_dim=DESCRIPTOR_DIM, method_names=list(METHODS),
        mean=mean, scale=scale, models=models,
        training=dict(model_seed=model_seed, steps=STEPS, known_count=len(known), oe_count=len(oe),
            optimizer=dict(OPTIMIZER), parameter_counts=counts, losses=losses,
            loss_record='full_batch_balanced_softplus_before_each_update', source_membership_supervision=True,
            oe_transmitter_labels_used=False, identity_head_trained=False, device='cpu', dropout=0.))


def _readout_state(state):
    if (not isinstance(state, dict) or set(state) != {'schema', 'descriptor_dim', 'method_names', 'mean', 'scale', 'models', 'training'}
            or state['schema'] != READOUT_SCHEMA or type(state['descriptor_dim']) is not int
            or state['descriptor_dim'] != DESCRIPTOR_DIM or state['method_names'] != list(METHODS)
            or not isinstance(state['models'], dict) or set(state['models']) != set(METHODS)):
        raise ValueError('Frozen source OE readout schema differs')
    mean = _array(state['mean'], (DESCRIPTOR_DIM,), 'readout mean')
    scale = _array(state['scale'], (DESCRIPTOR_DIM,), 'readout scale')
    if np.any(scale <= 0):
        raise ValueError('Readout scales must be positive')
    shapes = dict(linear_oe=dict(weight=(1, DESCRIPTOR_DIM), bias=(1,)),
                  mlp16_oe=dict(weight1=(16, DESCRIPTOR_DIM), bias1=(16,), weight2=(1, 16), bias2=(1,)))
    for name in METHODS:
        model = state['models'][name]
        if not isinstance(model, dict) or set(model) != set(shapes[name]):
            raise ValueError('Readout model parameter schema differs: ' + name)
        for key, shape in shapes[name].items():
            _array(model[key], shape, name + ' ' + key, fp32=True)
    return mean, scale, state['models']


def score_readouts(desc, state):
    """Pure NumPy fixed-state [N,2] forward; higher scores mean source-known."""
    values = _matrix(desc, 'evaluation descriptors', DESCRIPTOR_DIM)
    mean, scale, models = _readout_state(state)
    x = _fp32((values - mean) / scale, 'evaluation normalized descriptors')
    linear, mlp = models['linear_oe'], models['mlp16_oe']
    first = x @ linear['weight'].T + linear['bias']
    hidden = np.maximum(x @ mlp['weight1'].T + mlp['bias1'], np.float32(0.))
    second = hidden @ mlp['weight2'].T + mlp['bias2']
    return _fp32(np.concatenate((first, second), axis=1), 'readout scores')
