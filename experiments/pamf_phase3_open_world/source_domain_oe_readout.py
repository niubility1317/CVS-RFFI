"""Source OE readers with equal role-internal source RX/day domain mass.

Only source membership and source domains are used. Frozen F, geometry and
consensus descriptors are caller inputs; no TX identity, V, IO or threshold
calibration is accepted here. The output uses the original readout schema.
"""
from __future__ import annotations

import numpy as np

from . import source_open_readout as core


REGIME = 'source_role_domain_balanced_v1'
SOURCE_RECEIVERS = (1, 3, 4, 6, 8)
DOMAIN_COUNT = 15


def domains_from_metadata(rx_i, day_i):
    """Map Main RX order and days 1..3 to 0..14, without TX information."""
    rx, day = np.asarray(rx_i), np.asarray(day_i)
    if (rx.ndim != 1 or not len(rx) or day.shape != rx.shape
            or rx.dtype.kind not in 'iu' or day.dtype.kind not in 'iu'
            or not np.isin(rx, SOURCE_RECEIVERS).all() or not np.isin(day, (1, 2, 3)).all()):
        raise ValueError('Aligned nonempty integer source Main RX/day vectors required')
    positions = np.asarray([SOURCE_RECEIVERS.index(int(value)) for value in rx], np.int64)
    return positions * 3 + day.astype(np.int64) - 1


def _domains(value, count, name):
    domains = np.asarray(value)
    if (domains.shape != (count,) or domains.dtype.kind not in 'iu'
            or set(np.unique(domains).tolist()) != set(range(DOMAIN_COUNT))):
        raise ValueError('Each source role must cover all 15 integer domains: ' + name)
    return domains.astype(np.int64, copy=False)


def _canonical(values, domains):
    # Fixed numerical accumulation order under row permutations. This uses only
    # domain and the descriptor itself, with identical records indistinguishable.
    keys = tuple(values[:, column] for column in range(core.DESCRIPTOR_DIM - 1, -1, -1)) + (domains,)
    order = np.lexsort(keys)
    return values[order], domains[order]


def _role_weights(domains):
    counts = np.bincount(domains, minlength=DOMAIN_COUNT)
    weights = 1. / (DOMAIN_COUNT * counts[domains])
    return counts, weights


def _weight_summary(weights):
    return dict(minimum=float(weights.min()), maximum=float(weights.max()), sum=float(weights.sum()))


def fit_readouts(known_descriptors, oe_descriptors, known_domains, oe_domains, model_seed=2026092701):
    """Fit two 200-step CPU readers with domain-uniform membership risk.

    Loss role priors are .5/.5. Normalizer role priors remain .6/.4, inherited
    from the original 6300/4200 source training ratio, not estimated from novel
    data or the expanded training counts. Returned state has aggregates only.
    """
    known = core._matrix(known_descriptors, 'source known descriptors', core.DESCRIPTOR_DIM)
    oe = core._matrix(oe_descriptors, 'source OE descriptors', core.DESCRIPTOR_DIM)
    kd = _domains(known_domains, len(known), 'known')
    od = _domains(oe_domains, len(oe), 'OE')
    if type(model_seed) is not int or model_seed <= 0:
        raise ValueError('Positive integer CPU readout seed required')
    known, kd = _canonical(known, kd)
    oe, od = _canonical(oe, od)
    known_counts, known_weights = _role_weights(kd)
    oe_counts, oe_weights = _role_weights(od)
    train = np.concatenate((known, oe), axis=0)
    normalizer_weights = np.r_[.6 * known_weights, .4 * oe_weights]
    total_weight = normalizer_weights.sum()
    with np.errstate(over='ignore', invalid='ignore'):
        mean = (train * normalizer_weights[:, None]).sum(axis=0) / total_weight
        variance = (np.square(train - mean) * normalizer_weights[:, None]).sum(axis=0) / total_weight
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError('Nonfinite weighted source normalizer statistics')
    scale = np.sqrt(np.maximum(variance, 0.))
    scale = np.where(scale <= 1e-8, 1., scale)
    normalized = core._fp32((train - mean) / scale, 'domain-balanced training descriptors')

    import torch
    x = torch.tensor(normalized.tolist(), dtype=torch.float32, device='cpu')
    wk = torch.tensor(known_weights.tolist(), dtype=torch.float32, device='cpu')
    wo = torch.tensor(oe_weights.tolist(), dtype=torch.float32, device='cpu')
    models, losses, parameter_counts = {}, {}, {}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed)
        for name in core.METHODS:
            if name == 'linear_oe':
                parameters = dict(weight=torch.nn.Parameter(torch.zeros((1, core.DESCRIPTOR_DIM), dtype=torch.float32, device='cpu')),
                                  bias=torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cpu')))
            else:
                parameters = dict(weight1=torch.nn.Parameter(torch.empty((16, core.DESCRIPTOR_DIM), dtype=torch.float32, device='cpu')),
                                  bias1=torch.nn.Parameter(torch.zeros(16, dtype=torch.float32, device='cpu')),
                                  weight2=torch.nn.Parameter(torch.empty((1, 16), dtype=torch.float32, device='cpu')),
                                  bias2=torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cpu')))
                torch.nn.init.xavier_uniform_(parameters['weight1'])
                torch.nn.init.xavier_uniform_(parameters['weight2'])
            optimizer = torch.optim.AdamW(list(parameters.values()), lr=.001, weight_decay=.0001,
                                           betas=(.9, .999), eps=1e-8, foreach=False)
            trace = []
            for _ in range(core.STEPS):
                optimizer.zero_grad(set_to_none=True)
                if name == 'linear_oe':
                    g = (x @ parameters['weight'].T + parameters['bias']).reshape(-1)
                else:
                    hidden = torch.relu(x @ parameters['weight1'].T + parameters['bias1'])
                    g = (hidden @ parameters['weight2'].T + parameters['bias2']).reshape(-1)
                loss = .5 * (wk * torch.nn.functional.softplus(-g[:len(known)])).sum() + .5 * (wo * torch.nn.functional.softplus(g[len(known):])).sum()
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite source role/domain balanced loss')
                trace.append(float(loss.detach().item()))
                loss.backward()
                optimizer.step()
            models[name] = {key: core._fp32(value.detach().cpu().tolist(), name + ' ' + key)
                            for key, value in parameters.items()}
            losses[name] = trace
            parameter_counts[name] = sum(value.size for value in models[name].values())
    optimizer_metadata = dict(core.OPTIMIZER, betas=list(core.OPTIMIZER['betas']))
    return dict(schema=core.READOUT_SCHEMA, descriptor_dim=core.DESCRIPTOR_DIM, method_names=list(core.METHODS),
        mean=mean, scale=scale, models=models,
        training=dict(regime=REGIME, model_seed=model_seed, steps=core.STEPS,
            known_count=len(known), oe_count=len(oe), optimizer=optimizer_metadata,
            parameter_counts=parameter_counts, losses=losses,
            domain_count=DOMAIN_COUNT, domain_counts=dict(known=known_counts.tolist(), oe=oe_counts.tolist()),
            normalizer_role_prior=dict(known=.6, oe=.4), loss_role_prior=dict(known=.5, oe=.5),
            normalizer_prior_origin='fixed original source L6300/OE4200 ratio, not expanded counts or novel',
            normalizer_weight_sum=float(total_weight), loss_weights=dict(known=_weight_summary(known_weights), oe=_weight_summary(oe_weights)),
            normalizer_weight_range=dict(minimum=float(normalizer_weights.min()), maximum=float(normalizer_weights.max())),
            normalizer_variance='weighted population variance',
            loss_record='full_batch_role_domain_balanced_softplus_before_each_update',
            canonical_order='domain_then_descriptor_lexicographic', source_membership_supervision=True,
            oe_transmitter_labels_used=False, identity_head_trained=False, device='cpu', dropout=0.))
