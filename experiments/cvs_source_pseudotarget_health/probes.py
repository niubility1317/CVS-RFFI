"""Detached RX probes trained only on the preregistered L_s folds 0 and 1."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from . import design as d


def partition(L):
    from experiments.cvs_repeated_rx_margin.drift import partition as source_partition
    folds, info = source_partition(L)
    info.update(train_folds=[0, 1], diagnostic_fold=2,
                all_L_train=False,
                fold2_used_for_probe_training=False, V_used_for_probe_training=False)
    return folds, info


def rx_labels(rx):
    if not set(rx.tolist()) <= set(d.RX):
        raise ValueError('Only legal source RX labels are allowed')
    return (rx[:, None] == torch.tensor(d.RX, device=rx.device)).long().argmax(1)


def fit_probe(features, rx, seed, device):
    if features.ndim != 2 or len(features) != len(rx) or features.requires_grad:
        raise ValueError('Detached L training features required')
    if not torch.isfinite(features).all() or set(rx.tolist()) != set(d.RX):
        raise ValueError('Finite features from all five legal source RX required')
    features = features.to(device); labels = rx_labels(rx).to(device)
    mean = features.mean(0).detach(); scale = features.std(0, unbiased=False).clamp_min(d.PROBE['normalization_eps']).detach()
    standardized = ((features - mean) / scale).detach()
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        classifier = torch.nn.Linear(features.shape[1], len(d.RX)).to(device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=d.PROBE['learning_rate'],
                                      weight_decay=d.PROBE['weight_decay'])
        for _ in range(d.PROBE['steps']):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(standardized), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('RX probe loss is not finite')
            loss.backward(); optimizer.step()
    classifier.eval().requires_grad_(False)
    for parameter in classifier.parameters():
        parameter.grad = None
    return classifier, mean, scale


@torch.no_grad()
def evaluate_probe(classifier, mean, scale, features, rx, device):
    if features.ndim != 2 or features.requires_grad or len(features) != len(rx) or len(features) == 0:
        raise ValueError('Nonempty detached probe diagnostic features required')
    scores = classifier((features.to(device) - mean) / scale)
    labels = rx_labels(rx).to(device); predictions = scores.argmax(1)
    correct = predictions == labels
    confusion = torch.bincount(labels * len(d.RX) + predictions, minlength=len(d.RX) ** 2).reshape(len(d.RX), len(d.RX))
    return dict(count=len(features), accuracy=float(correct.float().mean()),
                correct_count=int(correct.sum()), confusion=confusion.cpu().tolist(),
                loss=float(F.cross_entropy(scores, labels)), RX_order=list(d.RX))


def run_probes(features, validation_features, L, V, seed, device):
    if tuple(features) != d.FEATURES or tuple(validation_features) != d.FEATURES:
        raise ValueError('Fixed shallow, middle and joint160 features required')
    folds, partition_info = partition(L)
    result = dict(partition=partition_info, recipe=dict(d.PROBE),
                  identity_features_detached=True, identity_updates=0, target_access=False,
                  probe_state_persisted=False, RX_information_is_not_pure_receiver_causality=True,
                  layers={})
    for level in d.FEATURES:
        z = features[level]; vz = validation_features[level]
        if len(z) != len(L['ids']) or len(vz) != len(V['ids']):
            raise ValueError('Feature/source role counts differ')
        if z.requires_grad or vz.requires_grad or not torch.isfinite(z).all() or not torch.isfinite(vz).all():
            raise ValueError('Detached finite identity features required')
        records = {}
        for tx in (None, *range(d.CLASSES)):
            group = torch.ones(len(z), dtype=torch.bool) if tx is None else L['y'] == tx
            vgroup = torch.ones(len(vz), dtype=torch.bool) if tx is None else V['y'] == tx
            train = group & (folds != 2); diagnostic = group & (folds == 2)
            classifier, mean, scale = fit_probe(z[train], L['rx'][train], seed + d.PROBE['seed_offset'], device)
            record = dict(TX=tx, probe_seed=seed + d.PROBE['seed_offset'], feature_dimension=z.shape[1],
                          training= evaluate_probe(classifier, mean, scale, z[train], L['rx'][train], device),
                          L_fold2= evaluate_probe(classifier, mean, scale, z[diagnostic], L['rx'][diagnostic], device),
                          source_V= evaluate_probe(classifier, mean, scale, vz[vgroup], V['rx'][vgroup], device),
                          fitted_steps=d.PROBE['steps'], normalization_fit_count=int(train.sum()),
                          fit_role='L_s folds0/1 only', performance_gate=False)
            records['unconditional' if tx is None else f'TX{tx}'] = record
        result['layers'][level] = records
    return result


@contextmanager
def capture(model):
    """Capture the two established native layers, without replacing their outputs."""
    from experiments.cvs_report_alignment_clean.alignment import capture as native_capture
    with native_capture(model) as maps:
        yield maps


def descriptors(maps, joint):
    if tuple(maps) != d.FEATURES[:2] or joint.ndim != 2 or joint.shape[1] != 160:
        raise ValueError('Actual fixed native feature interfaces differ')
    result = {}
    for level, h in maps.items():
        if h.ndim != 3 or len(h) != len(joint) or not torch.isfinite(h).all():
            raise ValueError('Finite native channel/time feature required')
        result[level] = torch.cat((h.mean(-1), h.std(-1, unbiased=False)), 1).detach().cpu()
    result['joint160'] = joint.detach().cpu()
    return result
