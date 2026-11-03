"""Train edge fusion on source IDs disjoint from its spectral expert."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_edgeblend.model import make_model as DynamicVolterra
from experiments.phase1_dura_safe_memory.objective import protected_source_objective
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
EXPERT = '20261102-phase1-spectral-aligned-source-s2026092701-r01'
EXPERT_COMMIT = 'b326a21a54ae7b9b198e3d077c01a3919d8bd135'
EXPERT_SHA = 'b06b2b2f7ccace9782ebf4587f9c0be2bf518795593bb3c0641fbe0e9d41ecc4'


def select_disjoint_fit(training, prior_ids, expert_ids, size):
    training, prior_ids, expert_ids = (np.asarray(x) for x in (training, prior_ids, expert_ids))
    if (size != 8192 or any(x.ndim != 1 for x in (training, prior_ids, expert_ids))
            or len(np.unique(training)) != len(training) or len(np.unique(prior_ids)) != len(prior_ids)
            or len(expert_ids) != 8192 or len(np.unique(expert_ids)) != 8192
            or not np.isin(prior_ids, training).all() or not np.isin(expert_ids, training).all()
            or np.intersect1d(prior_ids, expert_ids).size):
        raise ValueError('Distinct source training, prior and expert IDs required')
    pool = np.setdiff1d(training, np.concatenate((prior_ids, expert_ids)), assume_unique=False)
    if len(pool) < size:
        raise ValueError('Insufficient disjoint source training IDs')
    return np.sort(np.random.default_rng(2026110357).choice(pool, size=size, replace=False))


def select_crossfit(training, prior_ids, size):
    folder = ROOT/'runs'/EXPERT
    meta = json.loads((folder/'prediction_complete.json').read_text(encoding='utf8'))
    expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=EXPERT_COMMIT, count=672, fit_ids=8192,
                    F_frozen=True, TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0,
                    model_updates_after_prediction=0)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed label-free spectral expert provenance required')
    path = folder/'checkpoint.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPERT_SHA:
        raise ValueError('Exact frozen expert checkpoint required')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('schema') != 'phase1_spectral_aligned_v1' or state.get('commit') != EXPERT_COMMIT or state.get('epochs') != 5:
        raise ValueError('Exact source expert training state required')
    return select_disjoint_fit(training, prior_ids, state['fit_ids'], size)


def main():
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components',
             'extra_frozen_model_components', 'select_fit')
    old = tuple(getattr(base, name) for name in names)
    new = (DynamicVolterra, 'phase1_dura_crossfit_blend_v1', 'DURA_CROSSFIT_BLEND',
           protected_source_objective, True, load_components, input_noise, predict_batch,
           lambda dura, eye: (dura,), lambda model: (model.expert,), select_crossfit)
    for name, value in zip(names, new): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)


if __name__ == '__main__': main()
