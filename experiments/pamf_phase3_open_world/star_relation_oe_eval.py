"""Frozen source calibration and one cross-RX G10 future-Q falsification.

The only readout is normalized default-head negative entropy. Seven source
known groups constrain a single global threshold. Other source slices are
diagnostics. No source novel success gate, U fitting or private truth is used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import star_joint_ova_eval as previous
from . import star_consensus_readout as risk
from . import star_consensus_score as common
from . import star_relative_target as public
from .development_transfer import _snapshot_model, _check_model, _load_dura

RUN_ID = '20261030-phase3-star-relation-oe-s2026092701-r01'
SCHEMA = 'pamf_phase3_star_relation_oe_prediction_v1'
STATUS = 'STAR_RELATION_OE_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS = 'STAR_RELATION_OE_THRESHOLD_FROZEN'
ARMS = ('relation_control', 'relation_oe')
PRIMARY_ARM = 'relation_oe'
PRIMARY_METHOD = 'entropy_global'
FIELDS = previous.FIELDS | {'Q_public_root', 'Q_capsule_id', 'Q_split_id', 'dura_checkpoint', 'dura_release'}
KNOWN_FIELDS = {'ids', 'y', 'rx_i', 'day_i', 'scenario', 'default_logits', 'closed', 'baseline_closed', 'scores'}
NOVEL_FIELDS = {'ids', 'rx_i', 'day_i', 'default_logits', 'closed', 'scores', 'candidate_fused_unit', 'baseline_fused_unit'}
TARGET_FIELDS = {'ids', 'default_logits', 'closed', 'baseline_closed', 'scores'}
MAIN_GROUPS = ('overall',) + tuple('scene/' + s for s in common.SCENES)
Q_COUNT = 864
_read, _write, _npz = common._read, common._write, previous._npz


def entropy_score(logits):
    z = np.asarray(logits, np.float64)
    if z.ndim != 2 or z.shape[1] != 6 or not len(z) or not np.isfinite(z).all():
        raise ValueError('Finite full default-six logits required')
    shifted = z - z.max(1, keepdims=True)
    lp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    result = 1. + np.sum(np.exp(lp) * lp, axis=1) / np.log(6.)
    return np.clip(result, 0., 1.).astype(np.float64)


def calibrate_known(ids, y, closed, baseline, values, rx, scene):
    """Largest legal global complete-tie threshold; known-V inputs only."""
    ids, y, closed, baseline, values, rx, scene = map(np.asarray, (ids, y, closed, baseline, values, rx, scene))
    n = len(ids)
    if (not common._ids(ids, n) or not n or values.shape != (n,) or values.dtype.kind != 'f'
            or not np.isfinite(values).all() or any(a.shape != (n,) or a.dtype.kind not in 'iu' for a in (y, closed, baseline, rx))
            or any(np.any((a < 0) | (a >= 6)) for a in (y, closed, baseline))
            or not np.isin(rx, common.RECEIVERS).all() or scene.shape != (n,) or scene.dtype.kind != 'U'
            or not np.isin(scene, common.SCENES).all()):
        raise ValueError('Aligned legal source-known V metadata and finite evidence only')
    masks = dict(common._masks(rx, scene))
    coverage = {k: set(y[m].tolist()) == set(range(6)) for k, m in masks.items()}
    def group(key, tau):
        m = masks[key]
        return risk._risk_group(y[m], closed[m], baseline[m], values[m], tau) if coverage[key] else dict(
            certified=False, constraints_passed=False, reason='missing_known_class', known_count=int(m.sum()))
    def feasible(tau):
        return all(group(k, tau)['constraints_passed'] for k in MAIN_GROUPS)
    levels = np.unique(values)
    zero_pass = feasible(float(levels[0]))
    lo, hi = 0, len(levels) - 1
    if zero_pass:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if feasible(float(levels[mid])):
                lo = mid
            else:
                hi = mid - 1
    tau = float(levels[lo] if zero_pass else levels[0])
    following = float(levels[lo+1]) if zero_pass and lo+1 < len(levels) else None
    groups = {k: group(k, tau) for k in masks}
    return tau, dict(status=CALIBRATION_STATUS, method=PRIMARY_METHOD, threshold=tau, known_count=n,
        hard_constraint_groups=list(MAIN_GROUPS), hard_constraint_group_count=7, diagnostic_group_count=42,
        groups=groups, constraints_passed=bool(zero_pass and feasible(tau)), zero_rejection_fallback=not zero_pass,
        next_threshold=following, next_threshold_passed=feasible(following) if following is not None else None,
        known_only=True, unknown_inputs_used=False, target_inputs_used=False, source_V_neural_fit=False,
        threshold_rule='maximum legal global observed level; acceptance >= threshold, complete ties',
        limits=dict(known_macro_acceptance_min=.90, original_F0_closed_minus_open_max=.02),
        RX_and_RX_scene_diagnostic_only=True, shared_predicted_class_tail_constraint=False)


def validate_config(c):
    from .star_relation_oe_train import validate_config as train_validate
    if (not isinstance(c, dict) or set(c) != FIELDS or c.get('run_id') != RUN_ID or c.get('arm') not in ARMS
            or not isinstance(c.get('code_commit'), str) or re.fullmatch('[0-9a-f]{40}', c['code_commit']) is None
            or type(c.get('batch_size')) is not int or c['batch_size'] != 128):
        raise ValueError('Exact source-and-development evaluation schema required; no U/truth/K/quotas')
    train_validate(c['train_config'])
    if any(c['train_config'][k] != c[k] for k in ('run_id', 'arm', 'code_commit')):
        raise ValueError('Matching immutable trained row required')
    for k in FIELDS - {'batch_size', 'train_config'}:
        if not isinstance(c[k], str) or not c[k].strip():
            raise ValueError('Nonempty frozen state and public handles required')
    if (Path(c['checkpoint']) != Path(c['train_config']['output_root'])/'final.pt'
            or Path(c['Q_public_root']).name != 'public' or Path(c['Q_public_root']).parent.name != 'evaluation'
            or c['Q_capsule_id'] != 'p3dev-star-g10-71a5a0dec23ac6ccc3b479c2'
            or c['Q_split_id'] != '05de65cd4818f3e5e33ced5c'):
        raise ValueError('Only fixed RX24-1 G10 future Q and matching final checkpoint allowed')
    if any(Path(c['output_root']).resolve().is_relative_to(Path(c[k]).resolve()) for k in ('cache_root', 'Q_public_root')):
        raise ValueError('Cannot overwrite source or target inputs')
    if Path(c['output_root']).resolve().is_relative_to(Path(c['train_config']['output_root']).resolve()):
        raise ValueError('Cannot overwrite trained state')
    return c


def _backend(c, device):
    from .star_relation_oe_train import load_candidate
    model, cfg, payload = load_candidate(c['train_config']['output_root'], device)
    if cfg != c['train_config'] or payload['identity']['checkpoint'] != c['checkpoint']:
        raise ValueError('Loaded final checkpoint identity differs')
    return model, model.id_backbone.encoder, payload['identity']


def _predict(reader, iq, device, keep_unit=False):
    z, _, u = previous._predict(reader, None, iq, device, 128, keep_unit)
    return z, u


def _probe(reader, device, g=None):
    from .star_channel import observe
    mix = _read(Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json')
    t = np.arange(256)
    a = np.asarray([observe(np.stack((np.cos(w*t), np.sin(w*t))), physical_id='relation-oe-smoke-'+str(i),
        receiver='1-19', day='2021_03_08', episode=dict(id='MIX01', scene_seed=2026103001, channel_seed=2026104001), mixture=mix)[0]
        for i, w in enumerate((.07, .11))], np.float32)
    if g is not None:
        a = _restored(a, g, device)
    z, _ = _predict(reader, a, device)
    entropy_score(z)
    return dict(status='PASS', actual_candidate_checkpoint=True, synthetic_G10_rows=2,
        candidate_forward_rows=2, G_forward_rows=2 if g is not None else 0, real_data_reads=0, optimizer_updates=0)


def _restored(a, g, device):
    import torch
    from .source_open_export import _tensor
    with torch.no_grad():
        x = g[1](_tensor(a, device))[0].detach()
    out = np.asarray(x.cpu().tolist(), np.float32)
    if out.shape != a.shape or not np.isfinite(out).all():
        raise ValueError('Frozen DURA must return finite same-shape P1')
    return out


def run(c, device='cuda:0'):
    validate_config(c)
    out = Path(c['output_root'])
    if out.exists():
        raise FileExistsError('New immutable prediction root required')
    start = time.perf_counter()
    model, reader, identity = _backend(c, device)
    snap = _snapshot_model(model)
    smoke = _probe(reader, device)
    _check_model(model, snap)
    out.mkdir(parents=True)
    _write(out/'smoke.json', smoke)
    parent = identity['original_parent_identity']
    role = previous._load_role(c['cache_root'], 'V')
    try:
        previous._role(role, 'V', parent, None)
        dura_identity = role['manifest']['dura_identity']
        meta = role['metadata']
        idsV = set(role['ids'].tolist())
        z, _ = _predict(reader, role['restored'], device)
        closed = z.argmax(1).astype(np.int64)
        baseline = np.array(role['baseline_closed'][:, 1], copy=True)
        raw = entropy_score(z)
        tau, calibration = calibrate_known(role['ids'], meta['y'], closed, baseline, raw, meta['rx_i'], meta['scenario'])
        calibration.update(schema=SCHEMA, run_id=RUN_ID, code_commit=c['code_commit'], arm=c['arm'])
        (out/'calibration').mkdir()
        _write(out/'calibration/complete.json', calibration)
        with (out/'threshold.npy').open('xb') as f:
            np.save(f, np.asarray([tau], np.float64), allow_pickle=False)
        _npz(out/'known_predictions.npz', dict(ids=role['ids'], y=meta['y'], rx_i=meta['rx_i'], day_i=meta['day_i'],
            scenario=meta['scenario'], default_logits=z, closed=closed, baseline_closed=baseline, scores=raw-tau))
    finally:
        previous._close_role(role)
    _check_model(model, snap)
    print(CALIBRATION_STATUS, c['arm'], 'source_RU_is_not_target_entry_gate', flush=True)
    original, parent_loaded = previous._load_original(c, device)
    if parent_loaded != parent:
        raise ValueError('Original same-input F0 baseline identity differs')
    original_snap = _snapshot_model(original)
    role = previous._load_role(c['cache_root'], 'novel')
    try:
        previous._role(role, 'novel', parent, dura_identity)
        if idsV & set(role['ids'].tolist()):
            raise ValueError('Source known and heldout novel must be disjoint')
        z, units = _predict(reader, role['restored'], device, True)
        _, baseline_units = _predict(original, role['restored'], device, True)
        _npz(out/'novel_predictions.npz', dict(ids=role['ids'], rx_i=role['metadata']['rx_i'], day_i=role['metadata']['day_i'],
            default_logits=z, closed=z.argmax(1).astype(np.int64), scores=entropy_score(z)-tau,
            candidate_fused_unit=units, baseline_fused_unit=baseline_units))
    finally:
        previous._close_role(role)
    _check_model(model, snap)
    _check_model(original, original_snap)
    # No source result is read. Both rows reach the same development Q once.
    if c['dura_checkpoint'] != dura_identity['checkpoint'] or c['dura_release'] != dura_identity['release_root']:
        raise ValueError('Target DURA paths must match the frozen source-cache restorer')
    dura = _load_dura({'train_config': c['train_config']['native_config']}, c, device)
    dura_snap = _snapshot_model(dura[0])
    target_smoke = _probe(reader, device, dura)
    _write(out/'target_smoke.json', target_smoke)
    qspec = dict(c, Q_expected_count=Q_COUNT)
    _, idsQ, iq = public._open(qspec, 'Q')
    logits, base_logits = [], []
    try:
        for start_q in range(0, Q_COUNT, 128):
            x = _restored(np.asarray(iq[start_q:start_q+128]), dura, device)
            z, _ = _predict(reader, x, device)
            z0, _ = _predict(original, x, device)
            logits.append(z); base_logits.append(z0)
        z, z0 = np.concatenate(logits), np.concatenate(base_logits)
        _npz(out/'target_predictions.npz', dict(ids=idsQ, default_logits=z, closed=z.argmax(1).astype(np.int64),
            baseline_closed=z0.argmax(1).astype(np.int64), scores=entropy_score(z)-tau))
    finally:
        if hasattr(iq, '_mmap'):
            iq._mmap.close()
    _check_model(model, snap); _check_model(original, original_snap); _check_model(dura[0], dura_snap)
    done = dict(schema=SCHEMA, status=STATUS, run_id=RUN_ID, code_commit=c['code_commit'], arm=c['arm'], config=c,
        model_identity=identity, parent_F0_identity=parent, dura_identity=dura_identity,
        method=PRIMARY_METHOD, primary_arm=PRIMARY_ARM, path='P1', known_count=27000, novel_count=3150, target_count=Q_COUNT,
        source_only=False, target_RX='24-1', target_U_reads=0, target_Q_rows=Q_COUNT, target_Q_fit=False,
        source_truth_reads=0, target_truth_reads=0, source_V_neural_fit=False, source_V_threshold_fit=True,
        threshold=tau, thresholds_frozen_before_novel_and_target=True, source_unknown_success_target_entry_gate=False,
        source_hard_constraint_groups=list(MAIN_GROUPS), source_diagnostic_groups=42, eval_neural_updates=0,
        frozen_models_verified=True, source_bank_deployed=False, per_record_source_rows_deployed=0,
        all_new_IQ_satellite=True, target_raw_clean_reads=0, formal_target_accessed=False,
        candidate_real_P1_forward_rows=30150+Q_COUNT, candidate_synthetic_forward_rows=4,
        original_F0_real_P1_forward_rows=3150+Q_COUNT, G_real_forward_rows=Q_COUNT, G_synthetic_forward_rows=2,
        target_DA=False, target_REG=False, A_N=None, H_OW=None, final_phase3_complete=False,
        seconds=time.perf_counter()-start)
    _write(out/'complete.json', done)
    print(STATUS, c['arm'], flush=True)
    return done


def main():
    print(json.dumps(dict(event='STAR_RELATION_OE_EVAL_PROCESS', pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv)), flush=True)
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['run']); p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    run(_read(a.config), a.device)


if __name__ == '__main__':
    main()
