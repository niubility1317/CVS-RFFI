"""Focused synthetic checks: hidden labels, fixed statistics and detached probes."""
import argparse
import copy
import json
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import design as d
from .stats import calibration, selection_comparison, summarize, validate_role
from .probes import partition, fit_probe, evaluate_probe, run_probes
from .frozen import preserve_rng, snapshot, assert_unchanged
from .run import scan, run as diagnostic_run, _write_new, verify_registration


def fixture():
    entries = [(tx, rx, day, packet) for tx in range(6) for rx in d.RX
               for day in d.DAYS for packet in range(6)]
    generator = torch.Generator().manual_seed(789)
    L = dict(x=torch.randn(len(entries), 2, 256, generator=generator),
             y=torch.tensor([a for a, b, c, i in entries]),
             rx=torch.tensor([b for a, b, c, i in entries]),
             day=torch.tensor([c for a, b, c, i in entries]),
             ids=[f'tx{a}-rx{b}-day{c}-packet{i:03d}' for a, b, c, i in entries])
    U = {k: copy.deepcopy(L[k]) for k in ('x', 'ids', 'rx', 'day')}
    U.update(domain=torch.arange(len(entries)) % 15, eq=torch.zeros(len(entries), dtype=torch.long),
             sig=torch.arange(len(entries)), base_index=torch.arange(len(entries)),
             role='U_s', preprocessing='synthetic input; TX truth absent')
    return L, U


def _raises(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return
    raise AssertionError('Expected negative check to reject input')


def checks(device='cpu'):
    torch.set_num_threads(2)
    torch.manual_seed(20)
    L, U = fixture()
    validate_role(L, 'L_s', L['ids']); validate_role(U, 'U_s', U['ids'])
    _raises(lambda: validate_role(dict(U, y=L['y']), 'U_s', U['ids']))
    _raises(lambda: validate_role(dict(U, role='target'), 'U_s', U['ids']))
    q = torch.tensor([[.96, .01, .01, .01, .005, .005],
                      [.01, .96, .01, .01, .005, .005],
                      [.6, .1, .1, .1, .05, .05],
                      [.1, .1, .1, .6, .05, .05]])
    y = torch.tensor([0, 0, 0, 3]); own_stats = calibration(q, y)
    if own_stats['correct_count'] != 3 or own_stats['high_confidence_error'] != dict(count=1, selected=2, error_rate=.5):
        raise AssertionError('Accuracy/high confidence errors counted incorrectly')
    if abs(own_stats['ECE'] - .43) > 2e-7 or sum(row['count'] for row in own_stats['calibration_bins']) != 4:
        raise AssertionError('Fixed equal-width ECE bins calculated incorrectly')
    corrected = q.clone(); corrected[1] = corrected[0]
    others = torch.stack((corrected, corrected))
    peer, masks, _, _ = selection_comparison(q, others)
    if not torch.equal(peer, corrected) or not masks['peer_agree'].all() or int(masks['peer_agree_high'].sum()) != 2:
        raise AssertionError('Fixed other-two probability average or masks differ')
    rx = torch.tensor([1, 1, 3, 3]); day = torch.ones(4, dtype=torch.long)
    stats = summarize(q, others, rx, day, 'V', y)
    selection = stats['summary']['selection']['peer_agree_high']
    if selection['peer_corrected_own_errors'] != 1 or selection['peer_harmed_own_correct'] != 0 or selection['own_error_coverage'] != 1.:
        raise AssertionError('Peer error coverage/correction/harm counted incorrectly')
    hidden = summarize(q, others, rx, day, 'U_s')
    if hidden['summary']['own']['accuracy'] is not None or hidden['summary']['fixed_peer']['ECE'] is not None:
        raise AssertionError('Hidden U diagnostic reported correctness')
    if sum(row['count'] for row in hidden['RX_day'].values()) != 4 or sum(hidden['summary']['own']['prediction_counts']) != 4:
        raise AssertionError('Source RX/day or predicted TX totals differ')
    _raises(lambda: summarize(q, others, rx, day, 'U_s', y))
    # Probe train inputs and fitted normalization never contain held-out fold2.
    folds, info = partition(L)
    permutation = torch.randperm(len(folds), generator=torch.Generator().manual_seed(19))
    reordered = {k: v[permutation] if torch.is_tensor(v) else [v[i] for i in permutation.tolist()] for k, v in L.items()}
    reordered_folds, _ = partition(reordered)
    if not torch.equal(reordered_folds, folds[permutation]) or any((folds == k).sum() != len(folds) // 3 for k in range(3)):
        raise AssertionError('Sorted physical ID fold assignment is unstable')
    labels = (L['rx'][:, None] == torch.tensor(d.RX)).long().argmax(1)
    features = torch.nn.functional.one_hot(labels, len(d.RX)).float().repeat(1, 4)
    train = folds != 2; changed = features.clone(); changed[~train] += 10000
    seed = d.SEEDS[0] + d.PROBE['seed_offset']
    first, mean, scale = fit_probe(features[train], L['rx'][train], seed, device)
    second, mean2, scale2 = fit_probe(changed[train], L['rx'][train], seed, device)
    if not torch.equal(mean, mean2) or not torch.equal(scale, scale2) or any(
            not torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items()):
        raise AssertionError('Fold2 changed fitted probe state')
    probe_test = evaluate_probe(first, mean, scale, features[~train], L['rx'][~train], device)
    if probe_test['accuracy'] != 1. or probe_test['count'] != len(features) // 3:
        raise AssertionError('Ordinary linear RX probe evaluation differs')
    # One actual native identity fixture confirms the captured interfaces and freezing boundary.
    from experiments.cvs_repeated_rx_margin.model import build
    model = build(d.source_config(d.rows()[0]), device).eval().requires_grad_(False)
    before = snapshot(model)
    initial_rng = torch.get_rng_state().clone(); python_state = random.getstate(); numpy_state = np.random.get_state()
    cuda_state = torch.cuda.get_rng_state(torch.device(device)) if str(device).startswith('cuda') else None
    sample = {k: v[:12] if torch.is_tensor(v) else v[:12] for k, v in L.items()}
    with preserve_rng(device):
        torch.rand(3); random.random(); np.random.rand(3)
        probabilities, actual_features = scan([model], sample, 'L_s', device)
        assert_unchanged(model, before)
    if not torch.equal(initial_rng, torch.get_rng_state()) or random.getstate() != python_state:
        raise AssertionError('Diagnostic changed CPU/Python RNG')
    now_numpy = np.random.get_state()
    if now_numpy[0] != numpy_state[0] or not np.array_equal(now_numpy[1], numpy_state[1]) or now_numpy[2:] != numpy_state[2:]:
        raise AssertionError('Diagnostic changed NumPy RNG')
    if cuda_state is not None and not torch.equal(cuda_state, torch.cuda.get_rng_state(torch.device(device))):
        raise AssertionError('Diagnostic changed CUDA RNG')
    if probabilities.shape != (1, 12, 6) or [actual_features[0][k].shape[1] for k in d.FEATURES] != [96, 192, 160]:
        raise AssertionError('Actual native captured descriptor interfaces differ')
    frozen = assert_unchanged(model, before)
    # All probe families run from synthetic detached source features, no identity optimizer exists.
    all_features = {level: features for level in d.FEATURES}
    probes = run_probes(all_features, all_features, L, L, d.SEEDS[0], device)
    if len(probes['layers']) != 3 or any(len(records) != 7 for records in probes['layers'].values()):
        raise AssertionError('Unconditional/each TX probe matrix incomplete')
    if any(record['normalization_fit_count'] != (360 if record['TX'] is None else 60)
           for records in probes['layers'].values() for record in records.values()):
        raise AssertionError('Probe fit includes diagnostic folds')
    assert_unchanged(model, before)
    # Exclusive output and registration check are exercised without actual remote inputs.
    temporary_root = d.ROOT / 'local_artifacts'
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cvs_source_health_', dir=temporary_root) as folder:
        path = Path(folder) / 'new.json'; _write_new(path, dict(ok=True))
        _raises(lambda: _write_new(path, dict(ok=False)), FileExistsError)
        existing = Path(folder) / 'existing'; existing.mkdir()
        _raises(lambda: diagnostic_run(existing, device, registration=False), FileExistsError)
    spec = dict(run_id=d.RUN, rows=[dict(row_id=r['row_id'], config=d.config(r)) for r in d.rows()])
    with patch.object(d, 'read', return_value=spec):
        verify_registration()
    spec['rows'][0]['config']['confidence_threshold'] = .9
    with patch.object(d, 'read', return_value=spec):
        _raises(verify_registration)
    return dict(status='PASS', device=device, U_labels_rejected=True,
                exact_probability_count_calibration_highconfidence_error=True,
                fixed_peer_average_and_error_coverage=True, probe_fold2_excluded=True,
                native_capture_shapes=[96, 192, 160], probe_families=21,
                identity_frozen=frozen, RNG_preserved=True,
                output_overwrite_rejected=True, registration_mismatch_rejected=True,
                target_query_access=False, U_truth_read=False,
                synthetic_only=True, actual_source_cache_opened=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = checks(args.device)
    if args.output:
        _write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
