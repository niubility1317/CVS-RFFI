"""Focused source legality, detached probe matrix, and frozen identity checks."""
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
from .run import (validate_source_role, load_roles, scan, source_summary, analysis,
                  verify_registration, _write_new, _rng_state, _assert_rng)
from experiments.cvs_source_pseudotarget_health.probes import partition, fit_probe, descriptors, run_probes
from experiments.cvs_source_pseudotarget_health.frozen import preserve_rng, snapshot, assert_unchanged


def fixture():
    entries = [(tx, rx, day, packet) for tx in range(6) for rx in d.RX
               for day in d.DAYS for packet in range(3)]
    generator = torch.Generator().manual_seed(704)
    return dict(x=torch.randn(len(entries), 2, 256, generator=generator),
                y=torch.tensor([tx for tx, rx, day, packet in entries]),
                rx=torch.tensor([rx for tx, rx, day, packet in entries]),
                day=torch.tensor([day for tx, rx, day, packet in entries]),
                ids=[f'tx{tx}-rx{rx}-day{day}-packet{packet:03d}' for tx, rx, day, packet in entries])


def _raises(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return
    raise AssertionError('Expected source diagnostic negative check to reject input')


def checks(device='cpu'):
    torch.set_num_threads(2)
    L = fixture(); V = copy.deepcopy(L); V['ids'] = [value + '-V' for value in L['ids']]
    validate_source_role(L, 'L_s', L['ids']); validate_source_role(V, 'V', V['ids'])
    _raises(lambda: validate_source_role(L, 'U_s', L['ids']))
    _raises(lambda: validate_source_role(L, 'target', L['ids']))
    _raises(lambda: validate_source_role(dict(L, y=L['y'].float()), 'L_s', L['ids']))
    _raises(lambda: validate_source_role(dict(L, rx=L['rx'][:, None]), 'L_s', L['ids']))
    _raises(lambda: validate_source_role(dict(L, y=torch.full_like(L['y'], 6)), 'L_s', L['ids']))
    _raises(lambda: validate_source_role(L, 'L_s', [value + '-wrong' for value in L['ids']]))
    _raises(lambda: validate_source_role(dict(L, x=L['x'].requires_grad_()), 'L_s', L['ids']))
    L['x'] = L['x'].detach()
    from experiments.cvs_nocfo_foundation import source as source_loader
    with patch.object(source_loader, 'load_source', return_value=(L, {'forbidden': 'U'}, V, {})):
        _raises(load_roles)
    with patch.object(source_loader, 'load_source', return_value=(L, None, L, {'role_ids': {'L_s': L['ids'], 'V': L['ids']}})), \
            patch.dict(d.ROLE_COUNTS, {'L_s': len(L['ids']), 'V': len(L['ids'])}):
        _raises(load_roles)

    folds, partition_info = partition(L)
    labels = (L['rx'][:, None] == torch.tensor(d.RX)).long().argmax(1)
    base = torch.nn.functional.one_hot(labels, len(d.RX)).float()
    features = {level: base.repeat(1, (dim + 4) // 5)[:, :dim].contiguous()
                for level, dim in zip(d.FEATURES, d.DIMENSIONS)}
    _raises(lambda: fit_probe(features['time_down'].requires_grad_(), L['rx'], d.SEEDS[0], device))
    features['time_down'] = features['time_down'].detach()
    _raises(lambda: descriptors({'time_down': torch.zeros(2, 48, 128), 't2.pool': torch.zeros(2, 96, 32)}, torch.zeros(2, 159)))

    # Source labels/role groups and all 9 x 21 probe fits are independently counted.
    q = torch.full((len(L['ids']), 6), .01)
    q[torch.arange(len(q)), L['y']] = .95
    summaries = {'L_s': source_summary(q, L, 'L_s'), 'V': source_summary(q, V, 'V')}
    for value in summaries.values():
        summary = value['summary']
        if summary['correct_count'] != len(q) or sum(map(sum, summary['confusion'])) != len(q):
            raise AssertionError('Source TX confusion/count differs')
        for grouping in ('RX', 'day', 'RX_day', 'labeled_TX'):
            if sum(record['count'] for record in value[grouping].values()) != len(q):
                raise AssertionError('Source TX/RX/day groups do not partition source records')
    _raises(lambda: source_summary(q, L, 'U_s'))
    _raises(lambda: source_summary(q.requires_grad_(), L, 'L_s'))
    q = q.detach()

    records = []; actual_probe_count = 0; actual_update_count = 0; actual_confusions = 0
    before_probe_rng = _rng_state(device)
    with preserve_rng(device):
        for row in d.rows():
            probes = run_probes(features, features, L, V, row['model_seed'], device)
            for level, heads in probes['layers'].items():
                if len(heads) != 7:
                    raise AssertionError('Unconditional and six TX head families differ')
                for record in heads.values():
                    actual_probe_count += 1; actual_update_count += record['fitted_steps']
                    expected_train = 180 if record['TX'] is None else 30
                    if record['normalization_fit_count'] != expected_train or record['training']['count'] != expected_train:
                        raise AssertionError('Held fold2 entered probe fitting')
                    for role in ('training', 'L_fold2', 'source_V'):
                        evaluated = record[role]; confusion = torch.tensor(evaluated['confusion'])
                        actual_confusions += 1
                        if int(confusion.sum()) != evaluated['count'] or int(confusion.diag().sum()) != evaluated['correct_count']:
                            raise AssertionError('Independent probe confusion/count differs')
                        if evaluated['accuracy'] != 1.:
                            raise AssertionError('Synthetic linearly decodable RX not recovered')
            records.append(dict(row_id=row['row_id'], source_summary=summaries, rx_probes=probes))
    _assert_rng(before_probe_rng, device)
    result = analysis(records)
    if (actual_probe_count, actual_update_count, actual_confusions) != (189, 18900, 567):
        raise AssertionError('Fixed nine-row source probe matrix incomplete')
    if len(result['rows']) != 9 or len(result['three_seed']) != 3 or len(result['paired']) != 6:
        raise AssertionError('Full fixed all-arm/seed probe summary incomplete')
    for item in result['rows']:
        for level in d.FEATURES:
            value = item['RX_probes'][level]['L_fold2']['TX_conditional']
            if value['count'] != 90 or value['correct_count'] != 90 or value['accuracy'] != 1.:
                raise AssertionError('Same-TX micro aggregation differs')
    _raises(lambda: analysis(records[:-1]))

    # Actual native source-only architecture fixture confirms its layer interfaces.
    from experiments.cvs_conditional_rx_adversary.model import build
    model = build(d.source_config(d.rows()[0]), device).eval().requires_grad_(False)
    before = snapshot(model); before_rng = _rng_state(device)
    sample = {name: value[:12] if torch.is_tensor(value) else value[:12] for name, value in L.items()}
    with preserve_rng(device):
        torch.rand(3); random.random(); np.random.rand(3)
        probabilities, captured = scan([model], sample, 'L_s', device)
        frozen = assert_unchanged(model, before)
    _assert_rng(before_rng, device)
    if probabilities.shape != (1, 12, 6) or [captured[0][level].shape[1] for level in d.FEATURES] != list(d.DIMENSIONS):
        raise AssertionError('Actual native source descriptor dimensions differ')
    _raises(lambda: scan([model], sample, 'U_s', device))
    guarded = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)).eval().requires_grad_(False)
    original = snapshot(guarded)
    guarded[0].weight.data[0, 0] += 1
    _raises(lambda: assert_unchanged(guarded, original))
    guarded.load_state_dict(original['state']); guarded[1].running_mean[0] += 1
    _raises(lambda: assert_unchanged(guarded, original))
    guarded.load_state_dict(original['state']); guarded.train()
    _raises(lambda: assert_unchanged(guarded, original))
    guarded.eval(); guarded[0].weight.grad = torch.zeros_like(guarded[0].weight)
    _raises(lambda: assert_unchanged(guarded, original))

    spec = dict(run_id=d.RUN, rows=[dict(row_id=row['row_id'], config=d.config(row)) for row in d.rows()])
    with patch.object(d, 'read', return_value=spec):
        verify_registration()
    invalid = copy.deepcopy(spec); invalid['rows'][0]['config']['probe']['steps'] = 101
    with patch.object(d, 'read', return_value=invalid):
        _raises(verify_registration)
    invalid = copy.deepcopy(spec); invalid['rows'][-1] = copy.deepcopy(invalid['rows'][0])
    with patch.object(d, 'read', return_value=invalid):
        _raises(verify_registration)
    _raises(lambda: d.config(dict(d.rows()[0], arm='OTHER')))
    invalid = copy.deepcopy(d.config(d.rows()[0])); invalid['U_cache_opened'] = True
    _raises(lambda: d.validate(invalid))
    temporary_root = d.ROOT / 'local_artifacts'; temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cvs_source_conditional_rx_', dir=temporary_root) as folder:
        path = Path(folder) / 'new.json'; _write_new(path, dict(ok=True))
        _raises(lambda: _write_new(path, dict(ok=False)), FileExistsError)
    return dict(status='PASS', device=device, synthetic_only=True,
                actual_source_cache_opened=False, target_query_access=False, U_cache_opened=False,
                fixed_rows=9, probe_count=actual_probe_count, probe_optimizer_updates=actual_update_count,
                probe_diagnostic_confusions=actual_confusions, source_groups_recounted=True,
                held_fold2_not_fitted=True, detached_features_required=True,
                native_capture_shapes=list(d.DIMENSIONS), identity_frozen=frozen,
                parameters_buffers_modes_gradients_mutation_rejected=True, RNG_preserved=True,
                invalid_role_labels_metadata_config_and_registration_rejected=True,
                output_overwrite_rejected=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = checks(args.device)
    if args.output:
        _write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
