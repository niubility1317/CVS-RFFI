from copy import deepcopy
import sys

import pytest
import torch

from experiments.pamf_repl_source_adapt import full_suite, receiver_balance


def test_all_settings_have_three_matched_rows_and_only_exact_reuse():
    rows = full_suite.rows()
    assert len({row['row_id'] for row in rows}) == len(rows) == 33
    for setting in full_suite.paper.PROTOCOLS:
        matched = [row for row in rows if row['setting'] == setting]
        assert tuple(row['model_seed'] for row in matched) == full_suite.paper.SEEDS
    assert len(full_suite.new_rows()) == 15
    assert sum(row['reuses_original'] for row in rows) == 6
    assert sum(row['reuse_run'] is not None for row in rows) == 12
    assert all(row['weight'] == 6 and row['route'] == 'joint' and row['views'] == ['clean'] for row in rows)
    assert full_suite.GPUS == (0, 4, 5, 6)


def test_two_receivers_leave_positive_targets_identical_even_if_imbalanced():
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    rx = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1])
    distribution, metrics = receiver_balance.positive_target(labels, rx)
    assert distribution is None
    assert metrics['affected'] == 0


def test_only_registered_source_configuration_can_train():
    row = full_suite.new_rows()[0]
    c = dict(source_adaptation=row, run_id=full_suite.RUN, row_id=row['row_id'],
             model_seed=row['model_seed'], protocol_id=row['protocol_id'],
             paper_row={'row_id': full_suite.transfer.original_row(row)},
             epochs=50, optimizer_updates=11100, source_rxs=[0, 1, 2],
             hard_pl={'threshold': .95, 'weight': 6.}, hard_pl_gradient_guard={'weight': 6.},
             output_root=f"test/runs/{full_suite.RUN}/F/{row['row_id']}/source/segment30")
    full_suite.validate(c)
    for wrong in ({'target_truth': 'forbidden'}, {'model_seed': 0}, {'source_rxs': [0, 1]},
                  {'hard_pl_gradient_guard': {'weight': 3.}}):
        with pytest.raises(ValueError, match='Only predeclared'):
            full_suite.validate(dict(deepcopy(c), **wrong))


def test_dispatch_activates_frozen_source_runtime_before_prepare(monkeypatch):
    from experiments.pamf_repl_completion import runner

    config = {'run_id': full_suite.RUN}
    calls = []
    monkeypatch.setattr(sys, 'argv', ['full_suite', 'dispatch', '--config', 'runtime.json'])
    monkeypatch.setattr(full_suite, 'read', lambda path: config)
    monkeypatch.setattr(runner, 'activate', lambda c: calls.append(('activate', c)))
    monkeypatch.setattr(full_suite.transfer, 'dispatch', lambda c, **kwargs: calls.append(('dispatch', c)))
    full_suite.main()
    assert calls == [('activate', config), ('dispatch', config)]
