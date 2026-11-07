import json
import sys

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal import ground, train_source
from experiments.phase1_dura_orthogonal.quartic_recovery import DuraQuarticRecovery, QUARTIC_WEIGHT
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE


def test_ground_rejects_mismatched_quartic_source_or_label_use(monkeypatch, tmp_path):
    model = DuraQuarticRecovery(width=32)
    common = dict(commit='source-commit', epochs=5, architecture='dura_quartic', objective='wide_quartic_guard',
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  DURA_frozen=True, source_fit_size=WIDE_FIT_SIZE, DURA_training_reads=1,
                  source_quartic_weight=QUARTIC_WEIGHT, quartic_offsets=[1, 2, 4, 8], quartic_branch_scale=.1)
    summary = dict(status='SOURCE_TRAINED', fit_ids=WIDE_FIT_SIZE, **common)
    state = dict(schema=SCHEMA, fit_ids=np.arange(WIDE_FIT_SIZE), width=32, model=model.state_dict(), **common)
    monkeypatch.setattr(ground.core, 'read', lambda _: summary)
    monkeypatch.setattr(ground.torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(ground, 'frozen_dura', lambda *args: model)
    monkeypatch.setattr(ground.core, 'teacher', lambda *args: (model, None))
    loaded, _, _ = ground.load_fixed(tmp_path, 'source-run', 'source-commit')
    assert isinstance(loaded, DuraQuarticRecovery)
    state['quartic_offsets'] = [1, 2, 4]
    with pytest.raises(ValueError, match='quartic'):
        ground.load_fixed(tmp_path, 'source-run', 'source-commit')
    state['quartic_offsets'] = [1, 2, 4, 8]
    state['TX_labels_read'] = 1
    with pytest.raises(ValueError, match='provenance'):
        ground.load_fixed(tmp_path, 'source-run', 'source-commit')


def test_trainer_smoke_updates_base_and_quartic_branch(monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(42)
    data = tuple(torch.randn(96, 2, 256, generator=generator) for _ in range(3))
    data = (*data, torch.full((96,), .1), torch.randn(96, 2, 256, generator=generator))
    models = []

    class Captured(DuraQuarticRecovery):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            models.append(self)

    monkeypatch.setattr(train_source, 'DuraQuarticRecovery', Captured)
    monkeypatch.setattr(train_source, 'training_environments', lambda *a: None)
    monkeypatch.setattr(train_source, 'source_inputs', lambda root: (np.arange(32), None, None, None, None, None))
    monkeypatch.setattr(train_source, 'load_source_training', lambda *a, **k: (None, None, None, np.arange(32)))
    monkeypatch.setattr(train_source, 'frozen_dura', lambda *a: torch.nn.Identity().requires_grad_(False))
    monkeypatch.setattr(train_source, 'prepare', lambda *a, **k: (data, {'synthetic': 96}))
    out = tmp_path/'smoke'
    monkeypatch.setattr(sys, 'argv', ['train_source', '--root', str(tmp_path), '--out', str(out), '--commit', 'test',
        '--smoke', '--smoke-fit', '32', '--device', 'cpu', '--architecture', 'dura_quartic', '--objective', 'wide_quartic_guard'])
    train_source.main()
    summary = json.loads((out/'smoke.json').read_text(encoding='utf8'))
    assert summary['status'] == 'SOURCE_TRAINING_SMOKE_PASS' and summary['source_quartic_weight'] == QUARTIC_WEIGHT
    assert models[0].base.wave.weight.detach().abs().sum() > 0
    assert models[0].spectral_head.weight.detach().abs().sum() > 0
