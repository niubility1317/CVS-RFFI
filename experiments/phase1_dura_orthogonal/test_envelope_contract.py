import json
import sys

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal import ground, train_source
from experiments.phase1_dura_orthogonal.monotone_envelope import DuraMonotoneEnvelope, ENVELOPE_WEIGHT
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE


def test_ground_accepts_only_matching_envelope_source_checkpoint(monkeypatch, tmp_path):
    model = DuraMonotoneEnvelope(width=32)
    common = dict(commit='source-commit', epochs=5, architecture='dura_envelope', objective='wide_envelope_guard',
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  DURA_frozen=True, source_fit_size=WIDE_FIT_SIZE, DURA_training_reads=1,
                  source_envelope_weight=ENVELOPE_WEIGHT, source_envelope_quantiles=16,
                  envelope_segments=8, envelope_segment_width=.5)
    summary = dict(status='SOURCE_TRAINED', fit_ids=WIDE_FIT_SIZE, **common)
    state = dict(schema=SCHEMA, fit_ids=np.arange(WIDE_FIT_SIZE), width=32, model=model.state_dict(), **common)
    monkeypatch.setattr(ground.core, 'read', lambda _: summary)
    monkeypatch.setattr(ground.torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(ground, 'frozen_dura', lambda *args: model)
    monkeypatch.setattr(ground.core, 'teacher', lambda *args: (model, None))
    loaded, _, _ = ground.load_fixed(tmp_path, 'source-run', 'source-commit')
    assert isinstance(loaded, DuraMonotoneEnvelope)
    state['source_envelope_weight'] = 7.
    with pytest.raises(ValueError, match='envelope'):
        ground.load_fixed(tmp_path, 'source-run', 'source-commit')


def test_envelope_trainer_smoke_updates_both_branches(monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(42)
    data = tuple(torch.randn(96, 2, 256, generator=generator) for _ in range(3))
    data = (*data, torch.full((96,), .1), torch.randn(96, 2, 256, generator=generator))
    models = []
    class Captured(DuraMonotoneEnvelope):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            models.append(self)
    monkeypatch.setattr(train_source, 'DuraMonotoneEnvelope', Captured)
    monkeypatch.setattr(train_source, 'training_environments', lambda *a: None)
    monkeypatch.setattr(train_source, 'source_inputs', lambda root: (np.arange(32), None, None, None, None, None))
    monkeypatch.setattr(train_source, 'load_source_training', lambda *a, **k: (None, None, None, np.arange(32)))
    monkeypatch.setattr(train_source, 'frozen_dura', lambda *a: torch.nn.Identity().requires_grad_(False))
    monkeypatch.setattr(train_source, 'prepare', lambda *a, **k: (data, {'synthetic': 96}))
    out = tmp_path/'smoke'
    monkeypatch.setattr(sys, 'argv', ['train_source', '--root', str(tmp_path), '--out', str(out), '--commit', 'test',
        '--smoke', '--smoke-fit', '32', '--device', 'cpu', '--architecture', 'dura_envelope', '--objective', 'wide_envelope_guard'])
    train_source.main()
    summary = json.loads((out/'smoke.json').read_text(encoding='utf8'))
    assert summary['status'] == 'SOURCE_TRAINING_SMOKE_PASS' and summary['source_envelope_weight'] == ENVELOPE_WEIGHT
    assert models[0].base.wave.weight.detach().abs().sum() > 0
    assert models[0].envelope_head.weight.detach().abs().sum() > 0
