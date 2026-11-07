import json
import sys

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal import ground, train_source
from experiments.phase1_dura_orthogonal.phase_equivariant_dura import PhaseEquivariantDura
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE


def test_ground_checks_phase_recovery_contract(monkeypatch, tmp_path):
    model = PhaseEquivariantDura(width=32)
    common = dict(commit='source-commit', epochs=5, architecture='phase_equivariant_dura',
                  objective='wide_phase_equivariant_guard', TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True, source_fit_size=WIDE_FIT_SIZE,
                  DURA_training_reads=1, phase_residual_scale=.25, phase_equivariant_inputs='received_anchor_innovation',
                  phase_complex_kernel=5, phase_dilations=[1, 2, 4], phase_noise_gate=True)
    summary = dict(status='SOURCE_TRAINED', fit_ids=WIDE_FIT_SIZE, **common)
    state = dict(schema=SCHEMA, fit_ids=np.arange(WIDE_FIT_SIZE), width=32, model=model.state_dict(), **common)
    monkeypatch.setattr(ground.core, 'read', lambda _: summary)
    monkeypatch.setattr(ground.torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(ground, 'frozen_dura', lambda *args: model)
    monkeypatch.setattr(ground.core, 'teacher', lambda *args: (model, None))
    loaded, _, _ = ground.load_fixed(tmp_path, 'source-run', 'source-commit')
    assert isinstance(loaded, PhaseEquivariantDura)
    state['phase_residual_scale'] = .5
    with pytest.raises(ValueError, match='phase-equivariant'):
        ground.load_fixed(tmp_path, 'source-run', 'source-commit')
    state['phase_residual_scale'] = .25
    state['TX_labels_read'] = 1
    with pytest.raises(ValueError, match='provenance'):
        ground.load_fixed(tmp_path, 'source-run', 'source-commit')


def test_source_smoke_trains_both_complex_head_components(monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(42)
    data = tuple(torch.randn(96, 2, 256, generator=generator) for _ in range(3))
    data = (*data, torch.full((96,), .1), torch.randn(96, 2, 256, generator=generator))
    models = []
    class Captured(PhaseEquivariantDura):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            models.append(self)
    monkeypatch.setattr(train_source, 'PhaseEquivariantDura', Captured)
    monkeypatch.setattr(train_source, 'training_environments', lambda *a: None)
    monkeypatch.setattr(train_source, 'source_inputs', lambda root: (np.arange(32), None, None, None, None, None))
    monkeypatch.setattr(train_source, 'load_source_training', lambda *a, **k: (None, None, None, np.arange(32)))
    monkeypatch.setattr(train_source, 'frozen_dura', lambda *a: torch.nn.Identity().requires_grad_(False))
    monkeypatch.setattr(train_source, 'prepare', lambda *a, **k: (data, {'synthetic': 96}))
    out = tmp_path/'smoke'
    monkeypatch.setattr(sys, 'argv', ['train_source', '--root', str(tmp_path), '--out', str(out), '--commit', 'test',
        '--smoke', '--smoke-fit', '32', '--device', 'cpu', '--architecture', 'phase_equivariant_dura',
        '--objective', 'wide_phase_equivariant_guard'])
    train_source.main()
    summary = json.loads((out/'smoke.json').read_text(encoding='utf8'))
    assert summary['status'] == 'SOURCE_TRAINING_SMOKE_PASS' and summary['phase_noise_gate'] is True
    assert summary['TX_labels_read'] == summary['F_weights_read'] == summary['target_reads'] == 0
    assert models[0].head.real.detach().abs().sum() > 0 and models[0].head.imag.detach().abs().sum() > 0
