import json
import sys

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal import ground, train_source
from experiments.phase1_dura_orthogonal.antithetic_iq_flow import AntitheticIQFlow
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE


def test_ground_checks_antithetic_training_contract(monkeypatch, tmp_path):
    model = AntitheticIQFlow(width=32)
    common = dict(commit='source-commit', epochs=5, architecture='antithetic_iq_flow', objective='wide_antithetic_flow',
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  DURA_frozen=True, source_fit_size=WIDE_FIT_SIZE, DURA_training_reads=1,
                  flow_steps=4, flow_time_seed=2026110713, flow_path='noisy_anchor_to_clean_linear',
                  flow_latent_seed=2026110723, flow_train_latent_seed=2026110724, flow_latent_scale=.5,
                  flow_max_relative_std=.25, flow_antithetic_paths=2, flow_inference_gate='dura_noise_gate',
                  source_random_transfer='epoch')
    summary = dict(status='SOURCE_TRAINED', fit_ids=WIDE_FIT_SIZE, **common)
    state = dict(schema=SCHEMA, fit_ids=np.arange(WIDE_FIT_SIZE), width=32, model=model.state_dict(), **common)
    monkeypatch.setattr(ground.core, 'read', lambda _: summary)
    monkeypatch.setattr(ground.torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(ground, 'frozen_dura', lambda *args: model)
    monkeypatch.setattr(ground.core, 'teacher', lambda *args: (model, None))
    loaded, _, _ = ground.load_fixed(tmp_path, 'source-run', 'source-commit')
    assert isinstance(loaded, AntitheticIQFlow)
    state['flow_antithetic_paths'] = 1
    with pytest.raises(ValueError, match='antithetic'):
        ground.load_fixed(tmp_path, 'source-run', 'source-commit')
    state['flow_antithetic_paths'] = 2
    state['TX_labels_read'] = 1
    with pytest.raises(ValueError, match='provenance'):
        ground.load_fixed(tmp_path, 'source-run', 'source-commit')


def test_trainer_smoke_updates_flow_on_source_only(monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(42)
    data = tuple(torch.randn(96, 2, 256, generator=generator) for _ in range(3))
    data = (*data, torch.full((96,), .1), torch.randn(96, 2, 256, generator=generator))
    models = []
    class Captured(AntitheticIQFlow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            models.append(self)
    monkeypatch.setattr(train_source, 'AntitheticIQFlow', Captured)
    monkeypatch.setattr(train_source, 'training_environments', lambda *a: None)
    monkeypatch.setattr(train_source, 'source_inputs', lambda root: (np.arange(32), None, None, None, None, None))
    monkeypatch.setattr(train_source, 'load_source_training', lambda *a, **k: (None, None, None, np.arange(32)))
    monkeypatch.setattr(train_source, 'frozen_dura', lambda *a: torch.nn.Identity().requires_grad_(False))
    monkeypatch.setattr(train_source, 'prepare', lambda *a, **k: (data, {'synthetic': 96}))
    out = tmp_path/'smoke'
    monkeypatch.setattr(sys, 'argv', ['train_source', '--root', str(tmp_path), '--out', str(out), '--commit', 'test',
        '--smoke', '--smoke-fit', '32', '--device', 'cpu', '--architecture', 'antithetic_iq_flow', '--objective', 'wide_antithetic_flow'])
    train_source.main()
    summary = json.loads((out/'smoke.json').read_text(encoding='utf8'))
    assert summary['status'] == 'SOURCE_TRAINING_SMOKE_PASS' and summary['flow_antithetic_paths'] == 2
    assert summary['TX_labels_read'] == summary['F_weights_read'] == summary['target_reads'] == 0
    assert models[0].out.weight.detach().abs().sum() > 0
