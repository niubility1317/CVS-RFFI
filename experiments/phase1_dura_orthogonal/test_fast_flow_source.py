import json
import sys

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal import train_source
from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow


def test_fast_flow_option_preserves_actual_trainer_smoke(monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(42)
    data = tuple(torch.randn(96, 2, 256, generator=generator) for _ in range(3))
    data = (*data, torch.full((96,), .1), torch.randn(96, 2, 256, generator=generator))
    models = []

    class CapturedFlow(ConditionalIQFlow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            models.append(self)

    monkeypatch.setattr(train_source, 'ConditionalIQFlow', CapturedFlow)
    monkeypatch.setattr(train_source, 'training_environments', lambda *a: None)
    monkeypatch.setattr(train_source, 'source_inputs', lambda root: (np.arange(32), None, None, None, None, None))
    monkeypatch.setattr(train_source, 'load_source_training', lambda *a, **k: (None, None, None, np.arange(32)))
    monkeypatch.setattr(train_source, 'frozen_dura', lambda *a: torch.nn.Identity().requires_grad_(False))
    monkeypatch.setattr(train_source, 'prepare', lambda *a, **k: (data, {'synthetic': 96}))
    outputs = []
    for fast in (False, True):
        out = tmp_path / ('fast' if fast else 'original')
        argv = ['train_source', '--root', str(tmp_path), '--out', str(out), '--commit', 'test',
                '--smoke', '--smoke-fit', '32', '--device', 'cpu', '--architecture', 'conditional_iq_flow',
                '--objective', 'wide_conditional_flow']
        if fast:
            argv.append('--fast-flow-guards')
        monkeypatch.setattr(sys, 'argv', argv)
        train_source.main()
        outputs.append(json.loads((out / 'smoke.json').read_text(encoding='utf8')))
    assert outputs[0]['losses'] == outputs[1]['losses']
    assert not outputs[0]['fast_flow_guards'] and outputs[1]['fast_flow_guards']
    assert all(torch.equal(value, models[1].state_dict()[name]) for name, value in models[0].state_dict().items())


def test_fast_flow_option_rejects_other_objectives(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['train_source', '--root', str(tmp_path), '--out', str(tmp_path / 'x'),
        '--commit', 'test', '--fast-flow-guards'])
    with pytest.raises(ValueError, match='fast flow'):
        train_source.main()
