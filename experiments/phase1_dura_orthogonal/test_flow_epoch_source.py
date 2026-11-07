import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal import train_source
from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow


def test_epoch_option_rejects_other_objectives(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['train_source', '--root', str(tmp_path), '--out', str(tmp_path/'x'),
                                      '--commit', 'test', '--epoch-flow-times'])
    with pytest.raises(ValueError, match='[Ee]poch'):
        train_source.main()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Source flow runs on GPU')
def test_actual_eager_and_graph_trainers_match_epoch_time_prefetch(monkeypatch, tmp_path):
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
    summaries = []
    for execution, epoch in (('eager', False), ('eager', True), ('graph', False), ('graph', True)):
        out = tmp_path/(execution+str(epoch))
        argv = ['train_source', '--root', str(tmp_path), '--out', str(out), '--commit', 'test',
                '--smoke', '--smoke-fit', '32', '--device', 'cuda:0', '--deterministic',
                '--architecture', 'conditional_iq_flow', '--objective', 'wide_conditional_flow',
                '--cuda-graph-flow' if execution == 'graph' else '--fast-flow-guards']
        if epoch:
            argv.append('--epoch-flow-times')
        monkeypatch.setattr(sys, 'argv', argv)
        train_source.main()
        summaries.append(json.loads((out/'smoke.json').read_text(encoding='utf8')))
    assert all(s['losses'] == summaries[0]['losses'] for s in summaries)
    assert [s['source_flow_time_transfer'] for s in summaries] == ['batch', 'epoch', 'batch', 'epoch']
    assert all(torch.equal(a, b) for model in models[1:] for a, b in zip(models[0].parameters(), model.parameters()))
