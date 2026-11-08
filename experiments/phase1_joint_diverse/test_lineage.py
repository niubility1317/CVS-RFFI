import json

import pytest
import torch

from experiments.phase1_dura_orthogonal.source_diverse_views import diverse_epochs
from experiments.phase1_joint_diverse.ground import verify_schedule


def test_view_rotation_exposure_and_checkpoint_validation(tmp_path):
    schedule = [list(diverse_epochs(i)) for i in range(1, 21)]
    assert sorted(v for row in schedule for v in row) == sorted(list(range(1, 31))*2)
    folder = tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    metadata = dict(source_view_schedule=schedule, actual_fit_ids=32768,
                    source_counts_by_epoch=[{'clean': 98304} for _ in schedule])
    (folder/'source_summary.json').write_text(json.dumps(metadata), encoding='utf8')
    torch.save(metadata, folder/'final.pt')
    assert verify_schedule(tmp_path, 'source') == schedule
    metadata['source_view_schedule'][0] = [21, 22, 23]
    torch.save(metadata, folder/'final.pt')
    with pytest.raises(ValueError, match='source-view'):
        verify_schedule(tmp_path, 'source')


def test_smoke_rejects_missing_source_exposures(tmp_path):
    folder = tmp_path/'runs'/'source-training-smoke'
    folder.mkdir(parents=True)
    metadata = dict(source_view_schedule=[[1, 2, 3]], actual_fit_ids=64,
                    source_counts_by_epoch=[{'clean': 191}])
    (folder/'smoke.json').write_text(json.dumps(metadata), encoding='utf8')
    torch.save(metadata, folder/'smoke.pt')
    with pytest.raises(ValueError, match='exposure'):
        verify_schedule(tmp_path, 'source', preflight=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke requires a GPU')
def test_gpu_smoke_runs_six_updates_with_three_view_source(monkeypatch, tmp_path):
    import sys
    import types
    import numpy as np
    from experiments.phase1_joint_diverse import train

    source = tmp_path/'cache'
    (source/'train').mkdir(parents=True)
    roles = np.array(['L_s']*32 + ['U_s']*(32768-32))
    np.save(source/'train'/'roles.npy', roles)
    fit = np.arange(32768)
    monkeypatch.setattr(train, 'training_environments', lambda *a: None)
    monkeypatch.setattr(train, 'source_inputs', lambda root: (np.array([], dtype=int), None, None, None, None, None))
    monkeypatch.setattr(train, 'load_source_training', lambda *a, **k: (source, source, None, fit))
    monkeypatch.setattr(train, 'frozen_dura', lambda *a: torch.nn.Identity())
    epochs_seen = []
    def fake_prepare(root, dura, clean, fresh, ids, device, batch, source_epochs):
        epochs_seen.append(source_epochs)
        n = len(ids)*3
        signal = torch.ones((n, 2, 256))
        return (signal, torch.zeros_like(signal), torch.zeros_like(signal),
                torch.ones(n), signal), {'clean': n}
    monkeypatch.setattr(train, 'prepare', fake_prepare)
    monkeypatch.setattr(train, '_actions', lambda dura, received, variance:
                        (torch.zeros((len(received), 1), device=received.device),)*3)
    module = types.ModuleType('experiments.pamf_action_restore.operators')
    module.Action = lambda *args: args
    monkeypatch.setitem(sys.modules, 'experiments.pamf_action_restore.operators', module)
    class FakeRecovery(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.output = torch.nn.Conv1d(2, 2, 1, bias=False)
            torch.nn.init.zeros_(self.output.weight)
        def forward(self, y, anchor, direction, v, action, **kwargs):
            return anchor + self.output(y)
    monkeypatch.setattr(train, 'JointParticleRecovery', FakeRecovery)
    out = tmp_path/'smoke'
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR', str(tmp_path/'torch-cache'))
    train.run(tmp_path, out, 'test', smoke=True)
    data = json.loads((out/'smoke.json').read_text(encoding='utf8'))
    assert epochs_seen == [(1, 2, 3)]
    assert data['actual_fit_ids'] == 64 and data['fit_ids'] == 32768
    assert data['source_counts_by_epoch'] == [{'clean': 192}]
    assert data['source_view_schedule'] == [[1, 2, 3]]
    state = torch.load(out/'smoke.pt', weights_only=True)
    assert torch.count_nonzero(state['model']['output.weight']) > 0
