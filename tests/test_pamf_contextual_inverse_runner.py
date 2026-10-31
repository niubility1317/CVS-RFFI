from types import ModuleType, SimpleNamespace
import sys
import json

import numpy as np
import pytest
import torch

from experiments.pamf_dura_direct_train import contextual_inverse as trial
from experiments.pamf_dura_direct_train import contextual_recovery as recovery


def test_checkpoint_roundtrip_and_provenance(tmp_path):
    model = trial.make_model()
    path = tmp_path / 'checkpoint.pt'
    commit = 'a' * 40
    trial.save_model(path, model, trial.ARM, 0, 0, commit)
    loaded = trial.load_model(path, 'cpu', arm=trial.ARM, commit=commit, epoch=0)
    x = torch.randn(1, 2, 256)
    torch.testing.assert_close(model(x), loaded(x), rtol=0, atol=0)
    with pytest.raises(ValueError):
        trial.load_model(path, 'cpu', arm=trial.ARM, commit='b' * 40, epoch=0)
    with pytest.raises(ValueError):
        trial.load_model(path, 'cpu', arm=trial.ARM, commit=commit, epoch=30)


def test_objective_and_schedule_are_finite():
    torch.set_num_threads(2)
    model = trial.make_model()
    x, y = torch.randn(2, 2, 256), torch.randn(2, 2, 256)
    loss, terms = trial.objective(model, y, x)
    assert loss.shape == (2,)
    assert set(terms) == {'iq_nmse', 'spectral_error'}
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.spectral_head.weight.grad.abs().sum() > 0
    assert model.time_output.weight.grad.abs().sum() > 0
    assert trial.learning_rate(0) > 0
    assert trial.learning_rate(246) == pytest.approx(.001)
    assert trial.learning_rate(trial.STEPS - 1) == pytest.approx(.00002)
    with pytest.raises(ValueError):
        trial.learning_rate(trial.STEPS)


def test_dispatch_configuration_and_commands():
    c = trial.configuration()
    assert c['updates'] == 7410 and c['epochs'] == 30
    assert not c['training_F_access'] and not c['target_training'] and not c['OE_training']
    trial.configure()
    args = SimpleNamespace(root='/project', commit='a' * 40)
    cmd = trial.base.command(args, 'score', trial.ARM, 'target')
    assert trial.MODULE in cmd and '--role' in cmd and cmd[-1] == 'target'
    assert trial.base.ARMS == (trial.ARM,)


def test_all_source_predictions_satisfy_independent_scorer_contract(tmp_path, monkeypatch):
    from experiments.pamf_dura_direct_train import evaluation
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('cached_coordinate', torch.tensor(1.))
        def forward(self, x):
            # The real frozen sinc cache inspects a buffer version counter.
            _ = self.cached_coordinate._version
            mean = x.mean((1, 2))
            return torch.stack([mean + i for i in range(6)], dim=1)

    monkeypatch.setattr(evaluation, 'reference', lambda p, role: (tmp_path/'inputs', ids, None, None))
    frozen = ModuleType('experiments.pamf_fcrossrx_readout.frozen')
    frozen.load_reader = lambda *args: Reader()
    monkeypatch.setitem(sys.modules, frozen.__name__, frozen)
    monkeypatch.setattr(trial.base, 'tensor', lambda x, device: torch.as_tensor(np.array(x, copy=True)))
    original = trial.load_model
    monkeypatch.setattr(trial, 'load_model', lambda path, device, **kwargs: original(path, 'cpu', **kwargs))
    ids = np.array(['one', 'two'])
    inputs, output = tmp_path/'inputs', tmp_path/'run'
    inputs.mkdir(); output.mkdir()
    row = output/trial.ARM
    row.mkdir()
    for view in trial.base.VIEWS:
        np.save(inputs/(view+'.npy'), np.ones((2, 2, 256), dtype=np.float32))
    args = SimpleNamespace(arm=trial.ARM, commit='b'*40, checkpoint_commit='a'*40)
    trial.save_model(row/'final.pt', trial.make_model(), trial.ARM, 30, trial.STEPS, args.checkpoint_commit)
    trial.base.write_json(row/'training_complete.json', dict(status='SOURCE_TRAINED', epoch=30,
                          updates=trial.STEPS, target_reads=0))
    paths = dict(output=output, reader=tmp_path/'reader', source_cache=tmp_path/'source')
    trial.predict(args, paths, 'source')
    result = evaluation.complete_arrays(row/'source_prediction', ids, arm=trial.ARM,
                                        role='source', commit=args.commit)
    assert set(result) == set(trial.base.VIEWS)
    assert all(set(entry) == {'iq', 'logits', 'raw_logits'} for entry in result.values())
    metadata = row/'source_prediction'/'complete.json'
    complete = json.loads(metadata.read_text(encoding='utf-8'))
    complete['views'] = complete['views'][:-1]
    metadata.write_text(json.dumps(complete), encoding='utf-8')
    with pytest.raises(ValueError):
        evaluation.complete_arrays(row/'source_prediction', ids, arm=trial.ARM,
                                   role='source', commit=args.commit)


def test_recovery_requires_completed_original_and_uses_distinct_code_identity(tmp_path):
    root = tmp_path/'old'
    row = root/trial.ARM
    row.mkdir(parents=True)
    trial.save_model(row/'final.pt', trial.make_model(), trial.ARM, 30, trial.STEPS, recovery.MODEL_COMMIT)
    trial.base.write_json(row/'training_complete.json', dict(status='SOURCE_TRAINED', epoch=30,
        updates=7410, training_F_access=False, target_reads=0, OE_reads=0))
    trial.base.write_json(row/'failure.json', dict(error='RuntimeError: Inference tensors do not track version counter.'))
    trial.base.write_json(row/'matched_profile.json', dict(status='PROFILE_COMPLETE'))
    actual_row, model = recovery.verify_parent(dict(original_output=root))
    assert actual_row == row and not any(p.requires_grad for p in model.parameters())
    args = recovery.model_args('b'*40)
    assert args.commit == 'b'*40 and args.checkpoint_commit == recovery.MODEL_COMMIT
    assert recovery.MODULE in recovery.command(tmp_path, args.commit, 'score', 'target')
    (row/'failure.json').write_text(json.dumps(dict(error='unrelated failure')), encoding='utf-8')
    with pytest.raises(ValueError):
        recovery.verify_parent(dict(original_output=root))


def test_recovery_shares_original_worker_inventory_and_excludes_used_gpu(monkeypatch):
    tokens = recovery.goal_worker_tokens()
    assert len(tokens) > 30
    assert 'pamf_dura_direct_train.fast_lr worker' in tokens
    assert recovery.MODULE+' worker' in tokens
    monkeypatch.setattr(recovery.Path, 'read_bytes', lambda path:b'CUDA_VISIBLE_DEVICES=4\0')
    commands = ['123 python -m experiments.pamf_dura_direct_train.fast_lr worker',
                '124 python -m experiments.pamf_dura_direct_train.contextual_recovery worker',
                '125 unrelated worker']
    count, used = recovery.active_goal_workers(commands, tokens)
    assert count == 2 and used == {4}
    state = {4:dict(free_mb=20000,pids=set()), 6:dict(free_mb=20000,pids=set())}
    assert trial.base.eligible_gpus(state,used,3,[4,6]) == [6]
