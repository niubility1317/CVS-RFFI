from types import ModuleType, SimpleNamespace
import sys
import json

import numpy as np
import pytest
import torch

from experiments.pamf_dura_direct_train import contextual_inverse as trial


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
        def forward(self, x):
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
    args = SimpleNamespace(arm=trial.ARM, commit='a'*40)
    trial.save_model(row/'final.pt', trial.make_model(), trial.ARM, 30, trial.STEPS, args.commit)
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
