import json
import sys

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.antithetic_iq_flow import AntitheticIQFlow
from experiments.phase1_dura_orthogonal.antithetic_readout import corrected_paths, probability_mean, MODE
from experiments.phase1_dura_orthogonal import ground, antithetic_readout_controller as runner


def test_paths_preserve_zero_head_clean_gate_and_original_iq_mean():
    torch.manual_seed(56)
    model = AntitheticIQFlow(width=4).eval()
    x, anchor, direction = (torch.randn(3, 2, 256) for _ in range(3))
    variance = torch.tensor([0., .1, .2])
    with torch.no_grad():
        paths = corrected_paths(model, x, anchor, direction, variance)
        assert all(torch.equal(path, anchor) for path in paths)
        model.out.weight.normal_(std=.05)
        paths = corrected_paths(model, x, anchor, direction, variance)
        assert all(torch.equal(path[0], anchor[0]) for path in paths)
        torch.testing.assert_close(.5*(paths[0]+paths[1]), model(x, anchor, direction, variance), rtol=1e-6, atol=1e-7)
        order = torch.tensor([2, 0, 1])
        reordered = corrected_paths(model, x[order], anchor[order], direction[order], variance[order])
        for original, permuted in zip(paths, reordered):
            torch.testing.assert_close(original[order], permuted, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match='Antithetic'):
        corrected_paths(torch.nn.Identity(), x, anchor, direction, variance)


def test_probabilities_average_after_default_head_not_before_softmax():
    a = torch.tensor([[5., 0., 0., 0., 0., 0.]])
    b = torch.tensor([[0., 2., 0., 0., 0., 0.]])
    actual = probability_mean([a, b])
    assert torch.equal(actual, .5*(a.softmax(1)+b.softmax(1)))
    assert not torch.allclose(actual, (.5*(a+b)).softmax(1))
    torch.testing.assert_close(actual.sum(1), torch.ones(1))
    with pytest.raises(ValueError, match='finite'):
        probability_mean([a, b*float('nan')])
    with pytest.raises(ValueError, match='Two'):
        probability_mean([a])


def test_predict_rejects_different_smoke_mode_before_reading_target(monkeypatch, tmp_path):
    monkeypatch.setattr(ground.core, 'read', lambda _: {'status': 'PASS', 'commit': 'new', 'test_IQ_reads': 0,
        'source': 'source', 'source_commit': 'old', 'inference_mode': 'single'})
    monkeypatch.setattr(ground, 'load_fixed', lambda *a: pytest.fail('must reject before loading model'))
    with pytest.raises(ValueError, match='inference mode'):
        ground.predict(tmp_path, tmp_path/'out', 'source', 'new', source_commit='old', inference_mode=MODE)


def test_failed_source_smoke_prevents_prediction_and_scoring(monkeypatch, tmp_path):
    parent = tmp_path/'runs'/(runner.SOURCE+'-controller')
    parent.mkdir(parents=True)
    (parent/'status.json').write_text('{}', encoding='utf8')
    source = tmp_path/'runs'/runner.SOURCE
    source.mkdir()
    (source/'source_summary.json').write_text('{}', encoding='utf8')
    monkeypatch.setattr(sys, 'argv', ['runner', '--root', str(tmp_path), '--out', str(tmp_path/'new'), '--commit', 'new'])
    commands = []
    def fail_smoke(command, **kwargs):
        commands.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert command[6] == 'smoke'
        raise RuntimeError('failed source smoke')
    monkeypatch.setattr(runner.subprocess, 'run', fail_smoke)
    with pytest.raises(RuntimeError, match='failed source smoke'):
        runner.main()
    assert len(commands) == 1
    assert json.loads((tmp_path/'new-controller/status.json').read_text())['state'] == 'FAILED'


@pytest.mark.parametrize('mode', ['single', MODE])
def test_actual_prediction_manifest_is_accepted_by_independent_scorer(monkeypatch, tmp_path, mode):
    ids = np.arange(8)
    inputs = tmp_path/'synthetic_inputs'
    inputs.mkdir()
    reference = tmp_path/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
    reference.mkdir(parents=True)
    np.save(reference/'physical_ids.npy', ids)
    for view in ground.core.VIEWS:
        np.save(inputs/(view+'.npy'), np.zeros((8, 2, 256), dtype=np.float32))
        np.save(reference/(view+'.raw.npy'), np.zeros(8, dtype=np.int64))
    out = tmp_path/'out'
    out.mkdir()
    ground.core.write(out/'prediction_smoke.json', dict(status='PASS', commit='new', test_IQ_reads=0,
        source='source', source_commit='old', inference_mode=mode))
    monkeypatch.setattr(ground, 'load_fixed', lambda *a: (None, None, None))
    monkeypatch.setattr(ground.core, 'reference', lambda _: (inputs, ids))
    monkeypatch.setattr(ground, 'decisions', lambda x, *a, **k: [torch.zeros(len(x), dtype=torch.int64)]*3)
    ground.predict(tmp_path, out, 'source', 'new', source_commit='old', inference_mode=mode)
    truth = tmp_path/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    truth.parent.mkdir(parents=True)
    ground.core.write(truth, {str(i): dict(label=0, receiver='synthetic', day='synthetic') for i in ids})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    ground.core.score(tmp_path, out, 'new')
    assert ground.core.read(out/'flow/test_summary.json')['status'] == 'ANALYZED'
    assert ground.core.read(out/'flow/inference.json')['inference_mode'] == mode
