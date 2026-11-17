import numpy as np
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from experiments.mopc_adsb_repro.prepare import csil_features, split_rows
from experiments.mopc_adsb_repro.run import correct, regularization, score


def test_split_is_deterministic_disjoint_and_has_paper_stage_sizes():
    ids = np.repeat(np.arange(110), 400 + np.arange(110))
    a = split_rows(ids); b = split_rows(ids)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    ti, ty, qi, qy, selected, counts = a
    assert len(selected) == 100 and len(ti) == 20000
    assert len(set(ti).intersection(qi)) == 0
    assert np.all(ids[ti] == selected[ty]) and np.all(ids[qi] == selected[qy])
    assert np.all(counts[:-1] >= 334)
    assert np.all(np.diff(qy) >= 0)  # stage-prefix query without truth access


def test_csil_preprocessing_matches_scalar_matlab_equations():
    rng = np.random.default_rng(7)
    iq = rng.normal(size=1024) + 1j * rng.normal(size=1024)
    iq[:256] = 0
    amp = np.abs(iq); mu = np.mean(amp); std = np.std(amp, ddof=1)
    clean = amp * ((amp <= mu + 3 * std) & (amp >= mu - 3 * std))
    low, high = np.min(clean), np.max(clean)
    for _ in range(3):
        is_high = np.abs(clean-high) <= np.abs(clean-low)
        high, low = np.mean(clean[is_high]), np.mean(clean[~is_high])
    medoid = np.where(clean >= (high + low) / 2, high, low)
    fft = np.fft.fftshift(np.fft.fft(iq) - np.fft.fft(medoid))
    expected = np.stack([(medoid-clean).reshape(32, 32, order='F'),
                         fft.real.reshape(32, 32, order='F'),
                         fft.imag.reshape(32, 32, order='F')])
    expected = ((expected-expected.min()) / (expected.max()-expected.min())).astype('float32')
    np.testing.assert_allclose(csil_features(iq[None])[0], expected, rtol=1e-6, atol=1e-7)


def test_degenerate_signal_does_not_silently_create_invalid_data():
    with pytest.raises(ValueError, match='Degenerate'):
        csil_features(np.zeros((1, 1024), dtype=np.complex64))


def test_prototype_correction_distinguishes_repository_and_paper_formula():
    old = torch.tensor([[1., 2.], [2., 1.]])
    before = torch.tensor([[2., 3.], [4., -1.]])
    after = before + torch.tensor([[1., 0.], [0., 2.]])
    author = .97*old + .03*(torch.softmax(old @ before.T, 1) @ (after-before))
    paper = .97*old + .03*((F.normalize(old, dim=1) @ F.normalize(before, dim=1).T) @ (after-before))
    torch.testing.assert_close(correct(old, before, after, 'author'), author)
    torch.testing.assert_close(correct(old, before, after, 'paper'), paper)
    assert not torch.allclose(author, paper)


def test_hr_author_norm_and_paper_squared_norm():
    old = torch.nn.Linear(2, 2); new = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for p in old.parameters(): p.zero_()
        for p in new.parameters(): p.fill_(2.)
    assert regularization(new, old, 'author').item() == pytest.approx(4 + 0.5*np.sqrt(8))
    assert regularization(new, old, 'paper').item() == pytest.approx(16 + 0.5*8)


def test_truth_last_task_metrics_and_incomplete_predictions(tmp_path):
    import json
    data = tmp_path / 'data'; data.mkdir()
    predictions = tmp_path / 'predictions'; predictions.mkdir()
    np.save(data / 'query_labels.npy', np.array([0, 1, 50, 51, 75, 76]))
    (data / 'data_manifest.json').write_text(json.dumps({'query_prefix_lengths': [2, 4, 6]}))
    np.save(predictions / 'stage0_pred.npy', [0, 1])
    np.save(predictions / 'stage1_pred.npy', [0, 1, 50, 1])
    np.save(predictions / 'stage2_pred.npy', [0, 1, 50, 51, 0, 76])
    args = SimpleNamespace(data=str(data), predictions=str(predictions), output=str(tmp_path / 'score'))
    with pytest.raises(ValueError, match='not fixed'): score(args)
    (predictions / 'predictions_complete.json').write_text('{}')
    score(args)
    result = json.loads((tmp_path / 'score' / 'metrics.json').read_text())
    assert result['task_accuracy_by_stage'] == [[1.0], [1.0, .5], [1.0, 1.0, .5]]
    assert result['paper_A_avg'] == pytest.approx(5 / 6)
    assert result['average_forgetting'] == pytest.approx(-.25)
    assert result['intransigence'] is None
