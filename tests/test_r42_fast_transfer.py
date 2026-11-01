import numpy as np
import pytest
import torch


@pytest.mark.parametrize('readonly,strided', [(False, False), (True, False), (False, True), (True, True)])
def test_numpy_tensor_transfer_preserves_values_and_ownership(readonly, strided):
    from experiments.pamf_phase2_practical.fast_transfer import to_tensor, to_numpy

    value = np.arange(72, dtype=np.float32).reshape(6, 12)
    if strided:
        value = value[:, ::-2]
    if readonly:
        value.flags.writeable = False
    tensor = to_tensor(value, 'cpu', np.float32)
    result = to_numpy(tensor, np.float32)
    np.testing.assert_array_equal(result, value)
    assert result.flags.writeable
    result.flat[0] = -99
    assert tensor.flatten()[0].item() != -99


def test_orbit_fast_transfer_preserves_all_eight_views(monkeypatch):
    from experiments.pamf_phase2_practical import phase_orbit

    def fake_capture(reader, x):
        base = x.mean((1, 2), keepdim=False) + 2
        features = {name: base[:, None].repeat(1, 160) for name in ('z_id','t_emb','f_emb','pa_local')}
        features['sig_views'] = base[:, None].repeat(1, 336)
        return features, base[:, None].repeat(1, 6)
    monkeypatch.setattr(phase_orbit, 'capture', fake_capture)
    raw = np.random.default_rng(13).normal(size=(3, 2, 256)).astype(np.float32)
    kwargs = dict(reader=None, raw=raw, device='cpu', batch=2,
                  aggregate=phase_orbit.ensemble_moments, return_phase_logits=True, phase_eighths=True)
    original, old_logits = phase_orbit.extract(**kwargs)
    actual, new_logits = phase_orbit.extract(**kwargs, fast_transfer=True)
    assert original.keys() == actual.keys()
    for key in original:
        np.testing.assert_array_equal(actual[key], original[key])
    np.testing.assert_array_equal(new_logits, old_logits)


def test_baseline_fast_transfer_preserves_features(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_screen

    def fake_branches(reader, x):
        base = x.mean((1, 2)) + 2
        return ({name: base[:, None].repeat(1, 160) for name in ('z_id','t_emb','f_emb','pa_local')},
                base[:, None].repeat(1, 6))
    monkeypatch.setattr(milkr_screen, 'branches', fake_branches)
    raw = np.random.default_rng(17).normal(size=(3, 2, 256)).astype(np.float32)
    old, old_logits = milkr_screen.extract(None, raw, 'cpu', batch=2)
    new, new_logits = milkr_screen.extract(None, raw, 'cpu', batch=2, fast_transfer=True)
    for name in old:
        np.testing.assert_array_equal(new[name], old[name])
    np.testing.assert_array_equal(new_logits, old_logits)


def test_restore_fast_transfer_matches_original():
    from experiments.pamf_phase2_practical.r42_suite_worker import _restore

    raw = np.random.default_rng(19).normal(size=(5, 2, 256)).astype(np.float32)
    engine = lambda value: (value * 2,)
    old = _restore(engine, raw, 'cpu', 2)
    new = _restore(engine, raw, 'cpu', 2, fast_transfer=True)
    np.testing.assert_array_equal(new, old)
