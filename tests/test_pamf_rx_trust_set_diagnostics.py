import numpy as np
import pytest
from experiments.pamf_u_history.rx_trust_set_diagnostics import set_statistics, collection_script


def test_true_class_in_set_can_increase_wrong_margin_matches_finite_difference():
    raw = np.log([[.98, .004, .004, .004, .004, .004]])
    H = np.log([[.2, .5, .075, .075, .075, .075]])
    r = set_statistics(raw, H, np.array([0]), role='V')
    assert r['set_inclusion_pct'] == 100
    assert r['wrong_conditional_target_mean'] == pytest.approx(5 / 7)
    assert r['wrong_margin_logit_velocity_mean'] == pytest.approx(9 / 70)
    assert r['hard_raw_wrong_margin_logit_velocity_mean'] == pytest.approx(-1.3)
    def loss(z):
        p = np.exp(z - z.max())
        return -np.log(p[:2].sum() / p.sum())
    z = H[0].copy()
    grad = []
    eps = 1e-6
    for j in range(6):
        delta = np.eye(6)[j] * eps
        grad.append((loss(z + delta) - loss(z - delta)) / (2 * eps))
    assert grad[0] - grad[1] == pytest.approx(r['wrong_margin_logit_velocity_mean'], abs=1e-9)


def test_no_qualified_wrong_H_keeps_conditional_diagnostic_undefined():
    logits = np.array([[6., 0., 0., 0., 0., 0.]])
    r = set_statistics(logits, logits, np.array([0]), role='V')
    assert r['selected_raw_correct_H_wrong'] == 0
    assert r['wrong_margin_logit_velocity_mean'] is None
    assert r['wrong_conditional_target_mean'] is None


def test_source_roles_and_finite_input_required():
    x = np.ones((2, 6))
    with pytest.raises(ValueError):
        set_statistics(x, x, np.array([0, 1]), role='U')
    with pytest.raises(ValueError):
        set_statistics(x, x * np.nan, np.array([0, 1]), role='V')
    with pytest.raises(ValueError):
        collection_script({'run_id': 'target-not-source'})


def test_compiled_readout_has_no_torch_or_model_execution():
    from experiments.pamf_u_history.rx_mean_source import RUN
    script = collection_script({'run_id': RUN, 'run_root': '/source_only'})
    compile(script, 'source_set_margin', 'exec')
    assert 'torch' not in script
    assert 'source_validation/source_V_predictions.npz' in script
    assert 'np.array_equal' in script
