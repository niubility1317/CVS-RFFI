import pytest
import torch

from experiments.pamf_phase3_restoration import wide_tail_feature_trial as subject


def test_tail_loss_matches_quartile_mean_and_backpropagates_only_to_selected():
    error = torch.tensor([1., 4., 2., 3., 5., 8., 6., 7.], requires_grad=True)
    loss, selected = subject.tail_term(error)
    assert selected.tolist() == [0., 0., 0., 0., 0., 1., 0., 1.]
    torch.testing.assert_close(loss.mean(), torch.tensor(.005 * (8 + 7) / 2))
    loss.mean().backward()
    torch.testing.assert_close(error.grad, selected * (.005 / 2))


def test_partial_microbatch_uses_ceiling_and_preserves_normalization():
    error = torch.tensor([1., 2., 3., 4., 5.])
    loss, selected = subject.tail_term(error)
    assert selected.sum() == 2
    torch.testing.assert_close(loss.mean(), torch.tensor(.005 * (5 + 4) / 2))


@pytest.mark.parametrize('bad', [torch.tensor([]), torch.tensor([float('nan')]), torch.tensor([-1.]), torch.ones(2, 2)])
def test_rejects_invalid_source_errors(bad):
    with pytest.raises(ValueError):
        subject.tail_term(bad)


def test_fixed_arm_and_data_permissions():
    with pytest.raises(ValueError):
        subject.configuration('other')
    cfg = subject.configuration(subject.ARM)
    assert cfg['epochs'] == 30 and cfg['updates'] == 7410 and cfg['training_stages'] == 1
    assert cfg['TX_labels_read'] == cfg['V_reads'] == cfg['OE_reads'] == cfg['target_training_reads'] == 0
    assert cfg['source_feature_tail_fraction'] == .25 and cfg['source_feature_tail_weight'] == .005
    assert cfg['matched_control'] == subject.wide.RUN + '/' + subject.wide.ARM


def test_readout_adds_only_the_declared_tail_term(monkeypatch):
    error = torch.tensor([1., 4., 2., 3.], requires_grad=True)
    base = torch.tensor([.2, .4, .6, .8], requires_grad=True)
    aux = {'marker': True}
    metrics = {'feature_standardized_error': error}
    monkeypatch.setattr(subject.response, 'response_readout',
                        lambda *a, **kw: ('same_restored', aux, base, metrics))
    restored, actual_aux, loss, actual_metrics = subject.tail_readout()
    assert restored == 'same_restored' and actual_aux is aux
    torch.testing.assert_close((loss - base).mean(), .005 * error.max())
    assert actual_metrics['source_feature_tail_selected'].sum() == 1
    loss.mean().backward()
    torch.testing.assert_close(error.grad, torch.tensor([0., .005, 0., 0.]))
