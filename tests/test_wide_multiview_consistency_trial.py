import pytest
import torch

from experiments.pamf_phase3_restoration import wide_multiview_consistency_trial as subject


def test_second_draw_always_differs_from_original_fixed_30_epoch_schedule():
    schedule = list(range(1, 21)) + list(range(1, 11))
    assert subject.second_draw_number(0) == 2
    assert [subject.second_draw_number(i) for i in range(1, 31)] == [d % 20 + 1 for d in schedule]
    assert all(subject.second_draw_number(i) != d for i, d in enumerate(schedule, 1))
    with pytest.raises(ValueError):
        subject.second_draw_number(31)


def test_feature_consistency_is_symmetric_and_differentiable():
    first = torch.ones(2, 160, requires_grad=True)
    second = torch.zeros(2, 160, requires_grad=True)
    scale = torch.full((160,), 2., requires_grad=True)
    value = subject.consistency(first, second, scale)
    torch.testing.assert_close(value, torch.full((2,), .5))
    torch.testing.assert_close(value, subject.consistency(second, first, scale))
    value.mean().backward()
    assert first.grad.abs().sum() > 0 and second.grad.abs().sum() > 0
    assert scale.grad is None


@pytest.mark.parametrize('bad', [torch.zeros(160), torch.full((160,), float('nan'))])
def test_rejects_invalid_source_feature_scale(bad):
    features = torch.zeros(2, 160)
    with pytest.raises(ValueError):
        subject.consistency(features, features, bad)


def test_second_view_changes_loss_and_trains_detail_only_from_source(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detail = torch.nn.Module()
            self.detail.refinement = torch.nn.Module()
            self.detail.refinement.output = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.detail.refinement.output.weight.fill_(.5)
        def source_readout(self, signal, *, prior=None):
            return signal * self.detail.refinement.output.weight.reshape(()), {'relative_solve_residual': torch.zeros(len(signal))}
    class Reader:
        def features(self, x):
            return x.flatten(1)[:, :160]
    model = Model(); reader = Reader(); received = torch.ones(2, 2, 256)
    def base(model, reader, received, *args, **kwargs):
        restored, aux = model.source_readout(received)
        return restored, aux, received.new_zeros(len(received)), {}
    monkeypatch.setattr(subject.response, 'response_readout', base)
    monkeypatch.setattr(subject, 'choose_prior', lambda *args: None)
    seen = []
    def second(epoch, indices):
        seen.append((epoch, indices.clone()))
        return torch.zeros_like(received)
    indices = torch.tensor([1, 9])
    restored, aux, loss, metrics = subject.paired_readout(
        model, reader, received, received, None, torch.ones(160), None,
        source_indices=indices, epoch=0, priors=None, assignment=None,
        source_trace={'received': received}, second_source=second)
    torch.testing.assert_close(metrics['source_view_feature_distance'], torch.full((2,), .25))
    torch.testing.assert_close(loss, torch.full((2,), .0025))
    assert seen[0][0] == 0 and torch.equal(seen[0][1], indices)
    assert model.paired_view_smoke['actual_source_view_consistency_gradient']
    loss.mean().backward()
    assert model.detail.refinement.output.weight.grad.abs().sum() > 0


def test_fixed_arm_and_source_permissions():
    with pytest.raises(ValueError):
        subject.configuration('other')
    cfg = subject.configuration(subject.ARM)
    assert cfg['epochs'] == 30 and cfg['updates'] == 7410 and cfg['training_stages'] == 1
    assert cfg['TX_labels_read'] == cfg['V_reads'] == cfg['OE_reads'] == cfg['target_training_reads'] == 0
    assert cfg['matched_control'] == subject.wide.RUN + '/' + subject.wide.ARM


@pytest.mark.parametrize('field', ['TX_labels_read', 'target_reads'])
def test_second_draw_rejects_forbidden_source_cache(monkeypatch, tmp_path, field):
    meta = {'count': 63000, 'TX_labels_read': 0, 'target_reads': 0}
    meta[field] = 1
    monkeypatch.setattr(subject, 'read', lambda path: meta)
    provider = subject.SecondSourceDraw(tmp_path, 'cpu')
    with pytest.raises(ValueError, match='source draw'):
        provider(1, torch.tensor([0]))
