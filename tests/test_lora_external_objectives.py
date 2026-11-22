import copy

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_lora_main import phase1
from experiments.practical_full_2x2 import objectives


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(2)
        self.head = nn.Linear(2, 10)

    def forward(self, x):
        features = self.norm(x).mean(dim=-1)
        return features, self.head(features)


def augment(x, *args):
    return x, dict(applied_count=len(x), scenario='test', probability=1.)


def run_step(model, x, u):
    config = dict(family='ibrff', variant='full', U_s_use=True,
                  lambda_sat_cls=.68, pl_threshold=.95, pl_weight=1.,
                  accepted_pl_only=True, row_id='test')
    return objectives.step(model, config, x, torch.arange(len(x)),
                           torch.zeros(len(x), dtype=torch.long),
                           torch.optim.SGD(model.parameters(), lr=.01),
                           augment, .5, 0, unlabeled=u)


def test_rejected_pl_does_not_affect_parameters_or_bn_buffers(monkeypatch):
    torch.manual_seed(1)
    x = torch.randn(4, 2, 16)
    targets = torch.zeros(4, dtype=torch.long)
    keep = torch.zeros(4, dtype=torch.bool)
    monkeypatch.setattr(objectives, 'pseudo_targets', lambda *args: (targets, keep, keep.float()))
    a = Classifier()
    b = copy.deepcopy(a)
    record = run_step(a, x, x)
    run_step(b, x, x + 100.)
    assert record['pl_seen'] == 4 and record['pl_accepted'] == 0 and record['pl_loss'] == 0
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)


def test_partial_acceptance_preserves_full_batch_loss_denominator(monkeypatch):
    torch.manual_seed(2)
    x = torch.randn(4, 2, 16)
    u = torch.randn_like(x)
    targets = torch.tensor([3, 2, 5, 1])
    keep = torch.tensor([False, True, False, True])
    monkeypatch.setattr(objectives, 'pseudo_targets', lambda *args: (targets, keep, keep.float()))
    model = Classifier()
    expected_logits = copy.deepcopy(model)(torch.cat([x, x, u[keep]]))[1][8:]
    expected = F.cross_entropy(expected_logits, targets[keep], reduction='sum') / len(u)
    record = run_step(model, x, u)
    assert record['pl_seen'] == 4 and record['pl_accepted'] == 2
    torch.testing.assert_close(torch.tensor(record['pl_loss']), expected)


def test_drift_separation_uses_the_unit_norm_bound():
    net = nn.Module()
    net.split_dim = 2
    net.encoder = nn.Sequential(nn.Flatten(), nn.Linear(4, 4, bias=False))
    net.tx_classifier = nn.Linear(2, 10)
    net.rx_classifier = nn.Linear(2, 2)
    net.domain_discriminator = nn.Linear(2, 2)
    with torch.no_grad():
        net.encoder[1].weight.copy_(100 * torch.eye(4))
    x = torch.tensor([[[1., 0.], [-1., 0.]], [[0., 1.], [0., -1.]]])
    config = dict(family='drift', variant='full', U_s_use=False, pl_weight=1.,
                  lambda_sat_cls=.68, drift_normalized_separation=True, log_native_terms=True,
                  tx_label_smoothing=.05)
    labels = torch.tensor([0, 1])
    logits = net.tx_classifier(net.encoder(x)[:, :2])
    expected = F.cross_entropy(logits, labels, label_smoothing=.05)
    record = objectives.step(net, config, x, torch.tensor([0, 1]), torch.tensor([0, 1]),
                             torch.optim.SGD(net.parameters(), lr=.01), augment, .5, 0)
    assert record['native_terms']['loss_mse'] == -4.
    assert record['native_terms']['z_tx_norm'] == 100.
    torch.testing.assert_close(torch.tensor(record['loss'] - record['native_penalty']), 1.68 * expected)


def test_source_row_rejects_the_target_handle_before_opening_source():
    config = dict(run_id=phase1.RUN, family='ibrff', model_seed=phase1.SEEDS[0],
                  target_practical='/not/an/input')
    try:
        phase1.check_training(config)
    except ValueError as error:
        assert 'source rows' in str(error)
    else:
        raise AssertionError('Training accepted a target handle')


def test_only_the_two_registered_r03_family_options_apply():
    from experiments.practical_full_2x2.source import riei_tail_lr

    assert phase1.registered_rows(phase1.OPTIMIZATION_RUN) == [('riei', phase1.SEEDS[0]),
                                                             ('drift', phase1.SEEDS[0])]
    base = dict(code_commit='commit', source_root='src', source_practical='views.npy',
                run_root='run', ibrff_model_root='ib', sdg_model_root='sdg')
    new = dict(base, run_id=phase1.OPTIMIZATION_RUN)
    for name, seed in phase1.registered_rows(phase1.OPTIMIZATION_RUN):
        old = phase1.configuration(base, name, seed)
        row = phase1.configuration(new, name, seed)
        assert (row['epochs'], row['lr'], row['batch_size']) == (old['epochs'], old['lr'], old['batch_size'])
        assert row['riei_cosine_tail'] is (name == 'riei')
        assert row['tx_label_smoothing'] == (.05 if name == 'drift' else 0.)
        assert not old['riei_cosine_tail'] and old['tx_label_smoothing'] == 0.
    torch.testing.assert_close(torch.tensor([riei_tail_lr(1e-4, p) for p in (0., .5, .75, 1.)]),
                               torch.tensor([1e-4, 1e-4, 5.5e-5, 1e-5]))
