"""Focused native callback wiring checks; objective mathematics is not repeated."""
import copy
import json
from unittest.mock import patch

import torch
from torch import nn

from . import observe


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Module(); self.id_backbone.encoder = nn.Module()
        self.id_backbone.encoder.core = nn.Module()
        self.id_backbone.encoder.core.id_backbone = nn.Module()
        primary = self.id_backbone.encoder.core.id_backbone
        for name in ('sinc', 'time_fuse', 't1', 't2'):
            module = nn.Module(); module.weight = nn.Parameter(torch.tensor(.2))
            setattr(primary, name, module)
        self.id_backbone.encoder.sequence_pool = nn.Module()
        self.id_backbone.encoder.sequence_pool.weight = nn.Parameter(torch.tensor(.3))
        self.id_backbone.encoder.sequence_pool.gain_raw = nn.Parameter(torch.tensor(.1))
        self.default_head = nn.Module()
        self.default_head.weight = nn.Parameter(torch.eye(6, 160))


class Native:
    def __init__(self): self.meta_calls = 0
    def _meta_from_extra(self, extra):
        self.meta_calls += 1
        return extra[1]


def _bank():
    rx = torch.tensor([1, 1, 3, 3, 4, 4, 6, 6, 8, 8])
    ids = [(0, int(r), 1, 100+i, 100+i) for i, r in enumerate(rx)]
    return dict(x=torch.zeros(10, 2, 256), y=torch.arange(10).remainder(6),
                rx=rx, physical_ids=ids, role='L_s')


def _current(model):
    rx = torch.tensor([1, 1, 3, 3, 4, 4, 6, 6, 8, 8])
    base = torch.tensor([100] + list(range(201, 210)))
    meta = dict(eq_i=torch.zeros(10, dtype=torch.long), rx_i=rx,
                day_i=torch.ones(10, dtype=torch.long), sig_i=base.clone(), base_index=base)
    # Two output views expose the same original graph; callback must use only
    # the clean-count first slice and must not assume labeled_inputs is that view.
    logits = model.default_head.weight[:, :6].T.repeat(4, 1)[:20]
    return dict(model=model, labeled_output={'tx_logits': logits},
                labels=torch.arange(10).remainder(6), extra=(None, meta), domains=None,
                optimizer_step=3, identity_loss=logits.sum(), labeled_count=10,
                labeled_inputs=torch.full((10, 2, 256), 999.))


def _raises(callback):
    try: callback()
    except ValueError: return
    raise AssertionError('Expected source/callback boundary rejection')


def check():
    native = Native(); model = Fixture().train(); bank = _bank()
    configuration = dict(mode='CROSS_GAP', warmup_steps=2, frequency=1)
    callback = observe.Observer(native, configuration, bank, default_weight_name='default_head.weight')
    # Observer construction leaves student untouched before native's teacher copy.
    teacher = copy.deepcopy(model)
    assert callback.model is None and not hasattr(teacher, '_source_labeled_feature_hook')
    before_hooks = {module: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
                    for module in model.modules()}
    before_native = dict(native.__dict__)
    kwargs = _current(model); calls = []
    def fake_objective(m, logits, y, rx, hx, hy, hr, ids, hids, names, wname, mode, *, role):
        assert m is model and logits.shape == (10, 6) and logits.requires_grad
        assert torch.equal(logits, kwargs['labeled_output']['tx_logits'][:10])
        assert role == 'L_s' and mode == 'CROSS_GAP' and wname == 'default_head.weight'
        assert set(hr.tolist()) == {1, 3, 4, 6, 8}
        assert len(hx) == 9 and 100 not in {i[-1] for i in hids}
        assert not ({i[-1] for i in ids} & {i[-1] for i in hids})
        assert torch.count_nonzero(hx) == 0  # Never use misleading labeled_inputs.
        calls.append(dict(ids=ids, held=hids))
        loss = dict(m.named_parameters())[names[0]].square()*.1
        by_rx = {str(r):dict(clean_CE=1., pert_CE=1.01, gap=.01, positive_gap=.01)
                 for r in (1, 3, 4, 6, 8)}
        return dict(loss=loss, summary=dict(loss=float(loss.detach()), by_RX=by_rx,
                    extra_IQ_forwards=45, functional_forwards=6))
    with patch.object(observe, 'objective', fake_objective):
        inactive = dict(kwargs, optimizer_step=1)
        zero, metrics = callback(**inactive)
        assert zero.item() == 0 and metrics['crossRX_sharpness/active'] == 0 and not calls
        loss, metrics = callback(**kwargs)
    assert len(calls) == 1 and metrics['crossRX_sharpness/loss'] == float(loss.detach())
    assert callback.last_record['excluded_bank_count'] == 1
    assert callback.last_record['held_RX_counts'] == {'1':1, '3':2, '4':2, '6':2, '8':2}
    assert callback.summary()['extra_IQ_forwards'] == 45
    assert callback.summary()['calls'] == 2 and callback.summary()['eligible_calls'] == 1
    assert all(p.grad is None for p in model.parameters())
    grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    named = list(model.named_parameters())
    assert all(g is None for (name, _), g in zip(named, grads)
               if name == 'default_head.weight' or name.endswith('gain_raw'))
    assert all(before_hooks[module] == (dict(module._forward_hooks), dict(module._forward_pre_hooks))
               for module in model.modules())
    assert native.meta_calls == 1 and set(native.__dict__) == set(before_native)
    json.dumps(callback.summary(), allow_nan=False)
    frequency = observe.Observer(native, dict(configuration, frequency=2), bank,
                                 default_weight_name='default_head.weight')
    with patch.object(observe, 'objective', fake_objective):
        frequency(**kwargs)  # step3 is past warmup but not divisible by2.
        assert frequency.last_record['status'] == 'inactive'
        frequency(**dict(kwargs, optimizer_step=4))
    assert frequency.summary()['scheduled_calls'] == 1 and frequency.summary()['eligible_calls'] == 1

    def skipped(bank_value, call_kwargs, reason):
        obs = observe.Observer(native, configuration, bank_value, default_weight_name='default_head.weight')
        with patch.object(observe, 'objective', side_effect=AssertionError('Skipped call ran objective')):
            loss, metrics = obs(**call_kwargs)
        assert loss.item() == 0 and metrics['crossRX_sharpness/eligible'] == 0
        gradients = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
        assert all(g is None for (name, _), g in zip(named, gradients)
                   if name == 'default_head.weight' or name.endswith('gain_raw'))
        assert obs.last_record['status'] == reason and obs.summary()['extra_IQ_forwards'] == 0
        return obs
    missing = dict(kwargs); missing_meta = dict(kwargs['extra'][1])
    missing_meta['rx_i'] = missing_meta['rx_i'].clone(); missing_meta['rx_i'][8:] = 6
    missing['extra'] = (None, missing_meta)
    skipped(bank, missing, 'skip_missing_current_RX')
    reduced = {k:(v[:1] if torch.is_tensor(v) or isinstance(v, list) else v) for k,v in bank.items()}
    skipped(reduced, kwargs, 'skip_empty_bank')
    no_rx1 = {k:(v[1:] if torch.is_tensor(v) or isinstance(v, list) else v) for k,v in bank.items()}
    no_rx1['physical_ids'] = no_rx1['physical_ids'][1:]
    no_rx1['x'] = no_rx1['x'][1:]; no_rx1['y'] = no_rx1['y'][1:]; no_rx1['rx'] = no_rx1['rx'][1:]
    skipped(no_rx1, kwargs, 'skip_missing_held_RX')
    empty = dict(x=bank['x'][:0], y=bank['y'][:0], rx=bank['rx'][:0], physical_ids=[], role='L_s')
    skipped(empty, kwargs, 'skip_empty_bank')
    _raises(lambda: observe.Observer(native, dict(configuration, mode='BASE_CE'), bank))
    _raises(lambda: observe.Observer(native, dict(configuration, frequency=0), bank))
    _raises(lambda: observe.Observer(native, configuration, dict(bank, role='U_s')))
    bad = dict(kwargs); bad['labels'] = kwargs['labels'].clone(); bad['labels'][0] = -1
    _raises(lambda: observe.Observer(native, configuration, bank, default_weight_name='default_head.weight')(**bad))
    _raises(lambda: observe.Observer(native, configuration, bank, default_weight_name='default_head.weight')(**kwargs, role='U_s'))
    incomplete = dict(kwargs); incomplete['extra'] = (None, {'rx_i':bank['rx']})
    _raises(lambda: observe.Observer(native, configuration, bank, default_weight_name='default_head.weight')(**incomplete))
    return dict(status='VERIFIED', torch=torch.__version__, device='cpu',
                bank_current_base_indices_excluded=True, missing_RX_empty_whole_call_skipped=True,
                warmup_frequency_configured=True, main_logits_first_slice_used=True,
                labeled_inputs_not_assumed_same_view=True, one_objective_one_returned_loss=True,
                no_real_backward_optimizer_EMA_or_permanent_forward_hook=True,
                teacher_deepcopy_before_first_bind=True, extra_IQ_forwards=45,
                source_IQ_reads=0, boundary='Mock callback wiring only; unchanged objective not retested')


if __name__ == '__main__': print(json.dumps(check(), allow_nan=False, indent=2))
