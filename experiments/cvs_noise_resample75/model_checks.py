"""CPU tensor toys exercise the real six-L wrapper; no CVS or data loading."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from . import design as d, model
from experiments.cvs_full_channel_pair68 import checks as previous


class Toy(previous.ToyModel):
    def __init__(self, c):
        self._full_channel = previous.ToyShifted()
        self._sat_engine = previous.Engine('AS_CE')
        self._sat_steps, self._full_records, self._feature_learning_report = [], [], {}
        self.parameter = torch.tensor([.3, -.2, .7, -.5, .9, -.8], requires_grad=True)
        self.inputs = []
        model.attach(self, c)


def call(toy, raw, labels, metadata, function):
    first = toy.forward_identity_only(raw + .2)['tx_logits']
    toy.inputs.clear()
    ce = torch.nn.functional.cross_entropy(first, labels, label_smoothing=.01)
    kw = dict(labels=labels, optimizer_step=1, labeled_inputs=raw, extra=(None, metadata),
        domains=torch.zeros(128, dtype=torch.long), labeled_output=dict(tx_logits=first), identity_loss=ce)
    with patch.object(model.pair.original, 'hook', previous.toy_original):
        delta, info = function(toy, **kw)
    return ce + delta, info, kw


def run():
    negative, secondary_calls, physical_payloads = 0, [], []
    def bad(fn):
        nonlocal negative
        try:
            fn()
        except (ValueError, TypeError, KeyError, AttributeError):
            negative += 1
        else:
            raise AssertionError('Invalid second-noise hook context accepted')

    configs = {r['arm']: d.config(r) for r in d.rows()}
    assert set(configs) == {'C01', 'N01'}
    generator = torch.Generator().manual_seed(7501)
    raw = torch.randn(128, 2, 256, generator=generator)
    labels = torch.arange(128) % 6
    meta = dict(eq_i=torch.ones(128, dtype=torch.long), rx_i=torch.ones(128, dtype=torch.long),
        day_i=torch.arange(128) % 3 + 1, sig_i=torch.arange(128) + 10, base_index=torch.arange(128))
    mask = (torch.arange(128) + 1).remainder(2) != 0
    old_view, old_terms = model.pair.independent_view, model.pair.base.terms

    def secondary(primary, raw64, metadata, first, hardware):
        secondary_calls.append((primary, raw64.detach().clone(), deepcopy(first), list(hardware)))
        # Deterministic toy-only change: no physical engine, new RNG or CVS forward.
        x = raw64 + .07
        return x, deepcopy(first)

    # Same-IQ control must match the original SAT18 AS_CE hook and gradients.
    control, reference = Toy(configs['C01']), Toy(configs['C01'])
    torch.manual_seed(7511)
    with patch.object(model, 'second_noise_view', secondary):
        actual, _, _ = call(control, raw, labels, meta, model.hook)
    actual.backward()
    actual_rng = torch.get_rng_state().clone()
    torch.manual_seed(7511)
    expected, _, _ = call(reference, raw, labels, meta, model.pair.original.satellite_hook)
    expected.backward()
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert torch.allclose(control.parameter.grad, reference.parameter.grad, atol=1e-7, rtol=1e-7)
    assert torch.equal(torch.get_rng_state(), actual_rng) and not secondary_calls
    assert len(control._full_channel.obj.processed) == 64
    assert control._noise_generated == 64 and control._noise_secondary_generated == 0
    assert control._pair_generated == 64 and control._noise_latent_draws == 64 and control._noise_slots == 128
    assert torch.equal(control.inputs[1], control.inputs[2])
    c_record = control._noise_records[0]
    assert c_record['As_same_IQ'] and c_record['generated_As1'] == 0 and c_record['As_generated_packets'] == 64
    assert control._full_records[-1]['same_IQ_each_pair']

    candidate = Toy(configs['N01'])
    model.physics_sink(candidate, physical_payloads.append)
    torch.manual_seed(7511)
    np_before = np.random.get_state()
    with patch.object(model, 'second_noise_view', secondary):
        loss, _, kw = call(candidate, raw, labels, meta, model.hook)
    loss.backward()
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1]) and np_before[2:] == np_after[2:]
    # The wrapper adds no Torch RNG use; all original dropout draws are retained.
    assert torch.equal(torch.get_rng_state(), actual_rng)
    assert candidate.parameter.grad is not None and bool(torch.isfinite(candidate.parameter.grad).all())
    assert len(secondary_calls) == 1 and len(candidate.inputs) == 5
    assert len(candidate._full_channel.obj.processed) == 64
    assert torch.equal(secondary_calls[0][1], raw[mask])
    assert torch.equal(candidate.inputs[0], control.inputs[0])
    assert torch.equal(candidate.inputs[1][~mask], candidate.inputs[2][~mask])
    assert not torch.equal(candidate.inputs[1][mask], candidate.inputs[2][mask])
    assert torch.equal(candidate.inputs[3], raw) and torch.equal(candidate.inputs[4], raw)
    assert candidate._noise_generated == 128 and candidate._noise_secondary_generated == 64
    assert candidate._pair_generated == 64 and candidate._noise_latent_draws == 64 and candidate._noise_slots == 128
    assert sum(candidate._noise_coverage.values()) == 128 and sum(candidate._pair_coverage.values()) == 64
    record = candidate._noise_records[0]
    assert not record['As_same_IQ'] and record['generated_As1'] == 64 and record['As_generated_packets'] == 128
    assert record['same_H_and_receiver_state'] and not record['same_AGC'] and record['per_view_native_AGC_recomputed']
    assert record['cross_channel_KL'] is False and record['two_separate_CE_mean']
    assert physical_payloads[0]['actual_residual_snapshots'] == 128 and physical_payloads[0]['physical_latent_draws'] == 64
    model.physics_sink(candidate)
    assert candidate not in model._PHYSICS_SINKS

    # Existing instance overrides and all outer functions survive failures.
    def failure(*args, **kwargs):
        raise ValueError('Synthetic conditional-noise failure')
    for toy in (control, candidate):
        assert not toy._sat_engine.active
        assert 'forward_identity_only' not in vars(toy) and 'apply' not in vars(toy._full_channel)
    overridden = Toy(configs['N01'])
    override = overridden._full_channel.apply
    overridden._full_channel.apply = override
    forward_override = overridden.forward_identity_only
    overridden.forward_identity_only = forward_override
    with patch.object(model, 'second_noise_view', failure):
        bad(lambda: call(overridden, raw, labels, meta, model.hook))
    assert overridden._full_channel.apply is override and overridden.forward_identity_only is forward_override
    assert not overridden._sat_engine.active
    assert model.pair.independent_view is old_view and model.pair.base.terms is old_terms
    changed = Toy(configs['N01'])
    changed._noise_arm = 'FULL_INDEP'
    bad(lambda: call(changed, raw, labels, meta, model.hook))
    def invalid(primary, raw64, metadata, first, hardware):
        records = deepcopy(first)
        records[0]['state_end'] = 'OTHER'
        return raw64.clone(), records
    with patch.object(model, 'second_noise_view', invalid):
        bad(lambda: call(Toy(configs['N01']), raw, labels, meta, model.hook))
    assert model.pair.independent_view is old_view and model.pair.base.terms is old_terms
    assert not secondary_calls[0][1].requires_grad
    return dict(status='PASS', focused_negatives=negative,
        actual_six_L_hook=True, C_same_IQ_original_SAT18_loss_and_gradient_parity=True,
        N_only_existing_second_As64_changed=True, ordinary_raw_IQ_preserved=True,
        fixed_native_mask=True, primary_apply_once=True, C_secondary_native_calls=0,
        N_secondary_helper_calls=1, C_actual_snapshots=64, N_actual_snapshots=128,
        logical_slots_distinct_from_generations=True, no_wrapper_global_RNG_draws=True,
        CE_live_gradient=True, finally_restores_instance_and_global_overrides=True,
        physics_sink_outside_model=True, CVS_NN=0, real_IQ=0, checkpoint_reads=0,
        optimizer_updates=0, target_access=False)
