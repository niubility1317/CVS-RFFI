"""Actual-constructor synthetic held-L evaluation; no data or checkpoint."""
import copy
import json

import torch
from torch.nn import functional as F

from .evaluate import evaluate, HEAD_PATH, SEQUENCE_PATH, ENCODER, EPS
from experiments.cvs_identity_difference20.bank import SOURCE_RX, SOURCE_DAYS
from experiments.cvs_sigtime_heldcredit.held_credit import CACHE_ATTRIBUTES
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


def _bank():
    cells = [(y, receiver, day, side) for side in (0, 1) for y in range(6)
             for receiver in SOURCE_RX for day in SOURCE_DAYS]
    generator = torch.Generator().manual_seed(151720)
    return dict(x=torch.randn(180, 2, 256, generator=generator),
        y=torch.tensor([cell[0] for cell in cells]), rx=torch.tensor([cell[1] for cell in cells]),
        day=torch.tensor([cell[2] for cell in cells]), side=torch.tensor([cell[3] for cell in cells]),
        ids=[(1, receiver, day, 68 + side, 100000 + index)
             for index, (_, receiver, day, side) in enumerate(cells)])


def check():
    from experiments.cvs_crossrx_sharpness20 import design as design, model as api
    torch.set_num_threads(2)
    config = design.config(next(row for row in design.rows() if row['arm'] == 'BASE_CE'
                                and row['model_seed'] == design.SEEDS[0]))
    model = api.build(config, 'cpu').eval().requires_grad_(True)
    head, encoder, pool = (model.get_submodule(path) for path in (HEAD_PATH, ENCODER, SEQUENCE_PATH))
    # Fixture-only nonzero gains expose all three genuine deployed feature paths.
    with torch.no_grad():
        for gain in (head.gain, encoder.response_gain, pool.gain_raw):
            gain.fill_(.1)
    bank = _bank()
    with torch.no_grad():
        direct = model.forward_identity_only(bank['x'])
        expected_z = direct['id_feat_joint'].detach().clone()
        expected_native = direct['tx_logits'].detach().clone()
    named = dict(model.named_parameters()); names = list(named)
    assert len(names) == 188
    pre = tuple(named[name].detach().clone() for name in names)
    weight_index = next(index for index, name in enumerate(names) if named[name] is head.weight)
    gain_indices = [next(index for index, name in enumerate(names) if named[name] is gain)
                    for gain in (head.gain, encoder.response_gain, pool.gain_raw)]
    head.weight.grad = torch.ones_like(head.weight)
    original_grad = head.weight.grad
    existing = pool.output_norm.register_forward_hook(lambda module, inputs, output: None)
    before = copy.deepcopy(model.state_dict()); input_before = copy.deepcopy(bank)
    rng = _rng_snapshot()
    flags = [(module, module.training) for module in model.modules()]
    caches = [(module, key, getattr(module, key)) for module in model.modules()
              for key in CACHE_ATTRIBUTES if hasattr(module, key)]
    hooks = [(module, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in model.modules()]

    z, baseline = evaluate(model, names, pre, bank)
    assert torch.equal(z, expected_z), 'Materialized actual z differs from direct native model path'
    assert baseline['ACTUAL'] == baseline['PRE_W'], 'Same W used different readout precision/formula'
    expected_logits = 30. * F.linear(F.normalize(z.double(), dim=1, eps=EPS),
                                    F.normalize(head.weight.detach().double(), dim=1, eps=EPS))
    losses = F.cross_entropy(expected_logits, bank['y'], reduction='none')
    assert baseline['ACTUAL']['mean_CE'] == float(losses.mean())
    assert all(baseline['ACTUAL']['by_RX'][str(receiver)] == float(losses[bank['rx'] == receiver].mean())
               for receiver in SOURCE_RX)
    native_error = float((expected_logits - expected_native.double()).abs().max())
    assert baseline['native_default_logits_max_abs_FP32_rounding_error'] == native_error
    assert baseline['native_default_logits_used_for_path'] and baseline['native_default_logits_match_FP32_tolerance']
    assert baseline['W_change_norm'] == 0. and all(value == 0. for value in baseline['gain_delta_norms'].values())
    assert baseline['same_features_both_readouts'] and baseline['after_fusion_gains_retained']
    assert baseline['IQ_forward_count'] == 1 and baseline['network_IQ_forwards'] == 180
    assert baseline['new_head_fit_updates'] == 0 and baseline['actual_optimizer_updates'] == 0
    assert baseline['geometry']['mean_feature_norm'] > 0
    json.dumps(baseline, allow_nan=False)

    # Warm original native caches deliberately: W substitution must still be fresh.
    weight_only = list(pre)
    weight_only[weight_index] = weight_only[weight_index].roll(1, dims=0)
    w_z, w_summary = evaluate(model, names, tuple(weight_only), bank)
    assert torch.equal(w_z, z), 'Changing only default W changed the deployment feature'
    assert w_summary['PRE_W'] == baseline['PRE_W'], 'PRE_W accidentally consumed afterW'
    assert w_summary['ACTUAL']['mean_CE'] != baseline['ACTUAL']['mean_CE']
    assert w_summary['ACTUAL']['class_logit_margin_mean'] != baseline['ACTUAL']['class_logit_margin_mean']
    assert w_summary['W_change_norm'] > 0. and all(value == 0. for value in w_summary['gain_delta_norms'].values())

    # Each gain separately remains in the actual feature; PRE_W only fixes W.
    gain_checks = []
    for gain_index in gain_indices:
        gain_only = list(pre); gain_only[gain_index] = gain_only[gain_index] + .2
        gain_z, gain_summary = evaluate(model, names, tuple(gain_only), bank)
        gain_name = names[gain_index]
        assert not torch.equal(gain_z, z), 'Actual fusion gain was incorrectly restored to pre-step'
        assert gain_summary['PRE_W']['mean_CE'] != baseline['PRE_W']['mean_CE']
        assert gain_summary['PRE_W']['class_logit_margin_mean'] != baseline['PRE_W']['class_logit_margin_mean']
        assert gain_summary['W_change_norm'] == 0.
        assert gain_summary['gain_delta_norms'][gain_name] > 0.
        assert sum(value > 0. for value in gain_summary['gain_delta_norms'].values()) == 1
        gain_checks.append(gain_name)
    feature_after = list(pre)
    feature_index = names.index(SEQUENCE_PATH + '.output_norm.bias')
    feature_after[feature_index] = feature_after[feature_index].clone(); feature_after[feature_index][0] += .25
    feature_z, feature_summary = evaluate(model, names, tuple(feature_after), bank)
    assert not torch.equal(feature_z, z) and feature_summary['PRE_W']['mean_CE'] != baseline['PRE_W']['mean_CE']
    assert all(value == 0. for value in feature_summary['gain_delta_norms'].values())
    assert _equal(before, model.state_dict()) and _equal(input_before, bank) and _equal(rng, _rng_snapshot())
    assert head.weight.grad is original_grad and torch.equal(original_grad, torch.ones_like(original_grad))
    assert all(module.training == flag for module, flag in flags)
    assert all(getattr(module, key) is value for module, key, value in caches)
    assert all(dict(module._forward_hooks) == forward and dict(module._forward_pre_hooks) == previous
               for module, forward, previous in hooks)

    # eval flags are owned by the private copy, including when caller is training.
    model.train()
    train_z, _ = evaluate(model, names, pre, bank)
    assert torch.equal(train_z, z) and all(module.training for module in model.modules())
    model.eval()
    negative_checks = 0
    def reject(selected_names=names, state=pre, data=bank):
        nonlocal negative_checks
        try:
            evaluate(model, selected_names, state, data)
        except ValueError:
            negative_checks += 1
        else:
            raise AssertionError('Invalid after-state or non-source held bank was accepted')
    reject(names[:-1], pre[:-1])
    duplicate = list(names); duplicate[-1] = duplicate[0]; reject(duplicate)
    outside = list(names); outside[-1] = 'outside'; reject(outside)
    wrong_state = list(pre); wrong_state[0] = pre[0].reshape(-1)[:1]; reject(state=tuple(wrong_state))
    wrong_state = list(pre); wrong_state[0] = pre[0].double(); reject(state=tuple(wrong_state))
    wrong_state = list(pre); wrong_state[0] = torch.full_like(pre[0], float('nan')); reject(state=tuple(wrong_state))
    for key, value in (('x', bank['x'][:179]), ('x', torch.full_like(bank['x'], float('inf'))),
                       ('y', bank['y'].float()), ('y', torch.full_like(bank['y'], -1)),
                       ('rx', torch.full_like(bank['rx'], 2)), ('day', torch.zeros_like(bank['day'])),
                       ('side', torch.zeros_like(bank['side']))):
        wrong_bank = dict(bank); wrong_bank[key] = value; reject(data=wrong_bank)
    wrong_bank = copy.deepcopy(bank); wrong_bank['ids'][1] = wrong_bank['ids'][0]; reject(data=wrong_bank)
    wrong_bank = copy.deepcopy(bank)
    original_id = wrong_bank['ids'][0]; wrong_bank['ids'][0] = (original_id[0], 3, *original_id[2:])
    reject(data=wrong_bank)
    wrong_bank = dict(bank, role='U_s'); reject(data=wrong_bank)
    wrong_bank = copy.deepcopy(bank); wrong_bank['y'][0] = 1; reject(data=wrong_bank)
    assert _equal(before, model.state_dict()) and _equal(rng, _rng_snapshot())
    assert head.weight.grad is original_grad
    assert all(getattr(module, key) is value for module, key, value in caches)
    assert all(dict(module._forward_hooks) == forward and dict(module._forward_pre_hooks) == previous
               for module, forward, previous in hooks)
    existing.remove(); head.weight.grad = None
    return dict(status='PASS', actual_SIG_TIME_CENTER_constructor=True,
        actual_joint_matches_native_path_exact=True, shared_FP64_default_head_formula=True,
        W_only_actual_readout_changes_joint_PREW_unchanged=True,
        each_gain_only_joint_PREW_changes=gain_checks, genuine_feature_parameter_change_visible=True,
        original_state_RNG_gradients_flags_warm_caches_existing_hooks_unchanged=True,
        held_L180_balance=True, negative_checks=negative_checks,
        actual_data_reads=0, checkpoint_reads=0, actual_optimizer_updates=0, head_fit_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False))
