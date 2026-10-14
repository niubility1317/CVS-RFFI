"""Synthetic native AdamW history filtering, without IQ/checkpoint access."""
import copy
import json
import math

import torch

from .state import make_filtered_optimizer, W_NAME, GAIN_NAMES, MODES
from .projection import project_history
from experiments.cvs_u_schedule50.counterfactual import simulate, _equal, _rng_snapshot


def check():
    from experiments.cvs_crossrx_sharpness20 import design as design, model as api
    torch.set_num_threads(2)
    config = design.config(next(row for row in design.rows() if row['arm'] == 'BASE_CE'
                                and row['model_seed'] == design.SEEDS[0]))
    model = api.build(config, 'cpu').eval()
    named = dict(model.named_parameters()); names = list(named); parameters = list(named.values())
    assert len(parameters) == 188
    feature = [index for index, name in enumerate(names) if name.startswith('id_backbone.encoder.') and name != W_NAME]
    assert len(feature) == 71 and sum(parameters[index].numel() for index in feature) == 195041
    assert all(names.index(name) in feature for name in GAIN_NAMES)
    w_index = names.index(W_NAME)
    outside = [index for index in range(188) if index not in feature and index != w_index]
    none_feature, zero_feature = feature[:2]
    none_other, zero_other = outside[:2]
    saved = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4).state_dict()
    assert saved['param_groups'][0]['foreach'] is None
    main = [None] * 188; rows = [[None] * 188 for _ in range(5)]
    coordinates = (-2., -1., 0., 1., 2.)
    for index in feature:
        parameter = parameters[index]
        alternate = ((torch.arange(parameter.numel()).reshape(parameter.shape) % 2) * 2 - 1).float()
        across_tensor = torch.full_like(parameter, 1. if index % 2 else -1.)
        gradient = torch.full_like(parameter, .03 + .01 * (index % 5))
        if index == zero_feature:
            gradient.zero_()
        if index != none_feature:
            main[index] = gradient
            for receiver, coordinate in enumerate(coordinates):
                rows[receiver][index] = (gradient + .02 * coordinate * alternate
                    + .015 * (coordinate * coordinate - 2.) * across_tensor)
        saved['state'][index] = dict(step=torch.tensor(float(10 + index % 5)),
            exp_avg=.03 + .01 * alternate + .004 * across_tensor,
            exp_avg_sq=torch.full_like(parameter, .001 + .0002 * (index % 3)))
    for index in (w_index, none_other, zero_other):
        parameter = parameters[index]
        saved['state'][index] = dict(step=torch.tensor(17.), exp_avg=torch.full_like(parameter, .08),
                                    exp_avg_sq=torch.full_like(parameter, .003))
    main[w_index] = torch.full_like(parameters[w_index], .15)
    main[zero_other] = torch.zeros_like(parameters[zero_other])
    for row in rows:
        row[w_index] = main[w_index].clone(); row[zero_other] = main[zero_other].clone()
    main = tuple(main); receiver = tuple(tuple(row) for row in rows)
    parameters[none_other].grad = torch.ones_like(parameters[none_other])
    original_grad = parameters[none_other].grad
    payload_before = copy.deepcopy(saved)
    parameter_before = tuple(parameter.detach().clone() for parameter in parameters)
    gradients_before = copy.deepcopy((main, receiver))
    rng = _rng_snapshot()
    opts, summaries, simulated, actual_states = {}, {}, {}, {}
    for mode in MODES:
        optimizer, summary = make_filtered_optimizer(parameters, names, saved, main, receiver, mode)
        opts[mode], summaries[mode] = optimizer, summary
        assert summary['F_tensor_count'] == 71 and summary['F_coordinate_count'] == 195041
        assert summary['rank'] >= 1 and summary['rank'] <= 4
        assert summary['three_gains_in_F'] and summary['W_optimizer_state_unchanged']
        assert summary['exp_avg_sq_unchanged'] and summary['step_unchanged']
        assert summary['current_gradients_unchanged'] and summary['live_parameters_gradients_payload_RNG_unchanged']
        assert summary['inactive_F_None_count'] == 1 and summary['clip_factor'] < 1.
        assert summary['prospective_v_native_FP32_operations'] and summary['prospective_step_uses_native_saved_scalar_add']
        assert optimizer.param_groups[0]['lr'] == 2e-4
        assert set(optimizer.state_dict()['state']) == set(saved['state'])
        for index, state in saved['state'].items():
            actual = optimizer.state_dict()['state'][index]
            assert _equal(state['step'], actual['step']) and _equal(state['exp_avg_sq'], actual['exp_avg_sq'])
            if index not in feature:
                assert _equal(state, actual)
        simulated[mode] = simulate(optimizer, main, main, names=names)
        assert simulated[mode]['summary']['total_clip_factor'] == summary['clip_factor']
        assert simulated[mode]['summary']['total_grad_norm'] == summary['main_gradient_norm']
        assert all(torch.equal(value, reference) for value, reference in zip(simulated[mode]['base_after'], simulated[mode]['total_after']))
        # A separate fixture-only private native step exposes after-v and step.
        private = copy.deepcopy(optimizer)
        private_parameters = private.param_groups[0]['params']
        for parameter, gradient in zip(private_parameters, main):
            parameter.grad = None if gradient is None else gradient.clone()
        norm = torch.nn.utils.clip_grad_norm_(private_parameters, 5., error_if_nonfinite=True)
        assert float((5. / (norm + 1e-6)).clamp(max=1.)) == summary['clip_factor']
        clipped = tuple(None if parameter.grad is None else parameter.grad.clone() for parameter in private_parameters)
        private.step(); actual_states[mode] = private.state_dict()
        assert all(torch.equal(parameter.detach(), value) for parameter, value in zip(private_parameters, simulated[mode]['total_after']))
        for index, state in saved['state'].items():
            after_state = actual_states[mode]['state'][index]
            if main[index] is None:
                assert torch.equal(after_state['step'], state['step'])
                assert torch.equal(after_state['exp_avg_sq'], state['exp_avg_sq'])
                assert torch.equal(private_parameters[index].detach(), parameter_before[index])
            else:
                expected_v = state['exp_avg_sq'].clone().mul_(.999).addcmul_(clipped[index], clipped[index], value=1. - .999)
                assert torch.equal(after_state['exp_avg_sq'], expected_v), 'Prospective FP32 v differs from actual native after-v'
                assert torch.equal(after_state['step'], state['step'].clone().add_(1))
        json.dumps(summary, allow_nan=False)
    assert all(torch.equal(simulated[mode]['total_after'][w_index], simulated['KEEP']['total_after'][w_index]) for mode in MODES)
    assert all(torch.equal(simulated[mode]['total_after'][none_feature], parameter_before[none_feature]) for mode in MODES)
    assert all(torch.equal(simulated[mode]['total_after'][none_other], parameter_before[none_other]) for mode in MODES)
    assert not torch.equal(simulated['KEEP']['total_after'][zero_other], parameter_before[zero_other])
    assert _equal(opts['KEEP'].state_dict()['state'], saved['state'])
    assert summaries['KEEP']['removed_norm'] == 0. and summaries['KEEP']['rounding_residual_norm'] == 0.
    assert summaries['M_ZERO_F']['removed_norm'] > 0. and summaries['RX_M_FILTER']['removed_norm'] > 0.
    for index in feature:
        assert torch.count_nonzero(opts['M_ZERO_F'].state_dict()['state'][index]['exp_avg']) == 0
    assert not summaries['M_ZERO_F']['inactive_F_None_exp_avg_unchanged']
    assert summaries['RX_M_FILTER']['inactive_F_None_exp_avg_unchanged']
    assert torch.equal(opts['RX_M_FILTER'].state_dict()['state'][none_feature]['exp_avg'], saved['state'][none_feature]['exp_avg'])

    # Reconstruct projection/mapping from the *actual native after-v* reference.
    keep_after = actual_states['KEEP']['state']; beta1, beta2 = (.9, .999)
    history, contrasts, scales = [], [[] for _ in range(5)], {}
    for index in feature:
        parameter = parameters[index]
        if main[index] is None:
            history.append(torch.zeros(parameter.numel(), dtype=torch.float64))
            for chunks in contrasts: chunks.append(torch.zeros(parameter.numel(), dtype=torch.float64))
            continue
        native_after = keep_after[index]; next_step = int(float(native_after['step']))
        bc1, bc2 = 1. - beta1 ** next_step, 1. - beta2 ** next_step
        denominator = native_after['exp_avg_sq'].sqrt().div_(math.sqrt(bc2)).add_(1e-8)
        scale = beta1 / (bc1 * denominator.double()); scales[index] = scale
        history.append((saved['state'][index]['exp_avg'].double() * scale).flatten())
        values = [row[index].double() if row[index] is not None else torch.zeros_like(parameter, dtype=torch.float64) for row in receiver]
        average = torch.stack(values).mean(0)
        multiplier = (1. - beta1) * summaries['KEEP']['clip_factor'] / (bc1 * denominator.double())
        for chunks, value in zip(contrasts, values): chunks.append(((value - average) * multiplier).flatten())
    history = torch.cat(history); contrasts = torch.stack([torch.cat(row) for row in contrasts])
    ideal, reference_summary = project_history(history, contrasts)
    assert reference_summary['rank'] == summaries['RX_M_FILTER']['rank']
    mapped_history, moment_error, offset = [], [], 0
    filtered_states = opts['RX_M_FILTER'].state_dict()['state']
    for index in feature:
        count = parameters[index].numel(); chunk = ideal[offset:offset + count].reshape(parameters[index].shape)
        if main[index] is None:
            mapped_history.append(torch.zeros(count, dtype=torch.float64))
        else:
            ideal_m = chunk / scales[index]
            assert torch.equal(filtered_states[index]['exp_avg'], ideal_m.float()), 'Filtered FP32 m is not the projected history mapping'
            mapped_history.append((filtered_states[index]['exp_avg'].double() * scales[index]).flatten())
            moment_error.append(filtered_states[index]['exp_avg'].double() - ideal_m)
        offset += count
    realized = torch.cat(mapped_history)
    assert summaries['RX_M_FILTER']['rounding_residual_norm'] == float((realized - ideal).norm())
    assert summaries['RX_M_FILTER']['max_actual_normalized_inner_product_residual'] < 1e-6
    assert _equal(saved, payload_before) and _equal((main, receiver), gradients_before)
    assert all(torch.equal(parameter.detach(), value) for parameter, value in zip(parameters, parameter_before))
    assert parameters[none_other].grad is original_grad and _equal(rng, _rng_snapshot())

    negatives = 0
    def reject(payload=saved, gradients=main, rx=receiver, mode='RX_M_FILTER', chosen_names=names, maximum=5.):
        nonlocal negatives
        try:
            make_filtered_optimizer(parameters, chosen_names, payload, gradients, rx, mode, max_grad_norm=maximum)
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Unsupported native history/state/gradient was accepted')
    for option in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'fused'):
        payload = copy.deepcopy(saved); payload['param_groups'][0][option] = True; reject(payload=payload)
    for index in (feature[0], w_index, zero_other):
        payload = copy.deepcopy(saved); del payload['state'][index]; reject(payload=payload)
    payload = copy.deepcopy(saved); payload['state'][feature[1]]['exp_avg_sq'].fill_(-1.); reject(payload=payload)
    payload = copy.deepcopy(saved); payload['state'][feature[1]]['step'].fill_(1.5); reject(payload=payload)
    payload = copy.deepcopy(saved); payload['state'][feature[1]]['exp_avg'].fill_(float('inf')); reject(payload=payload)
    payload = copy.deepcopy(saved); payload['param_groups'][0]['betas'] = (0., .999); reject(payload=payload)
    bad = list(main); bad[feature[1]] = torch.zeros(1); reject(gradients=tuple(bad))
    bad = list(main); bad[feature[1]] = torch.full_like(parameters[feature[1]], float('nan')); reject(gradients=tuple(bad))
    bad = list(main); bad[feature[1]] = main[feature[1]].double(); reject(gradients=tuple(bad))
    bad = list(main); bad[feature[1]] = main[feature[1]].to_sparse(); reject(gradients=tuple(bad))
    bad_rows = [list(row) for row in receiver]; bad_rows[0][none_feature] = torch.zeros_like(parameters[none_feature])
    reject(rx=tuple(tuple(row) for row in bad_rows))
    reject(rx=receiver[:4]); reject(rx=tuple(tuple(row[index] for index in feature) for row in receiver))
    bad_names = list(names); bad_names[-1] = bad_names[0]; reject(chosen_names=bad_names)
    reject(mode='RESET_ALL'); reject(maximum=4.)
    assert _equal(saved, payload_before) and _equal((main, receiver), gradients_before) and _equal(rng, _rng_snapshot())
    assert all(torch.equal(parameter.detach(), value) for parameter, value in zip(parameters, parameter_before))
    assert parameters[none_other].grad is original_grad
    return dict(status='PASS', actual_native_parameter_traversal=188, F_tensors=71, F_coordinates=195041,
        three_gains_included=True, current_main_RX_gradients_unchanged=True,
        v_step_W_state_preserved=True, native_prospective_v_matches_actual_after_v=True,
        native_foreach_None_CPU=True,
        actual_clipping_factor_matches_simulate=True, actual_W_step_identical_all_modes=True,
        projected_history_FP32_mapping_and_rounding_residual=True,
        None_skips_but_explicit_zero_advances=True, absent_unused_state_preserved=True,
        live_parameters_gradients_payload_RNG_unchanged=True, negative_checks=negatives,
        data_reads=0, checkpoint_reads=0, actual_optimizer_updates=0, synthetic_temporary_steps=9)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False))
