"""Filter only feature-history AdamW momentum; preserve the current CE step."""
from collections.abc import Mapping, Sequence
import copy
import math

import torch
from torch import nn

from .projection import project_history
from experiments.cvs_evidence_recomposition20.capture import HEAD_PATH, SEQUENCE_PATH
from experiments.cvs_restart_state20.optimizer import make_optimizer
from experiments.cvs_u_schedule50.counterfactual import _equal, _preserve_rng, _rng_snapshot


MODES = ('KEEP', 'M_ZERO_F', 'RX_M_FILTER')
W_NAME = HEAD_PATH + '.weight'
GAIN_NAMES = (HEAD_PATH + '.gain', 'id_backbone.encoder.response_gain', SEQUENCE_PATH + '.gain_raw')
PARAMETER_COUNT = 188


def _gradients(values, parameters, name):
    if not isinstance(values, tuple) or len(values) != len(parameters):
        raise ValueError(name + ' must be the complete tuple188, preserving None semantics')
    for value, parameter in zip(values, parameters):
        if value is not None and (not isinstance(value, torch.Tensor) or value.requires_grad
                or value.layout != torch.strided or value.shape != parameter.shape
                or value.dtype != torch.float32 or value.device != parameter.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError(name + ' requires detached finite dense same-shape FP32 gradients or None')
    return values


def _norm(values):
    result = math.hypot(*(float(value.double().norm()) for value in values))
    if not math.isfinite(result):
        raise ValueError('Nonfinite momentum-analysis norm')
    return result


def _clip(parameters, gradients, maximum):
    # Use native clip_grad_norm_ on private parameter/gradient copies so its
    # FP32 norm, epsilon and multiplication match the later simulate() branch.
    private = [nn.Parameter(parameter.detach().clone(), requires_grad=False) for parameter in parameters]
    for parameter, gradient in zip(private, gradients):
        parameter.grad = None if gradient is None else gradient.clone()
    norm = torch.nn.utils.clip_grad_norm_(private, maximum, error_if_nonfinite=True)
    factor = float((maximum / (norm + 1e-6)).clamp(max=1.).item())
    clipped = tuple(None if parameter.grad is None else parameter.grad.detach().clone() for parameter in private)
    return clipped, float(norm), factor


def make_filtered_optimizer(parameters, names, saved_state, main_grads, rx_grads, mode,
                            restart_lr=2e-4, max_grad_norm=5.0):
    """Return a private-state AdamW bound to the original188 parameter objects.

    This function takes no step and changes neither live .grad nor parameters.
    The caller must use simulate(), which copies the optimizer before stepping.
    rx_grads is tuple5 x tuple188; each row uses the same native traversal.
    Only F exp_avg is changed. F includes all three gains and excludes actual W.
    """
    if mode not in MODES or type(max_grad_norm) not in (int, float) or max_grad_norm != 5.:
        raise ValueError('Fixed KEEP/M_ZERO_F/RX_M_FILTER and native clip5 required')
    if (not isinstance(parameters, (tuple, list)) or len(parameters) != PARAMETER_COUNT
            or not isinstance(names, Sequence) or isinstance(names, (str, bytes))
            or len(names) != PARAMETER_COUNT or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != PARAMETER_COUNT):
        raise ValueError('Original complete188 parameter/name traversal required')
    if (any(not isinstance(parameter, nn.Parameter) or parameter.dtype != torch.float32
            or parameter.layout != torch.strided for parameter in parameters)
            or len({id(parameter) for parameter in parameters}) != PARAMETER_COUNT
            or any(parameter.device != parameters[0].device for parameter in parameters)):
        raise ValueError('Distinct same-device dense original FP32 parameters required')
    if (W_NAME not in names or any(name not in names for name in GAIN_NAMES)
            or parameters[names.index(W_NAME)].shape != (6, 160)
            or any(parameters[names.index(name)].shape != (160,) for name in GAIN_NAMES)):
        raise ValueError('Actual original default W and three fusion gains must be in traversal')
    feature = [index for index, name in enumerate(names) if name.startswith('id_backbone.encoder.') and name != W_NAME]
    if not feature or any(names.index(name) not in feature for name in GAIN_NAMES):
        raise ValueError('Nonempty deployed encoder F including three gains required')
    main = _gradients(main_grads, parameters, 'main_grads')
    if not isinstance(rx_grads, tuple) or len(rx_grads) != 5:
        raise ValueError('Five original source RX gradient tuples required')
    receiver = tuple(_gradients(row, parameters, 'rx_grads') for row in rx_grads)
    if any(main[index] is None and any(row[index] is not None for row in receiver) for index in feature):
        raise ValueError('Unused F main gradient must be None in every same-graph RX gradient')
    if (not isinstance(saved_state, Mapping) or 'param_groups' not in saved_state
            or not isinstance(saved_state['param_groups'], list) or len(saved_state['param_groups']) != 1
            or not isinstance(saved_state['param_groups'][0], Mapping)):
        raise ValueError('Original single-group saved AdamW payload required')
    group = saved_state['param_groups'][0]
    if any(group.get(option) is not False for option in ('amsgrad', 'maximize', 'capturable', 'differentiable')):
        raise ValueError('AMSGrad/maximize/capturable/differentiable are unsupported by this native filter')
    if group.get('fused') not in (None, False):
        raise ValueError('Fused AdamW is outside the saved native filter implementation')
    if 'param_names' in group and group['param_names'] != list(names):
        raise ValueError('Saved parameter names differ from original traversal')
    states = saved_state.get('state')
    if not isinstance(states, Mapping):
        raise ValueError('Saved standard AdamW state mapping required')
    required = set(feature) | {index for index, gradient in enumerate(main) if gradient is not None}
    required |= {index for index in range(PARAMETER_COUNT) if any(row[index] is not None for row in receiver)}
    if any(index not in states for index in required):
        raise ValueError('F/current-gradient state is missing; no implicit moment initialization')
    original = copy.deepcopy(saved_state)
    live = [(parameter, parameter.detach().clone(), parameter._version, parameter.grad,
             None if parameter.grad is None else parameter.grad.detach().clone(), parameter.requires_grad)
            for parameter in parameters]
    main_before = tuple(None if gradient is None else gradient.clone() for gradient in main)
    receiver_before = tuple(tuple(None if gradient is None else gradient.clone() for gradient in row) for row in receiver)
    rng = _rng_snapshot()
    with _preserve_rng(), torch.no_grad():
        optimizer = make_optimizer(parameters, saved_state, 'RESTART_KEEP', restart_lr=restart_lr)
        options = optimizer.param_groups[0]
        beta1, beta2 = (float(value) for value in options['betas'])
        epsilon = float(options['eps'])
        if not 0. < beta1 < 1. or not 0. <= beta2 < 1.:
            raise ValueError('Native historical contribution requires beta1 in (0,1), beta2 in [0,1)')
        clipped, main_norm, clip_factor = _clip(parameters, main, max_grad_norm)
        history_chunks, contrast_chunks, scales, inactive = [], [[] for _ in range(5)], {}, []
        for index in feature:
            parameter, state = parameters[index], optimizer.state[parameters[index]]
            if main[index] is None:
                inactive.append(index)
                history_chunks.append(torch.zeros(parameter.numel(), dtype=torch.float64, device=parameter.device))
                for chunks in contrast_chunks:
                    chunks.append(torch.zeros(parameter.numel(), dtype=torch.float64, device=parameter.device))
                continue
            next_step = int(float(state['step'].clone().add_(1)))
            bc1, bc2 = 1. - beta1 ** next_step, 1. - beta2 ** next_step
            if bc1 <= 0. or bc2 <= 0.:
                raise ValueError('Invalid next-step AdamW bias corrections')
            # These are the actual native FP32 mul/addcmul/sqrt/div/add operations.
            prospective_v = state['exp_avg_sq'].clone().mul_(beta2).addcmul_(clipped[index], clipped[index], value=1. - beta2)
            denominator = prospective_v.sqrt().div_(math.sqrt(bc2)).add_(epsilon)
            if not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
                raise ValueError('Prospective native AdamW denominator must be finite and positive')
            scale = beta1 / (bc1 * denominator.double())
            scales[index] = scale
            history_chunks.append((state['exp_avg'].double() * scale).flatten())
            rows = [torch.zeros_like(parameter, dtype=torch.float64) if row[index] is None else row[index].double()
                    for row in receiver]
            mean = torch.stack(rows).mean(0)
            contrast_scale = (1. - beta1) * clip_factor / (bc1 * denominator.double())
            for chunks, row in zip(contrast_chunks, rows):
                chunks.append(((row - mean) * contrast_scale).flatten())
        history = torch.cat(history_chunks)
        contrasts = torch.stack([torch.cat(chunks) for chunks in contrast_chunks])
        filtered, projection = project_history(history, contrasts)
        selected = history if mode == 'KEEP' else torch.zeros_like(history) if mode == 'M_ZERO_F' else filtered
        actual_chunks, state_errors, moment_deltas = [], [], []
        offset = 0
        for index in feature:
            parameter, state = parameters[index], optimizer.state[parameters[index]]
            count = parameter.numel(); desired = selected[offset:offset + count].reshape(parameter.shape)
            old = state['exp_avg'].clone()
            if mode == 'M_ZERO_F':
                ideal_m = torch.zeros_like(parameter, dtype=torch.float64)
                state['exp_avg'].zero_()
            elif mode == 'RX_M_FILTER' and index not in inactive:
                ideal_m = desired / scales[index]
                mapped = ideal_m.to(dtype=torch.float32)
                if not bool(torch.isfinite(mapped).all()):
                    raise ValueError('Filtered history cannot be represented as finite native FP32 m')
                state['exp_avg'].copy_(mapped)
            else:
                ideal_m = old.double()
            state_errors.append(state['exp_avg'].double() - ideal_m)
            moment_deltas.append(state['exp_avg'].double() - old.double())
            actual_chunks.append(torch.zeros(count, dtype=torch.float64, device=parameter.device) if index in inactive
                                 else (state['exp_avg'].double() * scales[index]).flatten())
            offset += count
        actual_history = torch.cat(actual_chunks)
        residual = actual_history - selected
        removed = float((history - selected).norm())
        history_norm = float(history.norm())
        actual_products = contrasts @ actual_history
        row_norms = contrasts.norm(dim=1)
        normalized = [abs(float(product)) / (float(norm) * history_norm) if float(norm) * history_norm else 0.
                      for product, norm in zip(actual_products, row_norms)]
        result_state = optimizer.state_dict()
        # All modes keep complete v/step/non-F/W state and every group option.
        expected_group = copy.deepcopy(original['param_groups'][0]); expected_group['lr'] = 2e-4
        second_step_same = all(_equal(state['exp_avg_sq'], result_state['state'][index]['exp_avg_sq'])
                              and _equal(state['step'], result_state['state'][index]['step'])
                              for index, state in original['state'].items())
        nonfeature_same = all(_equal(state, result_state['state'][index]) for index, state in original['state'].items()
                              if index not in feature)
        absent_same = set(result_state['state']) == set(original['state'])
        none_same = all(_equal(original['state'][index]['exp_avg'], result_state['state'][index]['exp_avg'])
                        for index in inactive)
        if (not second_step_same or not nonfeature_same or not absent_same or (mode != 'M_ZERO_F' and not none_same)
                or not _equal(expected_group, result_state['param_groups'][0])):
            raise RuntimeError('History filtering changed v/step/non-F/W/options/None state')
        summary = dict(mode=mode, rank=projection['rank'], projection=projection,
            removed_norm=removed, removed_ratio=removed / history_norm if history_norm else 0.,
            history_contribution_norm_before=history_norm,
            history_contribution_norm_after_ideal=float(selected.norm()),
            history_contribution_norm_after_FP32=float(actual_history.norm()),
            exp_avg_change_norm=_norm(moment_deltas), FP32_m_mapping_error_norm=_norm(state_errors),
            rounding_residual_norm=float(residual.norm()),
            rounding_residual_max_abs=float(residual.abs().max()),
            actual_filtered_contrast_inner_products=[float(value) for value in actual_products],
            max_actual_normalized_inner_product_residual=max(normalized),
            clip_factor=clip_factor, main_gradient_norm=main_norm,
            clipped_main_gradient_norm=_norm([value for value in clipped if value is not None]),
            active_gradient_count=sum(value is not None for value in main),
            inactive_F_None_count=len(inactive), inactive_F_None_exp_avg_unchanged=none_same,
            F_tensor_count=len(feature), F_coordinate_count=len(history), F_names=[names[index] for index in feature],
            W_parameter_name=W_NAME, three_gains_in_F=True, rx_gradient_layout='5x188 original traversal',
            beta1=beta1, beta2=beta2, eps=epsilon, restart_lr=2e-4, max_grad_norm=5.,
            prospective_denominator='native FP32 v mul/addcmul/sqrt/div/add; FP64 u/h/projection',
            prospective_v_native_FP32_operations=True,
            prospective_step_uses_native_saved_scalar_add=True,
            exp_avg_sq_unchanged=True, step_unchanged=True, W_optimizer_state_unchanged=True,
            other_optimizer_state_unchanged=True, originally_absent_state_stays_absent=True,
            optimizer_options_same_except_restart_lr=True, actual_optimizer_steps=0)
    if (not _equal(original, saved_state) or not _equal(main_before, main) or not _equal(receiver_before, receiver)
            or not _equal(rng, _rng_snapshot())
            or any(parameter._version != version or not torch.equal(parameter.detach(), value)
                   or parameter.grad is not grad or parameter.requires_grad != flag
                   or (grad is not None and not torch.equal(grad, grad_value))
                   for parameter, value, version, grad, grad_value, flag in live)):
        raise RuntimeError('Momentum analysis changed live parameters/gradients/payload/inputs/RNG')
    summary.update(current_gradients_unchanged=True, live_parameters_gradients_payload_RNG_unchanged=True)
    return optimizer, summary
