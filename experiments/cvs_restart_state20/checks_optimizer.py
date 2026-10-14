"""Focused CPU AdamW restart-state checks, with no data/checkpoint loading."""
import copy
import json

import torch
from torch import nn

from .optimizer import make_optimizer
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


def _clone(parameters):
    return [nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad) for p in parameters]


def _raises(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected malformed AdamW payload rejection')


def _closed_form(before, state, gradient, group):
    """Independent scalar/tensor AdamW definition for this AMSGrad probe."""
    beta1, beta2 = group['betas']
    step = int(state['step']) + 1
    first = beta1 * state['exp_avg'] + (1 - beta1) * gradient
    second = beta2 * state['exp_avg_sq'] + (1 - beta2) * gradient.square()
    maximum = torch.maximum(state['max_exp_avg_sq'], second)
    direction = (first / (1 - beta1 ** step)) / (
        (maximum / (1 - beta2 ** step)).sqrt() + group['eps'])
    return before * (1 - group['lr'] * group['weight_decay']) - group['lr'] * direction


def check():
    torch.set_num_threads(2)
    parameters = [nn.Parameter(torch.tensor([.5, -.25], dtype=torch.float64)),
                  nn.Parameter(torch.tensor([2.], dtype=torch.float64)),
                  nn.Parameter(torch.tensor([3.], dtype=torch.float64), requires_grad=False)]
    live = torch.optim.AdamW(parameters, lr=.003, betas=(.85, .97), eps=1e-9,
                             weight_decay=.04, amsgrad=True, foreach=False)
    for _ in range(7):
        parameters[0].grad = torch.tensor([-1., 1.], dtype=torch.float64)
        parameters[1].grad = torch.tensor([-2.], dtype=torch.float64)
        live.step()
    live.param_groups[0]['lr'] = 1e-6
    live.param_groups[0]['initial_lr'] = .003
    parameters[0].grad = torch.tensor([.75, -.5], dtype=torch.float64)
    parameters[1].grad = None
    saved = copy.deepcopy(live.state_dict())
    snapshot = copy.deepcopy(saved)
    live_snapshot = copy.deepcopy(live.state_dict())
    before = [p.detach().clone() for p in parameters]
    versions = [p._version for p in parameters]
    flags = [p.requires_grad for p in parameters]
    grad_objects = [p.grad for p in parameters]
    grad_values = [None if p.grad is None else p.grad.clone() for p in parameters]
    rng = _rng_snapshot()
    factory = {mode: make_optimizer(parameters, saved, mode)
               for mode in ('END_SAVED', 'RESTART_KEEP', 'RESTART_RESET')}
    for mode, optimizer in factory.items():
        group = optimizer.param_groups[0]
        assert all(p is q for p, q in zip(group['params'], parameters))
        loaded = optimizer.state_dict()
        expected_group = copy.deepcopy(saved['param_groups'][0])
        if mode != 'END_SAVED':
            expected_group['lr'] = 2e-4
        assert _equal(loaded['param_groups'][0], expected_group)
        if mode == 'RESTART_RESET':
            assert not optimizer.state and not loaded['state']
        else:
            assert _equal(loaded['state'], saved['state'])
            for index, state in loaded['state'].items():
                for key, value in state.items():
                    assert value.data_ptr() != saved['state'][index][key].data_ptr()
    assert _equal(saved, snapshot) and _equal(live.state_dict(), live_snapshot)
    assert _equal(rng, _rng_snapshot())
    assert all(torch.equal(p.detach(), value) and p._version == version
               and p.requires_grad == flag and p.grad is grad
               and (grad is None or torch.equal(grad, grad_value))
               for p, value, version, flag, grad, grad_value in
               zip(parameters, before, versions, flags, grad_objects, grad_values))
    # Identical current gradients: saved moments differ only by LR between
    # END/KEEP, while KEEP/RESET share LR but have different moment histories.
    current = torch.tensor([1., -1.], dtype=torch.float64)
    after, steps = {}, {}
    for mode in factory:
        temporary = _clone(parameters)
        optimizer = make_optimizer(temporary, saved, mode)
        unused_state = copy.deepcopy(optimizer.state.get(temporary[1]))
        temporary[0].grad = current.clone()
        optimizer.step()
        after[mode] = temporary[0].detach().clone()
        steps[mode] = int(optimizer.state[temporary[0]]['step'])
        assert torch.equal(temporary[1].detach(), before[1])
        assert torch.equal(temporary[2].detach(), before[2])
        assert _equal(optimizer.state.get(temporary[1]), unused_state)
        assert temporary[2] not in optimizer.state
        assert [p.requires_grad for p in temporary] == flags
        if mode != 'RESTART_RESET':
            group = copy.deepcopy(saved['param_groups'][0])
            if mode == 'RESTART_KEEP':
                group['lr'] = 2e-4
            reference = _closed_form(before[0], saved['state'][0], current, group)
        else:
            # With empty state the first AdamW step has bias-corrected m=g,
            # v=g^2 (AMSGrad max is also g^2).
            reference = before[0] * (1 - 2e-4 * .04) - 2e-4 * current / (current.abs() + 1e-9)
        torch.testing.assert_close(after[mode], reference, rtol=1e-13, atol=1e-15)
    end_delta = after['END_SAVED'] - before[0]
    keep_delta = after['RESTART_KEEP'] - before[0]
    reset_delta = after['RESTART_RESET'] - before[0]
    torch.testing.assert_close(keep_delta, end_delta * 200, rtol=1e-8, atol=1e-14)
    assert keep_delta[0] > 0 and reset_delta[0] < 0
    assert steps == {'END_SAVED': 8, 'RESTART_KEEP': 8, 'RESTART_RESET': 1}
    # Same saved moments plus zero Tensor have decay/momentum semantics;
    # the None parameter above preserves its whole state and value.
    temporary = _clone(parameters)
    zero_optimizer = make_optimizer(temporary, saved, 'RESTART_KEEP')
    temporary[1].grad = torch.zeros_like(temporary[1])
    zero_optimizer.step()
    zero_displacement = float(temporary[1].detach() - before[1])
    assert zero_displacement != 0 and int(zero_optimizer.state[temporary[1]]['step']) == 8
    negatives = 0

    def reject(mutator, mode='END_SAVED'):
        nonlocal negatives
        bad = copy.deepcopy(saved)
        mutator(bad)
        _raises(lambda: make_optimizer(parameters, bad, mode))
        negatives += 1

    reject(lambda s: s['param_groups'].append(copy.deepcopy(s['param_groups'][0])))
    reject(lambda s: s['param_groups'][0].update(params=[1, 0, 2]))
    reject(lambda s: s['param_groups'][0].update(params=[0, 0, 2]))
    reject(lambda s: s['param_groups'][0].update(params=[0, 1]))
    reject(lambda s: s['param_groups'][0].update(params=[False, 1, 2]))
    reject(lambda s: s['param_groups'][0].pop('eps'))
    for key, value in (('lr', float('nan')), ('eps', float('inf')),
                       ('eps', -1.), ('weight_decay', -.1), ('betas', (.9, 1.)),
                       ('foreach', 'False'), ('amsgrad', 1), ('differentiable', True),
                       ('capturable', True)):
        reject(lambda s, key=key, value=value: s['param_groups'][0].update({key: value}))
    reject(lambda s: s['param_groups'][0].update(foreach=True, fused=True))
    reject(lambda s: s['param_groups'][0].update(decoupled_weight_decay=False))
    reject(lambda s: s['state'][0].update(exp_avg=torch.zeros(3, dtype=torch.float64)))
    reject(lambda s: s['state'][0].update(exp_avg=s['state'][0]['exp_avg'].float()))
    reject(lambda s: s['state'][0].update(exp_avg=s['state'][0]['exp_avg'].to('meta')))
    reject(lambda s: s['state'][0]['exp_avg'].fill_(float('nan')))
    reject(lambda s: s['state'][0]['exp_avg_sq'].fill_(-1.))
    reject(lambda s: s['state'][0].pop('max_exp_avg_sq'))
    reject(lambda s: s['state'][0].update(step=torch.tensor(7.5)))
    reject(lambda s: s['state'][0].update(step=torch.tensor(float('inf'))))
    reject(lambda s: s['state'].update({3: copy.deepcopy(s['state'][0])}))
    reject(lambda s: s['state'][0].update(extra_state=torch.tensor(0.)))
    reject(lambda s: s['param_groups'][0].update(initial_lr=float('nan')))
    reject(lambda s: s['state'][0]['exp_avg'].fill_(float('nan')), 'RESTART_RESET')
    for call in (lambda: make_optimizer(parameters, saved, 'UNKNOWN'),
                 lambda: make_optimizer(parameters, saved, 'RESTART_KEEP', restart_lr=1e-6),
                 lambda: make_optimizer(parameters + [parameters[0]], saved, 'END_SAVED'),
                 lambda: make_optimizer([p.detach() for p in parameters], saved, 'END_SAVED'),
                 lambda: make_optimizer([], saved, 'END_SAVED')):
        _raises(call)
        negatives += 1
    bad_parameters = _clone(parameters)
    with torch.no_grad():
        bad_parameters[0][0] = float('nan')
    _raises(lambda: make_optimizer(bad_parameters, saved, 'END_SAVED'))
    negatives += 1
    assert _equal(saved, snapshot) and _equal(live.state_dict(), live_snapshot)
    assert _equal(rng, _rng_snapshot())
    assert all(torch.equal(p.detach(), value) and p._version == version and p.grad is grad
               for p, value, version, grad in zip(parameters, before, versions, grad_objects))
    result = dict(status='PASS', device='cpu', torch_version=torch.__version__,
                  synthetic_prior_updates=7, saved_lr=1e-6, restart_lr=2e-4,
                  keep_to_end_displacement_ratio=200,
                  first_coordinate_deltas={mode: float(value[0]) for mode, value in
                                           (('END_SAVED', end_delta), ('RESTART_KEEP', keep_delta),
                                            ('RESTART_RESET', reset_delta))},
                  current_step_numbers=steps, None_unused_displacement=0.,
                  zero_tensor_saved_momentum_displacement=zero_displacement,
                  negative_checks=negatives, saved_payload_live_state_grad_flags_RNG_unchanged=True,
                  no_data_checkpoint_or_training=True)
    return result


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, allow_nan=False, indent=2))
