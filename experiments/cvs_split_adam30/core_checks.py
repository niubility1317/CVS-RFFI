"""No-Torch mathematical oracles plus focused checks of actual pure helpers.

These checks do not replace the launcher's actual-Torch real-checkpoint smoke.
"""
import ast
import math
from pathlib import Path


def _actual_helpers():
    source = Path(__file__).with_name('core.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    keep = {'_mode', '_finite_number', '_hyperparameters', '_clip_factor'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    namespace = {'math': math, 'MODES': ('MEASURE', 'SPLIT')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_core_pure_helpers', 'exec'), namespace)
    return namespace, tree


def _branch(state, gradient, beta1, beta2, eps):
    if gradient is None:
        return state, 0.
    step, first, second = state
    step += 1
    first = beta1 * first + (1 - beta1) * gradient
    second = beta2 * second + (1 - beta2) * gradient ** 2
    direction = (first / (1 - beta1 ** step)) / (math.sqrt(second / (1 - beta2 ** step)) + eps)
    return (step, first, second), direction


def _advance(parameter, rest_state, as_state, rest, satellite, *, lr=.01, wd=.1,
             beta1=.9, beta2=.99, eps=1e-8, native_active=True):
    if not native_active:
        return parameter, rest_state, as_state
    rest_state, r = _branch(rest_state, rest, beta1, beta2, eps)
    as_state, a = _branch(as_state, satellite, beta1, beta2, eps)
    return parameter * (1 - lr * wd) - lr * (.8 * r + .2 * a), rest_state, as_state


def run():
    helper, tree = _actual_helpers()
    done = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        done.append(name)

    def rejects(name, fn):
        try:
            fn()
        except (ValueError, TypeError, KeyError, OverflowError):
            done.append(name)
        else:
            raise AssertionError(name)

    group = dict(lr=.01, eps=1e-8, weight_decay=.1, betas=(.9, .99))
    check('actual_hyperparameters', helper['_hyperparameters'](group) == (.9, .99, .01, 1e-8, .1))
    check('actual_measure_mode', helper['_mode']('MEASURE') == 'MEASURE')
    check('actual_split_mode', helper['_mode']('SPLIT') == 'SPLIT')
    rejects('reject_unregistered_mode', lambda: helper['_mode']('SHARED'))
    for flag in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'fused', 'foreach'):
        rejects('reject_' + flag, lambda f=flag: helper['_hyperparameters'](dict(group, **{f: True})))
    for key in ('lr', 'eps', 'weight_decay'):
        rejects('reject_' + key + '_nan', lambda k=key: helper['_hyperparameters'](dict(group, **{k: float('nan')})))
        rejects('reject_' + key + '_inf', lambda k=key: helper['_hyperparameters'](dict(group, **{k: float('inf')})))
    for key in ('lr', 'eps'):
        rejects('reject_' + key + '_zero', lambda k=key: helper['_hyperparameters'](dict(group, **{k: 0.})))
    rejects('reject_negative_decay', lambda: helper['_hyperparameters'](dict(group, weight_decay=-1.)))
    for betas in ((-1., .9), (.9, 1.), (float('nan'), .9), (.9,), (.9, .99, .9)):
        rejects('reject_betas_' + str(betas), lambda b=betas: helper['_hyperparameters'](dict(group, betas=b)))
    check('actual_common_clip', math.isclose(helper['_clip_factor'](5., 10.), 5 / (10 + 1e-6)))
    check('actual_no_clip', helper['_clip_factor'](5., 1.) == 1.)
    for limit, norm in ((4., 1.), (5., -1.), (5., float('inf')), (float('nan'), 1.)):
        rejects('reject_clip_' + str((limit, norm)), lambda l=limit, n=norm: helper['_clip_factor'](l, n))

    zero = (0, 0., 0.)
    p, r, a = _advance(2., zero, zero, 3., -2.)
    expected = 2. * (1 - .01 * .1) - .01 * (.8 * 3 / (3 + 1e-8) - .2 * 2 / (2 + 1e-8))
    check('first_step_weighted_normalized_displacement', math.isclose(p, expected, rel_tol=1e-14))
    check('first_moments_separate', math.isclose(r[1], .3) and math.isclose(a[1], -.2))
    check('second_moments_separate', math.isclose(r[2], .09) and math.isclose(a[2], .04))
    p2, r2, a2 = _advance(p, r, a, -1., 4.)
    mr = .9 * .3 + .1 * -1.
    vr = .99 * .09 + .01 * 1.
    ma = .9 * -.2 + .1 * 4.
    va = .99 * .04 + .01 * 16.
    ur = mr / (1 - .9 ** 2) / (math.sqrt(vr / (1 - .99 ** 2)) + 1e-8)
    ua = ma / (1 - .9 ** 2) / (math.sqrt(va / (1 - .99 ** 2)) + 1e-8)
    check('two_step_closed_form', math.isclose(p2, p * .999 - .01 * (.8 * ur + .2 * ua), rel_tol=1e-14))
    check('two_independent_counters', r2[0] == a2[0] == 2)
    _, inactive_r, inactive_a = _advance(2., r, a, None, None, native_active=False)
    check('native_none_preserves_both_states', inactive_r == r and inactive_a == a)
    untouched, _, _ = _advance(2., r, a, None, None, native_active=False)
    check('native_none_skips_decay', untouched == 2.)
    p_none, r_none, a_none = _advance(2., r, a, 0., None)
    p_zero, r_zero, a_zero = _advance(2., r, a, 0., 0.)
    check('none_branch_skips_counter', r_none[0] == 2 and a_none == a)
    check('zero_branch_advances_counter', r_zero[0] == 2 and a_zero[0] == 2)
    check('zero_retains_momentum_effect', p_zero != p_none)
    decay_only, _, _ = _advance(2., zero, zero, 0., 0.)
    check('weight_decay_once', decay_only == 2. * .999)
    sinc_adaptive, _, _ = _advance(2. * .999, zero, zero, 0., 0., wd=0.)
    check('sinc_external_decay_not_repeated', sinc_adaptive == 2. * .999)
    factor = helper['_clip_factor'](5., 10.)
    check('same_clip_preserves_component_sum', math.isclose(factor * 3. + factor * -2., factor * (3. - 2.)))
    _, shared = _branch(zero, 3. - 2., .9, .99, 1e-8)
    _, rr = _branch(zero, 3., .9, .99, 1e-8)
    _, aa = _branch(zero, -2., .9, .99, 1e-8)
    check('split_not_misreported_as_same_effective_as_weight', not math.isclose(shared, .8 * rr + .2 * aa, rel_tol=1e-3))
    # Merely computing two diagnostic branch gradients cannot mutate shared
    # AdamW state; the MEASURE implementation must not install a split step.
    shared_before = _branch(zero, 1., .9, .99, 1e-8)
    _branch(zero, 3., .9, .99, 1e-8)
    _branch(zero, -2., .9, .99, 1e-8)
    shared_after = _branch(zero, 1., .9, .99, 1e-8)
    check('shared_measure_mathematical_control_unchanged', shared_before == shared_after)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    check('actual_capture_interface', [a.arg for a in functions['capture'].args.args] == ['model', 'total_loss', 'Asloss', 'optimizer', 'step'])
    check('actual_clip_interface', [a.arg for a in functions['after_clip'].args.args] == ['model', 'parameters', 'max_norm', 'total_norm'])
    check('actual_attach_interface', [a.arg for a in functions['attach_optimizer'].args.args] == ['optimizer', 'model'])
    check('actual_two_vjp_calls', sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'grad' for n in ast.walk(functions['capture'])) == 2)
    return dict(status='PASS', checks=len(done), names=done,
                scope='pure_python_math_and_actual_pure_helpers; actual_Torch_smoke_required')


def actual_torch_smoke():
    """Actual CPU optimizer checks, with fixed toy tensors and zero NN forwards.

    No checkpoint, IQ, data loader or model forward is opened. These toys do
    not simulate physical Sinc units; the real-checkpoint preflight covers it.
    """
    import copy
    import torch
    from torch.optim.adamw import AdamW
    from experiments.cvs_split_adam30 import core

    done = []
    largest_oracle_error = 0.
    largest_measure_error = 0.
    optimizer_steps = 0

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        done.append(name)

    def rejects(name, fn):
        try:
            fn()
        except (ValueError, FloatingPointError):
            done.append(name)
        else:
            raise AssertionError(name)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.main = torch.nn.Parameter(torch.tensor([.7, -.4], dtype=torch.float64))
            self.rest_only = torch.nn.Parameter(torch.tensor([1.2], dtype=torch.float64))
            self.zero_as = torch.nn.Parameter(torch.tensor([-.8], dtype=torch.float64))
            self.unused = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))

    settings = dict(lr=.02, betas=(.6, .8), eps=.03, weight_decay=.1, foreach=False)

    def losses(model, index, as_on=True):
        target_rest = model.main.new_tensor([1.5 + .1 * index, -2. + .05 * index])
        target_as = model.main.new_tensor([-.6 + .03 * index, .2 - .04 * index])
        rest = 3. * (model.main - target_rest).square().sum() + .3 * model.rest_only.square().sum() + 0. * model.zero_as.sum()
        satellite = .25 * (model.main - target_as).square().sum() + 0. * model.zero_as.sum()
        if not as_on:
            satellite = satellite * 0.
        return rest + satellite, satellite

    def make(mode=None):
        model = Toy()
        opt = AdamW(model.parameters(), **settings)
        events = []

        def existing_pre(optimizer, args, kwargs):
            events.append(('pre', float(model.main.detach()[0])))

        def existing_post(optimizer, args, kwargs):
            events.append(('post', float(model.main.detach()[0])))

        # Existing hooks are deliberately registered before the core hooks.
        opt.register_step_pre_hook(existing_pre)
        opt.register_step_post_hook(existing_post)
        if mode is not None:
            core.initialize(model, mode)
            core.attach_optimizer(opt, model, amp=False)
        return model, opt, events

    def prepared(model, opt, index, *, as_on=True, captured=True):
        opt.zero_grad(set_to_none=True)
        total, satellite = losses(model, index, as_on)
        if captured:
            core.capture(model, total, satellite, opt, model._sa_steps + 1)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 5.)
        if captured:
            core.after_clip(model, list(model.parameters()), 5., norm)
        gradients = tuple(None if p.grad is None else p.grad.clone() for p in model.parameters())
        return gradients

    def advance(model, opt, index, *, as_on=True, captured=True):
        nonlocal optimizer_steps
        before = prepared(model, opt, index, as_on=as_on, captured=captured)
        opt.step()
        optimizer_steps += 1
        if not all((g is None and p.grad is None) or
                   (g is not None and p.grad is not None and torch.equal(g, p.grad))
                   for p, g in zip(model.parameters(), before)):
            raise AssertionError('optimizer_changed_native_clipped_gradients')

    def state_error(first, second):
        maximum = 0.
        if first.keys() != second.keys():
            raise AssertionError('optimizer_state_keys_changed')
        for key in first:
            a, b = first[key], second[key]
            if torch.is_tensor(a):
                if not torch.is_tensor(b) or a.shape != b.shape:
                    raise AssertionError('optimizer_state_tensor_changed')
                maximum = max(maximum, float((a - b).abs().max()))
            elif a != b:
                raise AssertionError('optimizer_state_scalar_changed')
        return maximum

    # Both As and graph-connected-zero As controls preserve native AdamW.
    for as_on in (True, False):
        measured, measure_opt, measure_events = make('MEASURE')
        native, native_opt, native_events = make()
        for index in (1, 2):
            advance(measured, measure_opt, index, as_on=as_on)
            advance(native, native_opt, index, as_on=as_on, captured=False)
            for mp, np in zip(measured.parameters(), native.parameters()):
                largest_measure_error = max(largest_measure_error, float((mp - np).abs().max()),
                    state_error(measure_opt.state.get(mp, {}), native_opt.state.get(np, {})))
        check('measure_' + ('as' if as_on else 'zero_as') + '_native_parameter_state', largest_measure_error <= 1e-13)
        check('measure_' + str(as_on) + '_existing_hooks_once',
              [event[0] for event in measure_events] == ['pre', 'post', 'pre', 'post'] and
              [event[0] for event in native_events] == ['pre', 'post', 'pre', 'post'])
        check('measure_' + str(as_on) + '_two_vjps_each', measured._sa_vjp_calls == 4 and measured._sa_steps == 2)
        if as_on:
            native_shared_state = copy.deepcopy(native_opt.state_dict())

    split, split_opt, split_events = make('SPLIT')
    oracle_parameters = {n: [float(v) for v in p.detach()] for n, p in split.named_parameters()}
    oracle_states = {n: [((0, 0., 0.), (0, 0., 0.)) for _ in p] for n, p in oracle_parameters.items()}

    def oracle(index):
        nonlocal largest_oracle_error
        current = {n: list(values) for n, values in oracle_parameters.items()}
        gradients = dict(
            main=([6. * (current['main'][0] - (1.5 + .1 * index)),
                   6. * (current['main'][1] - (-2. + .05 * index))],
                  [.5 * (current['main'][0] - (-.6 + .03 * index)),
                   .5 * (current['main'][1] - (.2 - .04 * index))]),
            rest_only=([.6 * current['rest_only'][0]], [None]),
            zero_as=([0.], [0.]), unused=([None], [None]))
        norm = math.sqrt(sum((float(r or 0.) + float(a or 0.)) ** 2
                            for rest, satellite in gradients.values()
                            for r, a in zip(rest, satellite)))
        factor = min(1., 5. / (norm + 1e-6))
        for name, values in current.items():
            rest, satellite = gradients[name]
            for coordinate, value in enumerate(values):
                r = None if rest[coordinate] is None else rest[coordinate] * factor
                a = None if satellite[coordinate] is None else satellite[coordinate] * factor
                rstate, astate = oracle_states[name][coordinate]
                updated, rstate, astate = _advance(value, rstate, astate, r, a,
                    lr=settings['lr'], wd=settings['weight_decay'],
                    beta1=settings['betas'][0], beta2=settings['betas'][1], eps=settings['eps'],
                    native_active=name != 'unused')
                oracle_parameters[name][coordinate] = updated
                oracle_states[name][coordinate] = (rstate, astate)

    for index in (1, 2):
        oracle(index)
        advance(split, split_opt, index)
        for name, p in split.named_parameters():
            predicted = p.new_tensor(oracle_parameters[name])
            largest_oracle_error = max(largest_oracle_error, float((p - predicted).abs().max()))
            state = split_opt.state.get(p, {})
            for part_index, part in enumerate(('rest', 'as')):
                states = [pair[part_index] for pair in oracle_states[name]]
                count = states[0][0]
                if state.get('_sa_' + part + '_step', 0) != count:
                    raise AssertionError('independent_branch_counter_disagrees_with_oracle')
                if count:
                    for entry, offset in (('exp_avg', 1), ('exp_avg_sq', 2)):
                        expected = p.new_tensor([value[offset] for value in states])
                        largest_oracle_error = max(largest_oracle_error,
                            float((state['_sa_' + part + '_' + entry] - expected).abs().max()))
    check('split_two_step_parameter_and_moment_oracle', largest_oracle_error <= 1e-12)
    check('split_existing_hooks_once_in_order', [v[0] for v in split_events] == ['pre', 'post', 'pre', 'post'] and
          all(split_events[i][1] != split_events[i + 1][1] for i in (0, 2)))
    check('split_native_gradients_retained', all(record['component_sum_max_abs_error'] <= 1e-12 for record in split._sa_records))
    check('split_shared_clip_factor_exercised', all(record['clip_factor'] < 1. for record in split._sa_records))
    check('split_rest_only_as_none_counter', '_sa_as_step' not in split_opt.state[split.rest_only])
    check('split_graph_zero_as_counter', split_opt.state[split.zero_as]['_sa_as_step'] == 2)
    check('split_unused_none_no_state_or_decay', not split_opt.state.get(split.unused) and float(split.unused[0]) == 2.)
    check('split_active_zero_decay_exactly_once', math.isclose(float(split.zero_as[0]), -.8 * (1 - .02 * .1) ** 2, abs_tol=1e-15))

    # Actual native state_dict load calls AdamW.__setstate__, not a hand copy.
    restored, restored_opt, restored_events = make('SPLIT')
    restored.load_state_dict(copy.deepcopy(split.state_dict()))
    restored_opt.load_state_dict(copy.deepcopy(split_opt.state_dict()))
    advance(split, split_opt, 3)
    advance(restored, restored_opt, 3)
    roundtrip_error = 0.
    for p, q in zip(split.parameters(), restored.parameters()):
        roundtrip_error = max(roundtrip_error, float((p - q).abs().max()),
                              state_error(split_opt.state.get(p, {}), restored_opt.state.get(q, {})))
    check('split_state_dict_roundtrip_third_step', roundtrip_error <= 1e-13)
    check('split_restored_hooks_once', [v[0] for v in restored_events] == ['pre', 'post'])
    check('split_restored_independent_counter_continues',
          restored_opt.state[restored.main]['_sa_rest_step'] == 3 and
          restored_opt.state[restored.main]['_sa_as_step'] == 3)

    bad = Toy()
    core.initialize(bad, 'SPLIT')
    rejects('actual_amp_rejected', lambda: core.attach_optimizer(AdamW(bad.parameters(), **settings), bad, amp=True))
    rejects('actual_unregistered_mode_rejected', lambda: core.initialize(Toy(), 'SHARED'))
    bad_flags = Toy()
    core.initialize(bad_flags, 'SPLIT')
    rejects('actual_unsupported_adamw_flag_rejected',
            lambda: core.attach_optimizer(AdamW(bad_flags.parameters(), **dict(settings, amsgrad=True)), bad_flags))
    bad_shared = Toy()
    shared_opt = AdamW(bad_shared.parameters(), **settings)
    shared_opt.load_state_dict(native_shared_state)
    core.initialize(bad_shared, 'SPLIT')
    rejects('actual_shared_moments_cannot_become_split', lambda: core.attach_optimizer(shared_opt, bad_shared))
    bad, opt, _ = make('SPLIT')
    rejects('actual_clip_before_capture_rejected', lambda: core.after_clip(bad, list(bad.parameters()), 5., 1.))
    rejects('actual_nonfinite_loss_rejected', lambda: core.capture(bad, bad.main.sum() * float('inf'), 0. * bad.main.sum(), opt, 1))
    total, satellite = losses(bad, 1)
    core.capture(bad, total, satellite, opt, 1)
    rejects('actual_duplicate_capture_rejected', lambda: core.capture(bad, total, satellite, opt, 1))
    inconsistent, opt, _ = make('SPLIT')
    opt.zero_grad(set_to_none=True)
    total, satellite = losses(inconsistent, 1)
    core.capture(inconsistent, total, satellite, opt, 1)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(list(inconsistent.parameters()), 5.)
    inconsistent.main.grad.add_(.2)
    rejects('actual_wrong_total_gradient_rejected', lambda: core.after_clip(inconsistent, list(inconsistent.parameters()), 5., norm))
    stale, opt, _ = make('SPLIT')
    prepared(stale, opt, 1)
    stale.main.grad.add_(.1)
    rejects('actual_postclip_gradient_mutation_rejected', opt.step)
    singular = Toy()
    with torch.no_grad():
        singular.rest_only.zero_()
    core.initialize(singular, 'SPLIT')
    singular_opt = AdamW(singular.parameters(), **settings)
    core.attach_optimizer(singular_opt, singular)
    finite_loss = singular.rest_only.sqrt().sum() + 0. * singular.main.sum()
    rejects('actual_finite_loss_nonfinite_vjp_rejected',
            lambda: core.capture(singular, finite_loss, 0. * singular.main.sum(), singular_opt, 1))
    check('actual_singular_vjp_did_not_accumulate_parameter_gradients', all(p.grad is None for p in singular.parameters()))
    return dict(status='PASS', checks=len(done), names=done, device='cpu', dtype='float64',
                optimizer_steps=optimizer_steps, NN_forwards=0, checkpoint_loads=0, IQ_reads=0,
                measure_parameter_state_max_abs_error=largest_measure_error,
                split_parameter_moment_oracle_max_abs_error=largest_oracle_error,
                state_dict_roundtrip_max_abs_error=roundtrip_error,
                physical_Sinc_units='not_simulated; real_checkpoint_preflight_required')


if __name__ == '__main__':
    import json
    print(json.dumps(run(), ensure_ascii=False, indent=2))
