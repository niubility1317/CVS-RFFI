"""CPU-only tiny AdamW checks; no IQ, checkpoint, or CVS model is opened."""
from collections import OrderedDict
from copy import deepcopy
import json
import time

import torch

from .optimizer_state import restore_optimizer, state_summary, validate_saved


def _fixture():
    params = [torch.nn.Parameter(torch.tensor([1., -2.])),
              torch.nn.Parameter(torch.tensor([.4])),
              torch.nn.Parameter(torch.tensor([[.6]]))]
    opt = torch.optim.AdamW([{'params': params[:2]}, {'params': params[2:], 'lr': .02, 'weight_decay': 0.}],
                            lr=.002, weight_decay=.0001)
    opt._sinc_unit_policy = dict(mode='CENTER_UNITS', names=('sinc.low', 'sinc.band'), sinc=params[1:])
    # Match the actual factory: body first, then the two named Sinc leaves.
    opt.param_groups[0]['params'] = params[:1]
    opt.param_groups[1]['params'] = params[1:]
    opt._sinc_unit_steps = 0
    opt._sinc_physical_clip_calls = 0
    opt._sinc_unit_lr_records = []
    opt._sinc_pre_step = {}
    pre = lambda optimizer, args, kwargs: None
    post = lambda optimizer, args, kwargs: None
    opt.register_step_pre_hook(pre)
    opt.register_step_post_hook(post)
    names = [('body', params[0]), ('sinc.low', params[1]), ('sinc.band', params[2])]
    model = OrderedDict((name, p.detach().clone()) for name, p in names)
    saved = opt.state_dict()
    saved['param_groups'][0]['lr'] = .0000003
    saved['param_groups'][1]['lr'] = .000003
    saved['state'] = {
        0: dict(step=torch.tensor(4440.), exp_avg=torch.tensor([.2, -.3]), exp_avg_sq=torch.tensor([.08, .07])),
        1: dict(step=torch.tensor(3108.), exp_avg=torch.tensor([.15]), exp_avg_sq=torch.tensor([.04])),
        # A never-used parameter has no state; this is legal.
    }
    return opt, params, names, model, saved


def run():
    started = time.perf_counter()
    names_passed = []

    def passed(name):
        names_passed.append(name)

    def reject(name, mutate, *, full=True):
        opt, params, named, model, saved = _fixture()
        mutate(opt, params, named, model, saved)
        try:
            if full:
                restore_optimizer(opt, saved, named_parameters=named, parent_model=model)
            else:
                validate_saved(saved)
        except (ValueError, TypeError):
            passed(name)
            return
        raise AssertionError('Accepted malformed case: ' + name)

    opt, params, named, model, saved = _fixture()
    initial_options = [{k: deepcopy(v) for k, v in g.items() if k != 'params'} for g in opt.param_groups]
    original_defaults = deepcopy(opt.defaults)
    original_hooks = [tuple(opt._optimizer_step_pre_hooks.items()), tuple(opt._optimizer_step_post_hooks.items())]
    parent_snapshot = deepcopy(saved)
    summary = restore_optimizer(opt, saved, named_parameters=named, parent_model=model)
    assert summary['state_entries'] == 2 and summary['state_missing_parameters'] == 1
    assert summary['state_step_min'] == 3108 and summary['state_step_max'] == 4440
    assert summary['state_step_histogram'] == {'3108': 1, '4440': 1}
    assert params[2] not in opt.state
    assert [g['lr'] for g in opt.param_groups] == [.002, .02]
    assert all({k: v for k, v in g.items() if k != 'params'} == expected
               for g, expected in zip(opt.param_groups, initial_options))
    assert opt.defaults == original_defaults and opt._sinc_unit_steps == opt._sinc_physical_clip_calls == 0
    assert original_hooks == [tuple(opt._optimizer_step_pre_hooks.items()), tuple(opt._optimizer_step_post_hooks.items())]
    assert summary['all_state_tensors_equal'] and summary['parent_named_layout_checked'] and summary['parent_state_independent']
    json.dumps(summary, allow_nan=False)
    passed('matched_restore_missing_unused_state_fresh_options_hooks_and_counters')

    reset, reset_params, _, _, _ = _fixture()
    assert [g['lr'] for g in opt.param_groups] == [g['lr'] for g in reset.param_groups]
    empty = state_summary(reset)
    assert empty['state_entries'] == 0 and empty['state_step_min'] is None and empty['exp_avg_l2'] == 0
    passed('fresh_reset_empty_state_summary')
    for kept, fresh in zip(params, reset_params):
        kept.grad = torch.full_like(kept, .2)
        fresh.grad = torch.full_like(fresh, .2)
    opt.step()
    reset.step()
    assert all(torch.equal(saved['state'][pid][key], original)
               for pid, state in parent_snapshot['state'].items() for key, original in state.items())
    passed('parent_step_and_moments_unchanged_after_existing_tiny_update')
    assert not torch.equal(params[0], reset_params[0])
    after = state_summary(opt)
    assert after['state_step_histogram'] == {'1': 1, '3109': 1, '4441': 1}
    assert state_summary(reset)['state_step_histogram'] == {'1': 3}
    passed('keep_reset_same_fresh_lr_different_tiny_adamw_update')

    opt, _, named, model, saved = _fixture()
    saved['state'][1]['step'] = 1
    assert restore_optimizer(opt, saved, named_parameters=named, parent_model=model)['state_step_min'] == 1
    passed('positive_low_step_and_python_integer_allowed')

    reject('empty_state', lambda o, p, n, m, s: s.update(state={}), full=False)
    reject('external_top_level_state', lambda o, p, n, m, s: s.update(other={}), full=False)
    reject('external_parameter_state', lambda o, p, n, m, s: s['state'].update({99: deepcopy(s['state'][0])}), full=False)
    reject('external_moment_state', lambda o, p, n, m, s: s['state'][0].update(max_exp_avg_sq=torch.ones(2)), full=False)
    reject('missing_moment', lambda o, p, n, m, s: s['state'][0].pop('exp_avg'), full=False)
    reject('moment_shape', lambda o, p, n, m, s: s['state'][0].update(exp_avg=torch.ones(1), exp_avg_sq=torch.ones(1)))
    reject('moment_dtype', lambda o, p, n, m, s: s['state'][0].update(exp_avg=torch.ones(2, dtype=torch.float64), exp_avg_sq=torch.ones(2, dtype=torch.float64)))
    reject('nonfinite_first_moment', lambda o, p, n, m, s: s['state'][0]['exp_avg'].fill_(float('nan')), full=False)
    reject('nonfinite_second_moment', lambda o, p, n, m, s: s['state'][0]['exp_avg_sq'].fill_(float('inf')), full=False)
    reject('negative_second_moment', lambda o, p, n, m, s: s['state'][0]['exp_avg_sq'].fill_(-1.), full=False)
    reject('moment_autograd_graph', lambda o, p, n, m, s: s['state'][0]['exp_avg'].requires_grad_(True), full=False)
    for label, step in [('zero', 0.), ('negative', -1.), ('fractional', 2.5), ('nonfinite', float('nan')), ('over_budget', 4441.), ('bool', True)]:
        reject('step_' + label, lambda o, p, n, m, s, step=step: s['state'][0].update(step=step), full=False)
    reject('step_vector', lambda o, p, n, m, s: s['state'][0].update(step=torch.tensor([4.])), full=False)
    reject('group_count', lambda o, p, n, m, s: s['param_groups'].append(deepcopy(s['param_groups'][1])))
    reject('group_size', lambda o, p, n, m, s: s['param_groups'][0]['params'].append(s['param_groups'][1]['params'].pop(0)))
    reject('parameter_id_order', lambda o, p, n, m, s: s['param_groups'][1]['params'].reverse(), full=False)
    reject('group_order', lambda o, p, n, m, s: s['param_groups'].reverse(), full=False)
    reject('amsgrad', lambda o, p, n, m, s: s['param_groups'][0].update(amsgrad=True), full=False)
    reject('maximize', lambda o, p, n, m, s: s['param_groups'][0].update(maximize=True), full=False)
    reject('capturable', lambda o, p, n, m, s: s['param_groups'][0].update(capturable=True), full=False)
    reject('unknown_group_option', lambda o, p, n, m, s: s['param_groups'][0].update(external=1), full=False)
    reject('changed_beta', lambda o, p, n, m, s: s['param_groups'][0].update(betas=(.8, .999)))
    reject('bad_eps', lambda o, p, n, m, s: s['param_groups'][0].update(eps=0), full=False)
    reject('nonfinite_lr', lambda o, p, n, m, s: s['param_groups'][0].update(lr=float('inf')), full=False)
    reject('parent_model_name_order', lambda o, p, n, m, s: m.move_to_end('body'))
    reject('parent_model_shape', lambda o, p, n, m, s: m.update(body=torch.ones(5)))
    reject('fresh_group_parameter_order', lambda o, p, n, m, s: o.param_groups[1]['params'].reverse())
    reject('fresh_nonzero_counter', lambda o, p, n, m, s: setattr(o, '_sinc_unit_steps', 1))
    reject('fresh_nonempty_state', lambda o, p, n, m, s: o.state.update({p[0]: deepcopy(s['state'][0])}))
    opt, _, _, _, saved = _fixture()
    try:
        restore_optimizer(opt, saved)
    except ValueError:
        passed('center_units_requires_named_layout')
    else:
        raise AssertionError('CENTER_UNITS names omitted')
    return dict(status='PASS', checks=len(names_passed), passed=names_passed,
                wall_seconds=time.perf_counter() - started,
                real_CVS_NN=0, IQ_reads=0, checkpoint_reads=0,
                toy_optimizer_steps=2, sample_restore_summary=summary,
                toy_after_keep=after)


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, allow_nan=False))
