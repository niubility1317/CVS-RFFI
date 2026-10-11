"""Bounded native checks for whole-batch L style selection, without real data."""
import argparse
import json
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from . import style as s


def run(device='cpu'):
    from experiments.cvs_early_geometry import design
    from experiments.cvs_early_geometry.model import build
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    c = design.config(next(row for row in design.rows() if row['arm'] == 'PHYS'))
    model = build(c, device).train()
    assert not hasattr(model, 'multilevel')
    generator = torch.Generator().manual_seed(41133)
    base_x = torch.randn(30, 2, 256, generator=generator).to(device)
    index = torch.cat((torch.arange(30), torch.arange(2)))
    x = base_x[index.to(device)]
    y = torch.arange(6).repeat_interleave(5)[index]
    rx = torch.tensor([1, 3, 4, 6, 8]).repeat(6)[index]
    day = torch.ones(len(index), dtype=torch.long)
    ids = index.tolist()
    config = dict(model_seed=2026092701, style_selection='hard', candidates=4)
    step = 1333
    modes = [(m, m.training) for m in model.modules()]
    dropouts = [m for m in model.modules() if isinstance(m, nn.modules.dropout._DropoutNd)]
    assert dropouts and all(m.training for m in dropouts)
    hooks_before = sum(len(m._forward_hooks) for m in model.modules())
    state = s._snapshot(model)
    clean_rng = s.capture_rng(model)
    clean = model(x)
    after_clean_rng = s.capture_rng(model)
    observed, donor_checks = [], []
    actual_style = s.weak_mixstyle_u

    def checked(value, labels, receivers, days, physical_ids, seed, **kwargs):
        output, record = actual_style(value, labels, receivers, days, physical_ids, seed, **kwargs)
        assert kwargs == dict(p=1., alpha=.1, eta=1.)
        for i, j in enumerate(record['donors']):
            assert labels[i] == labels[j] and receivers[i] != receivers[j]
            assert days[i] == days[j] and physical_ids[i] != physical_ids[j]
        donor_checks.append(True)
        observed.append(dict(seed=seed, training=model.training, grad_enabled=torch.is_grad_enabled(),
                             output_requires_grad=output.requires_grad,
                             actual_delta=float((output.detach()-value.detach()).abs().max())))
        return output, record

    with patch.object(s, 'weak_mixstyle_u', side_effect=checked):
        hard, detail = s.styled_forward(model, x, y, rx, day, ids, step, config, 1., clean_rng)
    assert len(observed) == 10 and len(donor_checks) == 10
    assert all(not v['training'] and not v['grad_enabled'] and not v['output_requires_grad'] for v in observed[:8])
    assert all(v['training'] and v['grad_enabled'] and v['output_requires_grad'] for v in observed[8:])
    assert min(v['actual_delta'] for v in observed) > 0
    assert detail['selected_index'] == max(range(4), key=lambda i: detail['candidate_losses'][i])
    assert detail['eval_selected']['CE'] == max(detail['candidate_losses'])
    assert detail['actual_hook_order'] == list(s.LEVELS)
    assert detail['total_extra_forwards'] == 6 and detail['eval_readouts'] == 5
    assert detail['candidate_eval_forwards'] == 4 and detail['train_forwards'] == 1
    assert detail['candidate_state_unchanged'] and detail['candidate_rng_unchanged']
    assert detail['original_training_flags_restored'] and detail['main_rng_restored']
    assert detail['changed_packets'] > 0 and detail['duplicate_occurrences'] == 2
    assert s._same_state(model, state) and s._same_rng(after_clean_rng, s.capture_rng(model))
    assert all(m.training == mode for m, mode in modes)
    # Ensure the selected eval candidate's complete two-layer seeds are replayed.
    assert [r['seed'] for r in observed[-2:]] == [detail['selected_seed'], detail['selected_seed']+1709]
    named = list(model.base.named_parameters())
    loss = F.cross_entropy(hard, y.to(device)) + F.cross_entropy(clean, y.to(device))
    grads = torch.autograd.grad(loss, tuple(p for _, p in named), allow_unused=True)
    norms = {level: sum(float(g.detach().double().square().sum()) for (name, _), g in zip(named, grads)
                       if g is not None and 'id_backbone' in name and pattern in '.'+name)**.5
             for level, pattern in [('time_down', '.time_fuse.'), ('t2.pool', '.t2.')]}
    assert all(v > 0 for v in norms.values())
    random_logits, random_detail = s.styled_forward(model, x, y, rx, day, ids, step,
                                                   dict(config, style_selection='random'), 1., clean_rng)
    assert random_detail['candidate_seeds'] == detail['candidate_seeds']
    assert random_detail['candidate_losses'] == detail['candidate_losses']
    assert random_detail['selected_index'] == s._random_index(config['model_seed'], step)
    assert random_logits.requires_grad and random_detail['eval_readouts'] == 5
    _, tied = s.styled_forward(model, x, y, rx, day, ids, step, config, 0., clean_rng)
    assert len(set(tied['candidate_losses'])) == 1 and tied['selected_index'] == 0
    replay, replay_detail = s.styled_forward(model, x, y, rx, day, ids, step,
                                            dict(config, style_selection='replay'), 1., clean_rng)
    torch.testing.assert_close(replay, clean, rtol=0, atol=0)
    assert replay.requires_grad and not replay_detail['levels'] and not replay_detail['actual_hook_order']
    assert replay_detail['total_extra_forwards'] == 1 and replay_detail['eval_readouts'] == 0
    assert replay_detail['candidate_losses'] == [] and replay_detail['selected_index'] is None
    assert s._same_rng(after_clean_rng, s.capture_rng(model))
    invalid = []
    for kind in ('U_role', 'bad_PID', 'bad_RX', 'bad_day', 'bad_label', 'metadata_length', 'bad_candidate_count'):
        yy, rr, dd, pp = y.clone(), rx.clone(), day.clone(), list(ids)
        cc = dict(config)
        if kind == 'U_role': cc['source_role'] = 'U_s'
        elif kind == 'bad_PID': pp[5] = pp[0]
        elif kind == 'bad_RX': rr[0] = 0
        elif kind == 'bad_day': dd[0] = 0
        elif kind == 'bad_label': yy[0] = 6
        elif kind == 'metadata_length': dd = dd[:-1]
        else: cc['candidates'] = 3
        try:
            s.styled_forward(model, x, yy, rr, dd, pp, step, cc, 1., clean_rng)
        except ValueError:
            invalid.append(kind)
        else:
            raise AssertionError('Illegal L style input accepted: '+kind)
    # Force failure while selected hooks exist, then verify a usable next call.
    with patch.object(s, 'weak_mixstyle_u', side_effect=RuntimeError('intentional hook failure')):
        try:
            s.styled_forward(model, x, y, rx, day, ids, step, config, 1., clean_rng)
        except RuntimeError as error:
            assert str(error) == 'intentional hook failure'
        else:
            raise AssertionError('Intentional hook failure did not propagate')
    assert sum(len(m._forward_hooks) for m in model.modules()) == hooks_before
    assert all(m.training == mode for m, mode in modes)
    assert s._same_rng(after_clean_rng, s.capture_rng(model)) and s._same_state(model, state)
    def fail_training_hook(*args, **kwargs):
        if model.training:
            raise RuntimeError('intentional selected training failure')
        return actual_style(*args, **kwargs)
    with patch.object(s, 'weak_mixstyle_u', side_effect=fail_training_hook):
        try:
            s.styled_forward(model, x, y, rx, day, ids, step, config, 1., clean_rng)
        except RuntimeError as error:
            assert str(error) == 'intentional selected training failure'
        else:
            raise AssertionError('Selected-training hook failure did not propagate')
    assert sum(len(m._forward_hooks) for m in model.modules()) == hooks_before
    assert all(m.training == mode for m, mode in modes)
    assert s._same_rng(after_clean_rng, s.capture_rng(model))
    replay_again, _ = s.styled_forward(model, x, y, rx, day, ids, step,
                                      dict(config, style_selection='replay'), 1., clean_rng)
    torch.testing.assert_close(replay_again, clean, rtol=0, atol=0)
    # A stochastic eval forward is rejected and still unwinds RNG/modes/hooks.
    original_forward = model.forward
    def bad_eval(value):
        if not model.training:
            torch.rand(1, device=value.device)
        return original_forward(value)
    with patch.object(model, 'forward', side_effect=bad_eval):
        try:
            s.styled_forward(model, x, y, rx, day, ids, step, config, 1., clean_rng)
        except RuntimeError as error:
            assert 'global RNG' in str(error)
        else:
            raise AssertionError('Stochastic candidate eval was not detected')
    assert sum(len(m._forward_hooks) for m in model.modules()) == hooks_before
    assert all(m.training == mode for m, mode in modes)
    assert s._same_rng(after_clean_rng, s.capture_rng(model))
    # Simulate a faulty eval module that mutates a persistent buffer. Detection
    # must reject it and restore that buffer without rewriting unchanged weights.
    model.register_buffer('_style_check_state', torch.zeros(1, device=device))
    def mutating_eval(value):
        if not model.training:
            model._style_check_state.add_(1)
        return original_forward(value)
    try:
        with patch.object(model, 'forward', side_effect=mutating_eval):
            try:
                s.styled_forward(model, x, y, rx, day, ids, step, config, 1., clean_rng)
            except RuntimeError as error:
                assert 'model state' in str(error)
            else:
                raise AssertionError('Mutable candidate state was not detected')
        assert model._style_check_state.item() == 0.
    finally:
        delattr(model, '_style_check_state')
    assert s._same_state(model, state) and s._same_rng(after_clean_rng, s.capture_rng(model))
    assert all(m.training == mode for m, mode in modes)
    assert sum(len(m._forward_hooks) for m in model.modules()) == hooks_before
    json.dumps([detail, random_detail, tied, replay_detail], allow_nan=False)
    return dict(status='PASS', device=device, real_checkpoint_loaded=False, target_access=False,
                synthetic_L_only=True, headless_native_model=True, whole_batch_argmax=True,
                smallest_index_tie=True, random_private_uniform_index=True,
                candidate_pool_shared_between_arms=True, clean_plus_four_eval_readouts=True,
                selected_complete_candidate_replayed=True, eval_no_grad_training_replay_grad=True,
                duplicate_occurrences_allowed=True, real_PID_donor_constraints=True,
                same_mask_replay_bitwise_equal=True, main_RNG_restored=True,
                model_state_and_training_flags_unchanged=True, native_gradient_norms=norms,
                exception_hook_cleanup=True, selected_train_failure_cleanup=True,
                stochastic_eval_rejected=True, mutating_eval_rejected_and_restored=True,
                invalid_inputs_rejected=invalid, hard_selected_index=detail['selected_index'],
                random_selected_index=random_detail['selected_index'], candidate_losses=detail['candidate_losses'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2, allow_nan=False))
