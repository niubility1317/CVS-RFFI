"""Native main-path L-style routing checks, with synthetic L and no checkpoint."""
import argparse
import json
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import style as s


def run(device='cpu'):
    from experiments.cvs_early_geometry import design as fixture_design
    from experiments.cvs_early_geometry.model import build
    from . import design
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fixture = fixture_design.config(next(row for row in fixture_design.rows() if row['arm'] == 'PHYS'))
    model = build(fixture, device)
    assert not hasattr(model, 'multilevel'), 'Style-only checks must not depend on auxiliary heads'
    model.train()
    generator = torch.Generator().manual_seed(32911)
    x = torch.randn(60, 2, 256, generator=generator).to(device)
    labels = torch.arange(6).repeat_interleave(10)
    rx = torch.tensor(s.SOURCE_RX).repeat_interleave(2).repeat(6)
    days = torch.ones(60, dtype=torch.long)
    index = torch.cat((torch.arange(60), torch.arange(4)))
    x = x[index.to(device)]
    labels, rx, days = (v[index] for v in (labels, rx, days))
    ids = index.tolist()
    configs = {row['arm']: dict(design.config(row)['multilevel'], model_seed=row['model_seed'])
               for row in design.rows() if row['model_seed'] == design.SEEDS[0]}
    seed = design.SEEDS[0]
    step = next(i for i in range(1, 65) if all(np.random.default_rng(seed+i*701+j*1709).random() < .5 for j in range(2)))
    modules, paths = s._paths(model)
    start_hooks = sum(len(m._forward_hooks) for m in model.modules())
    modes = [(m, m.training) for m in model.modules()]
    drops = [(m, m.training) for m in model.modules() if isinstance(m, nn.modules.dropout._DropoutNd)]
    assert drops and all(mode for _, mode in drops)

    def rng():
        return [torch.random.get_rng_state().clone()]+([v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])

    def same_rng(before):
        assert all(torch.equal(a, b) for a, b in zip(before, rng()))

    def stable_forward():
        devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
        with torch.random.fork_rng(devices=devices):
            return model(x)

    untouched = {k: v.detach().clone() for k, v in model.state_dict().items()}
    baseline = stable_forward().detach()
    routing = {}
    for arm in ('STYLE_SHALLOW', 'STYLE_MIDDLE', 'STYLE_BOTH'):
        config = configs[arm]
        assert not config['enabled'] and config['mixstyle_l']
        before = rng()
        map_deltas = {}
        actual = s.weak_mixstyle_u

        def spy(value, *args, **kwargs):
            output, detail = actual(value, *args, **kwargs)
            map_deltas[len(map_deltas)] = float((output.detach()-value.detach()).abs().max())
            return output, detail

        with patch.object(s, 'weak_mixstyle_u', side_effect=spy):
            with s.labeled_style(model, labels, rx, days, ids, step, config, 1.) as detail:
                assert all(m.training == mode for m, mode in modes)
                logits = model(x)
                assert all(m.training for m, _ in drops)
        same_rng(before)
        assert list(detail['levels']) == config['style_levels'] == detail['actual_hook_order']
        assert list(map_deltas.values()) and min(map_deltas.values()) > 0
        assert detail['changed_packets'] > 0 and detail['styled_forwards'] == 1
        assert detail['duplicate_occurrences'] == 4
        for level, record in detail['levels'].items():
            assert record['path'] == paths[level] and record['enabled'] and record['eta'] == 1.
            assert record['input_requires_grad'] and record['output_requires_grad']
            for i, j in enumerate(record['donors']):
                assert labels[i] == labels[j] and rx[i] != rx[j] and days[i] == days[j] and ids[i] != ids[j]
        loss = F.cross_entropy(logits, labels.to(device))
        named = list(model.base.named_parameters())
        gradients = torch.autograd.grad(loss, tuple(p for _, p in named), allow_unused=True)
        gradient_norm = sum(float(v.detach().double().square().sum()) for v in gradients if v is not None)**.5
        groups = {level: sum(float(g.detach().double().square().sum()) for (name, _), g in zip(named, gradients)
                             if g is not None and 'id_backbone' in name and pattern in '.'+name)**.5
                  for level, pattern in [('time_down', '.time_fuse.'), ('t2.pool', '.t2.')]}
        assert all(value > 0 for value in groups.values())
        assert gradient_norm > 0 and not torch.equal(logits.detach(), baseline)
        assert sum(len(m._forward_hooks) for m in model.modules()) == start_hooks
        assert all(m.training == mode for m, mode in modes)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in untouched.items())
        torch.testing.assert_close(stable_forward().detach(), baseline, rtol=0, atol=0)
        routing[arm] = dict(actual_hook_order=detail['actual_hook_order'], changed_packets=detail['changed_packets'],
                            actual_map_max_abs_change=list(map_deltas.values()), real_backbone_gradient_norm=gradient_norm,
                            native_convolution_gradient_norms=groups)
    before = rng()
    with s.labeled_style(model, labels, rx, days, ids, step, configs['STYLE_BOTH'], .4) as ramped:
        model(x)
    same_rng(before)
    assert ramped['eta'] == .4 and all(r['movement_max'] <= .4+1e-6 for r in ramped['levels'].values())
    # A zero-ramp forward must have ordinary training dropout and match exactly
    # the same-RNG plain forward, while still restoring the extra-forward RNG.
    with s.labeled_style(model, labels, rx, days, ids, step, configs['STYLE_BOTH'], 0.) as zero:
        zero_logits = model(x)
    same_rng(before)
    torch.testing.assert_close(zero_logits.detach(), baseline, rtol=0, atol=0)
    assert not zero['levels'] and not zero['enabled']
    assert all(m.training == mode for m, mode in modes)
    # Matched extra-supervision control: no style hooks, ordinary labeled CE,
    # with the same whole-extra-forward random-state preservation.
    replay_config = configs['L_REPLAY']
    assert replay_config['mixstyle_l'] and not replay_config['style_levels']
    with s.labeled_style(model, labels, rx, days, ids, step, replay_config, 1.) as replay:
        replay_logits = model(x)
    same_rng(before)
    assert not replay['levels'] and replay['actual_hook_order'] == [] and replay['changed_packets'] == 0
    torch.testing.assert_close(replay_logits.detach(), baseline, rtol=0, atol=0)
    assert sum(len(m._forward_hooks) for m in model.modules()) == start_hooks
    # Exception after a real styled forward restores hooks, RNG, and mode.
    try:
        with s.labeled_style(model, labels, rx, days, ids, step, configs['STYLE_BOTH'], 1.):
            model(x)
            raise RuntimeError('intentional forward-context failure')
    except RuntimeError as error:
        assert str(error) == 'intentional forward-context failure'
    same_rng(before)
    assert sum(len(m._forward_hooks) for m in model.modules()) == start_hooks
    assert all(m.training == mode for m, mode in modes)
    # Hook-side failure also cleans up; a subsequent valid call still succeeds.
    try:
        with s.labeled_style(model, labels, rx, days, ids, step, configs['STYLE_BOTH'], 1.):
            model(x[:2])
    except ValueError as error:
        assert 'batch' in str(error)
    else:
        raise AssertionError('Mismatched actual forward batch accepted')
    same_rng(before)
    assert sum(len(m._forward_hooks) for m in model.modules()) == start_hooks
    with s.labeled_style(model, labels, rx, days, ids, step, configs['STYLE_BOTH'], 1.):
        model(x)
    same_rng(before)
    invalid = []
    for kind in ('U_role', 'bad_label', 'bad_RX', 'bad_day', 'short_metadata', 'PID_conflict', 'unknown_layer', 'duplicate_layer'):
        yy, rr, dd, pp = labels.clone(), rx.clone(), days.clone(), list(ids)
        config = dict(configs['STYLE_BOTH'])
        if kind == 'U_role': config['source_role'] = 'U_s'
        elif kind == 'bad_label': yy[0] = 6
        elif kind == 'bad_RX': rr[0] = 0
        elif kind == 'bad_day': dd[0] = 0
        elif kind == 'short_metadata': dd = dd[:-1]
        elif kind == 'PID_conflict': pp[10] = pp[0]
        elif kind == 'unknown_layer': config['style_levels'] = ['final']
        else: config['style_levels'] = ['time_down', 'time_down']
        try:
            with s.labeled_style(model, yy, rr, dd, pp, step, config, 1.):
                raise AssertionError('Invalid L style input accepted')
        except ValueError:
            invalid.append(kind)
    model.eval()
    with torch.no_grad():
        eval_plain = model(x)
        with s.labeled_style(model, labels, rx, days, ids, step, configs['STYLE_BOTH'], 1.) as off:
            eval_styled = model(x)
    assert not off['enabled'] and not off['levels']
    torch.testing.assert_close(eval_plain, eval_styled, rtol=0, atol=0)
    assert sum(len(m._forward_hooks) for m in model.modules()) == start_hooks
    json.dumps([routing, detail, ramped, zero, off], allow_nan=False)
    return dict(status='PASS', device=device, target_access=False, real_checkpoint_loaded=False,
                synthetic_L_only=True, style_only_without_heads=True, selected_native_hooks=routing,
                duplicate_occurrences_accepted=True, same_PID_donor_forbidden=True,
                normal_training_dropout_preserved=True, complete_extra_forward_RNG_restored=True,
                ramp_strength_applied=True, outside_context_output_bitwise_unchanged=True,
                L_REPLAY_no_hooks_output_unchanged_RNG_preserved=True,
                eval_context_bitwise_unchanged=True, input_validation=invalid,
                exception_and_hook_failure_cleanup=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2, allow_nan=False))
