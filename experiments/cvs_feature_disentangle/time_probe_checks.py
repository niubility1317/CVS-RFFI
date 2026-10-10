"""Regression for the native dual wrapper used by post-training source V."""
import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from . import design as d
from .runtime import installed
from .validation import _equal, evaluate_source_stress, source_readonly, time_contribution_off


def run(device='cpu', full_v=False):
    torch.set_num_threads(2)
    c = d.config(d.rows()[0])
    args = d.make_args(c, device)
    args.num_domains = 5
    args.input_len = 256
    with installed(c, training=False) as native:
        model = native.build_baseline_model(args, torch.device(device))
    old_points = [name for name, _ in model.named_modules() if name.split('.')[-1] == 't_proj']
    assert old_points == ['id_backbone.encoder.core.id_backbone.t_proj', 'dom_backbone.t_proj']
    # This is the failed production precondition, on its actual factory output.
    assert len(old_points) != 1
    expected = old_points[0]
    model.train()
    before = copy.deepcopy(model.state_dict())
    modes = [module.training for module in model.modules()]
    initial_hooks = {name: len(module._forward_hooks) for name, module in model.named_modules()}
    x = torch.randn(6, 2, 256, generator=torch.Generator().manual_seed(3701)).to(device)
    captures, handles = {}, []
    for name, module in model.named_modules():
        if name.startswith('id_backbone.') and name.split('.')[-1] in ('f_proj', 'pa_proj', 'response'):
            captures[name] = []
            handles.append(module.register_forward_hook(
                lambda module, inputs, output, key=name: captures[key].append(output.detach().clone())))
    with source_readonly(model), torch.no_grad():
        full = model(x)
        with time_contribution_off(model) as probe:
            off = model(x)
        restored = model(x)
        assert probe == dict(count=1, hook=expected)
        delta = float((full-off).abs().max())
        assert delta > 0 and torch.equal(full, restored)
        assert all(len(values) == 3 and torch.equal(values[0], values[1])
                   and torch.equal(values[0], values[2]) for values in captures.values())
        assert {name.split('.')[-1] for name in captures} == {'f_proj', 'pa_proj', 'response'}
        try:
            with time_contribution_off(model):
                raise RuntimeError('test diagnostic exception cleanup')
        except RuntimeError:
            pass
    for handle in handles:
        handle.remove()
    assert _equal(before, model.state_dict())
    assert modes == [module.training for module in model.modules()]
    assert initial_hooks == {name: len(module._forward_hooks) for name, module in model.named_modules()}
    # Keep fail-closed behavior for a malformed identity, even when the domain
    # subtree alone happens to provide a projection.
    malformed = nn.Module()
    malformed.id_backbone = nn.Identity()
    malformed.dom_backbone = nn.Module()
    malformed.dom_backbone.t_proj = nn.Linear(2, 160)
    try:
        with time_contribution_off(malformed):
            pass
    except ValueError as error:
        assert 'identity downstream' in str(error)
    else:
        raise AssertionError('Domain-only projection accepted as identity')

    def loader():
        generator = torch.Generator().manual_seed(7011)
        count = 27000 if full_v else 6
        for start in range(0, count, 256):
            size = min(256, count-start)
            indices = torch.arange(start, start+size)
            yield (torch.randn(size, 2, 256, generator=generator), indices % 6,
                   torch.zeros(size, dtype=torch.long),
                   dict(rx_i=torch.tensor([1, 3, 4, 6, 8])[indices % 5], day_i=indices % 3))

    with tempfile.TemporaryDirectory(prefix='feature-native-time-probe-') as temporary:
        if full_v:
            result = evaluate_source_stress(model, loader(), device, c, temporary)
            assert result['status'] == 'SOURCE_STRESS_COMPLETE'
            assert all(view['count'] == 27000 for view in result['views'].values())
            assert result['time_dependency_probe']['hook'] == expected
            assert (Path(temporary)/'source_stress.json').is_file()
        else:
            # The real validator executes all eight views and the probe; only
            # its strict full-V packet accounting rejects this small fixture.
            try:
                evaluate_source_stress(model, loader(), device, c, temporary)
            except ValueError as error:
                assert str(error) == 'Incomplete source V pressure validation'
            else:
                raise AssertionError('Incomplete fixture accepted as full source V')
            assert not (Path(temporary)/'source_stress.json').exists()
    assert _equal(before, model.state_dict())
    assert modes == [module.training for module in model.modules()]
    assert initial_hooks == {name: len(module._forward_hooks) for name, module in model.named_modules()}
    return dict(status='PASS', device=device, actual_model=type(model).__name__,
                old_global_probe_match_count=len(old_points), identity_probe_hook=expected,
                probe_calls_per_forward=probe['count'], max_logit_change=delta,
                restored_forward_exact=True, frequency_PA_reference_outputs_unchanged=True,
                all_state_modes_and_hooks_unchanged=True, exception_cleanup=True,
                missing_identity_rejected=True, source_views=8,
                full_V_fixture_count=27000 if full_v else None,
                source_validation_path='full 27000 synthetic packets' if full_v else 'all views then expected incomplete-V rejection',
                target_access=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--full-v', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.device, args.full_v)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
