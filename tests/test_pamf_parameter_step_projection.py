import math
import pytest
import torch
from torch import nn
from experiments.pamf_ssl_research.parameter_step_projection import project_parameters


def run_case(base, joint, full, clean, physical):
    baseline = (torch.tensor(base, dtype=torch.float32),)
    joint = (torch.tensor(joint, dtype=torch.float32),)
    named = [('backbone.weight', nn.Parameter(baseline[0].clone()))]
    anchors = [tuple([torch.tensor(v, dtype=torch.float32)]) for v in (full, clean, physical)]
    before = baseline[0].clone()
    result, details = project_parameters(named, baseline, joint, *anchors)
    assert torch.equal(baseline[0], before)
    delta = baseline[0].double() - result[0].double()
    raw = baseline[0].double() - joint[0].double()
    assert delta.norm() <= raw.norm() * (1 + 1e-6) + 1e-12
    for anchor in anchors:
        a = anchor[0].double()
        if a.square().sum() > 1e-12:
            assert a @ delta >= -1e-6 * max(float(a.norm() * delta.norm()), 1e-12)
    return result, details['other'], delta, raw


def test_actual_fp32_parameter_difference_is_protected_with_high_retention():
    _, info, delta, raw = run_case([.25, -.5], [.2502, -.5], [1, 100], [1, 100], [1, 100])
    assert float(torch.tensor([1., 100.]).double() @ raw) < 0
    assert delta.norm() / raw.norm() > .9
    assert not info['quantization_fallback']


def test_already_feasible_joint_parameters_are_unchanged():
    result, info, _, _ = run_case([.25, -.5], [.249, -.501], [1, 1], [1, 0], [0, 1])
    assert torch.equal(result[0], torch.tensor([.249, -.501]))
    assert info['passthrough']


def test_opposing_component_anchors_allow_nonzero_tangent_motion():
    _, info, delta, _ = run_case([0, 0], [1, -1], [0, 1], [1, 0], [-1, 0])
    torch.testing.assert_close(delta, torch.tensor([0., 1.], dtype=torch.float64), rtol=0, atol=1e-12)
    assert not info['quantization_fallback']


@pytest.mark.parametrize('epsilon', [1e-4, 1e-6, 1e-8])
def test_nearly_dependent_anchors_meet_actual_parameter_constraints(epsilon):
    run_case([0, 0, 0], [2e-4, 2e-4, epsilon * 1e-4],
             [1, 1, epsilon], [1, 0, 0], [0, 1, 0])


def test_zero_full_l_anchor_retains_exact_l_only_parameters():
    result, info, _, _ = run_case([.25, -.5], [.249, -.501], [0, 0], [1, 0], [0, 1])
    assert torch.equal(result[0], torch.tensor([.25, -.5]))
    assert info['tiny_L']


def test_infeasible_quantized_tangent_is_explicitly_counted_as_fallback():
    a = [1, math.sqrt(2)]
    result, info, _, _ = run_case([1., 1.], [1. - 2e-5, 1. + 1e-5], [1, 0], a, [-x for x in a])
    assert info['quantization_fallback']
    assert torch.equal(result[0], torch.ones(2))


@pytest.mark.parametrize('bad', ['nonfinite', 'dtype', 'length'])
def test_rejects_invalid_step_vectors(bad):
    p = nn.Parameter(torch.zeros(2))
    anchors = ([torch.ones(2)],) * 3
    baseline, joint = [p.detach().clone()], [torch.ones(2)]
    if bad == 'nonfinite':
        joint[0][0] = float('nan')
    elif bad == 'dtype':
        joint[0] = joint[0].double()
    else:
        joint = []
    with pytest.raises(ValueError):
        project_parameters([('body.weight', p)], baseline, joint, *anchors)
