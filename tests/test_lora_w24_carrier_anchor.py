from types import SimpleNamespace

import torch

from experiments.pamf_lora_main.w24_carrier_anchor import final_map
from experiments.pamf_lora_main.w24_third_prior import CARRIER_ANCHOR_ARMS, fit_inputs, trial


def test_final_map_uses_affine_likelihood_with_original_normal_operator():
    received = torch.tensor([[[.9 + .2j, 1.1 - .2j, .7 + .6j, 1.3 - .6j]]])
    current = received + .02j
    variance = torch.tensor([[[.2, .3, .5, .7]]])
    gain = torch.tensor([[[1.1 + .3j, .8j, .7 - .1j, 1.3 + .2j]]])
    image = .25j

    class Action:
        def __call__(self, x):
            return gain * x + image * x.conj()

        def adjoint(self, x):
            return gain.conj() * x + image * x.conj()

    action = Action()
    carrier = received.mean(-1, keepdim=True)

    def map_two(operator, observed, initial, prior, noise, gradient):
        assert operator is action and initial is current and prior is received and noise is variance
        expected_error = (received - carrier) - action(current - carrier)
        torch.testing.assert_close(observed - action(current), expected_error)
        torch.testing.assert_close(gradient, action.adjoint(expected_error / variance))
        return gradient

    final_map(SimpleNamespace(map_two=map_two), action, received, current, received,
              variance, received - action(current), None)


def test_carrier_anchor_trial_accepts_only_source_fit_paths_and_runs_full():
    spec = trial('carrier-anchor')
    assert spec.steps == 1800 and spec.routes == ('full_native',)
    assert spec.arms == CARRIER_ANCHOR_ARMS
    assert spec.source_modes[2:4] == ('carrier_linear_half', 'carrier_affine_half')
    for arm in spec.arms:
        assert fit_inputs(dict(source_root='source', source_parent='parent', run_root='run',
                               classifier_root='F', native_target_root='target'), arm) == {
            'source_root': 'source', 'source_parent': 'parent', 'run_root': f'run/{arm}',
        }
