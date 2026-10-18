"""Two focused CPU tensor checks; no real CVS, source IQ, CP or channel."""
import json
import time

import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_norm = nn.LayerNorm(160); self.pa_norm = nn.LayerNorm(160)
        self.gain = nn.Parameter(torch.linspace(-.5, .6, 160))
        self.weight = nn.Parameter(torch.randn(6, 160)); self.scale = 30.
        self.fail = False

    def classify(self, joint):
        if self.fail: raise RuntimeError('original-head-failure')
        return self.scale * F.linear(F.normalize(joint, dim=1, eps=1e-4),
            F.normalize(self.weight, dim=1, eps=1e-4))


class Sequence(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_norm = nn.LayerNorm(160)
        self.gain_raw = nn.Parameter(torch.linspace(.1, .7, 160))
        self.modes = ('SIG_TIME',) * 3
        self.views = nn.ModuleList([nn.Identity() for _ in range(3)])

    def forward(self, x):
        return .2 * self.gain_raw.tanh() * self.output_norm(x)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = nn.Module(); encoder.core = nn.Module(); core = nn.Module()
        encoder.core.id_backbone = core; core.cls_head = Head()
        encoder.response_projection = nn.Sequential(nn.Linear(160, 160), nn.LayerNorm(160))
        encoder.response_gain = nn.Parameter(torch.linspace(-.4, .3, 160))
        encoder.sequence_pool = Sequence()
        self.id_backbone = nn.Module(); self.id_backbone.encoder = encoder
        self.register_buffer('sentinel', torch.arange(3.))
        self.bypass_signature = False; self.wrong_joint = False

    def __deepcopy__(self, memo):
        raise AssertionError('Capture must not deepcopy the model')

    def forward_identity_only(self, x):
        ref = self.id_backbone.encoder; head = ref.core.id_backbone.cls_head
        values = x.flatten(1)
        base = head.base_norm(values[:, :160].sin() + values[:, 160:320])
        pa = head.pa_norm(values[:, 160:320].cos() + 1.25 * values[:, 320:480])
        response = ref.response_projection(values[:, 160:320])
        signature = ref.sequence_pool(values[:, 320:480]) if not self.bypass_signature else torch.zeros_like(base)
        joint = base + head.gain.tanh() * pa
        joint = joint + ref.response_gain.tanh() * response
        joint = joint + signature
        if self.wrong_joint: joint = joint + .2
        return dict(tx_logits=head.classify(joint), id_feat_joint=joint)


def run():
    from . import capture
    start = time.perf_counter(); passed = []
    torch.manual_seed(74)
    m = Toy().eval().requires_grad_(False)
    x = torch.arange(3 * 2 * 256, dtype=torch.float32).reshape(3, 2, 256) / 470
    before = {k: v.clone() for k, v in m.state_dict().items()}
    head = m.id_backbone.encoder.core.id_backbone.cls_head
    original_forward = head.classify
    full = capture.observe(m, x)
    nonoise = capture.observe(m, x * .7 + .12)
    clean = capture.observe(m, x.flip(2) * .8 - .2)
    assert full['contributions'].keys() == {'TF', 'PA', 'REF', 'SIG'}
    assert torch.equal(full['joint'], sum(full['contributions'].values()))
    assert full['parity_head_examples'] == full['head_examples'] == len(x)
    assert full['native_model_examples'] == len(x)
    result = capture.interventions(m, full, nonoise, clean)
    logits = result['logits']
    assert len(logits) == 12 and result['head_examples'] == 12 * len(x)
    for branch in capture.BRANCHES:
        for operation, replacement in [('zero', torch.zeros_like(full['joint'])),
                ('swap_nonoise', nonoise['contributions'][branch]),
                ('swap_clean', clean['contributions'][branch])]:
            expected_parts = dict(full['contributions']); expected_parts[branch] = replacement
            expected = original_forward(sum(expected_parts.values()))
            assert torch.allclose(logits[operation + '_' + branch], expected, atol=1e-6, rtol=1e-6)
    # Final unit normalization couples all contributions, so head-logit subtraction
    # and averaging two head predictions are not the required intervention.
    wrong_subtraction = full['logits'] - original_forward(full['contributions']['PA'])
    wrong_average = .5 * (full['logits'] + clean['logits'])
    assert not torch.allclose(logits['zero_PA'], wrong_subtraction, atol=1e-4)
    assert not torch.allclose(logits['swap_clean_PA'], wrong_average, atol=1e-4)
    assert all(not value.requires_grad for pack in (full, nonoise, clean)
        for value in (pack['logits'], pack['joint'], *pack['contributions'].values()))
    passed.append('four_postnorm_gain_contributions_native_joint_logits_and_nonlinear_head_interventions')

    def reject(fn):
        try: fn()
        except ValueError: pass
        else: raise AssertionError('Invalid capture/intervention accepted')
    for bad in (x[:, :, :-1], x.double(), torch.empty(0, 2, 256), x * float('nan')):
        reject(lambda bad=bad: capture.observe(m, bad))
    m.train(); reject(lambda: capture.observe(m, x)); m.eval()
    short = capture.observe(m, x[:2])
    reject(lambda: capture.interventions(m, full, short, clean))
    different = Toy().eval().requires_grad_(False)
    reject(lambda: capture.interventions(different, full, nonoise, clean))
    m.bypass_signature = True; reject(lambda: capture.observe(m, x)); m.bypass_signature = False
    m.wrong_joint = True; reject(lambda: capture.observe(m, x)); m.wrong_joint = False
    # Existing instance overrides must be restored by identity on success/failure.
    head.classify = original_forward
    capture.observe(m, x); assert head.classify is original_forward
    head.fail = True
    try: capture.observe(m, x)
    except RuntimeError as error: assert str(error) == 'original-head-failure'
    else: raise AssertionError('Original failure hidden')
    head.fail = False; assert head.classify is original_forward
    del head.classify
    assert 'classify' not in vars(head)
    assert all(not module._forward_hooks for module in m.modules())
    assert all(torch.equal(value, m.state_dict()[key]) for key, value in before.items())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in m.parameters())
    passed.append('actual_batch_shape_negatives_exception_method_hook_restore_no_deepcopy_or_state_gradient_mutation')
    return dict(status='PASS', checks=len(passed), passed=passed,
        wall_seconds=time.perf_counter() - start, real_CVS_NN=0, IQ_reads=0,
        checkpoint_reads=0, actual_channel_calls=0)


if __name__ == '__main__':
    print(json.dumps(run(), allow_nan=False))
