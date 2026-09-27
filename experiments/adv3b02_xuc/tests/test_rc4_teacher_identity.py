"""The independent RC4 teacher switch cannot change student or routing modes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.train_rc4_practical import native, build_args


ROOT = Path(__file__).resolve().parents[1]


def test_independent_rc4_teacher_flag_is_default_off():
    args = native.build_arg_parser().parse_args(['--output_dir', 'unused'])
    assert args.a1_rc4_teacher_identity_only is False
    args = native.build_arg_parser().parse_args(['--output_dir', 'unused', '--a1_rc4_teacher_identity_only', 'true'])
    assert args.a1_rc4_teacher_identity_only is True
    assert args.a1_runtime_fast is False
    assert args.a1_gradient_snapshot is False


@pytest.mark.parametrize('flag,is_ema,fast,identity', [
    (False, True, False, False), (True, True, False, True),
    (True, False, False, False), (False, False, True, True),
])
def test_dispatch_is_limited_to_explicit_ema_and_preserves_legacy_fast(flag, is_ema, fast, identity):
    class Teacher(torch.nn.Module):
        def forward(self, x, **kwargs):
            return {'kind': 'full', 'tx_logits': x, 'z_id': x}
        def forward_identity_only(self, x, **kwargs):
            return {'kind': 'identity', 'tx_logits': x, 'z_id': x}
    teacher = Teacher().eval()
    args = SimpleNamespace(a1_runtime_fast=fast, a1_rc4_teacher_identity_only=flag)
    result = native._forward_rc4_teacher(teacher, torch.ones(2, 3), domain_labels=None,
                                        grl_lambda=.5, args=args, is_ema_teacher=is_ema)
    assert result['kind'] == ('identity' if identity else 'full')
    assert vars(args) == dict(a1_runtime_fast=fast, a1_rc4_teacher_identity_only=flag)


@pytest.mark.parametrize('device,amp', [('cpu', False), ('cuda', False), ('cuda', True)])
def test_real_teacher_outputs_state_rng_and_amp(device, amp):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required to validate training AMP path')
    args = build_args(ROOT / 'configs/rc4_practical_full_noeq_20260918.json',
                      'unused', 'unused', 'unused', 'unused', 'unused', 'synthetic')
    args.a1_runtime_fast = False
    args.a1_rc4_teacher_identity_only = False
    torch.manual_seed(913)
    ma = native.merge_checkpoint_args({'model': None, 'args': {}, 'stats': {}, 'split_info': None},
                                     args, input_len=256, num_domains=15)
    model = native.build_baseline_model(native._apply_model_cli_args(ma, args), torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(8, 2, 256, device=device)
    domains = torch.arange(8, device=device) % 15
    state = deepcopy(model.state_dict())
    flags = [module.training for module in model.modules()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
        full = native._forward_rc4_teacher(model, x, domain_labels=domains, grl_lambda=.7,
                                          args=args, is_ema_teacher=True)
        args.a1_rc4_teacher_identity_only = True
        identity = native._forward_rc4_teacher(model, x, domain_labels=domains, grl_lambda=.7,
                                              args=args, is_ema_teacher=True)
    # Both modes run the identical identity backbone at the identical batch size.
    # Demand exact equality, stricter than an AMP tolerance that could alter a route.
    for key in ('z_id', 'tx_logits'):
        assert torch.isfinite(full[key]).all() and torch.isfinite(identity[key]).all()
        torch.testing.assert_close(full[key], identity[key], rtol=0, atol=0)
        assert not identity[key].requires_grad
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert flags == [module.training for module in model.modules()]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))


def test_enabled_path_rejects_training_mode():
    class Teacher(torch.nn.Module):
        def forward_identity_only(self, x, **kwargs):
            return {'z_id': x, 'tx_logits': x}
    with pytest.raises(ValueError, match='eval mode'):
        native._forward_rc4_teacher(Teacher(), torch.ones(2, 3), domain_labels=None,
            grl_lambda=1., args=SimpleNamespace(a1_runtime_fast=False, a1_rc4_teacher_identity_only=True),
            is_ema_teacher=True)
