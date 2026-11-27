import pytest
import torch

from experiments.pamf_lora_main import runner
from experiments.pamf_lora_main.w24_packet_dura import NativePacketDura
from experiments.pamf_lora_main.w24_practical_supervision import PracticalLoss, matched_direction, regional_power
from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_ARMS, PRACTICAL_KIND, fit_inputs, trial


def test_equal_dose_uses_exact_complex_direction_without_increasing_zero_noise():
    ideal = torch.tensor([[[2+2j, 3-4j]], [[0j, 0j]]])
    teacher = torch.tensor([[[2j, -1j]], [[1j, 2j]]])
    edit, scale = matched_direction(ideal, teacher)
    torch.testing.assert_close(regional_power(edit)[:1], regional_power(teacher)[:1])
    assert torch.equal(edit[1], ideal[1]) and scale[1].item() == 0.
    torch.testing.assert_close(edit, ideal * scale)


@pytest.mark.parametrize('matched', [False, True])
def test_training_targets_reach_native_output_and_full_only_scope(matched):
    spec = trial(PRACTICAL_KIND)
    assert spec.routes == ('full_native',) and spec.fit_arms == PRACTICAL_ARMS
    runtime = dict(source_root='source', run_root='run', practical_cache='cache', dose_parent='parent', native_target_root='private',
                   classifier_root='private-f')
    arm = PRACTICAL_ARMS[0 if matched else 1]
    config = fit_inputs(runtime, arm)
    assert set(config) == {'source_root','run_root','practical_cache'} | ({'dose_parent'} if matched else set())
    model = NativePacketDura()
    teacher = NativePacketDura().eval().requires_grad_(False) if matched else None
    clean = torch.randn(4, 8192, dtype=torch.complex64)
    signal = clean * (.95+.05j)
    received = signal + .03*torch.randn_like(signal)
    raw = torch.stack((clean, received, signal), dim=1)
    loss, metrics = PracticalLoss('cpu', teacher)(model, None, raw, None, with_identity=True, recipe=None, step=1)
    assert metrics.shape == (6,) and torch.isfinite(metrics).all()
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    if teacher:
        assert all(p.grad is None for p in teacher.parameters())


def test_training_cap_counts_existing_trainers_and_unknown_contexts(tmp_path, monkeypatch):
    for pid, command in [('2', b'matlab\0allcs_parallel_validate('),
                         ('3', b'matlab\0allcs_parallel_train(\0/attempts/validate_mixed')]:
        proc = tmp_path / pid
        proc.mkdir()
        (proc/'comm').write_text('MATLAB\n')
        (proc/'cmdline').write_bytes(command)
    monkeypatch.setattr(runner, 'Path', lambda _: tmp_path)
    responses = {
        '--query-compute-apps=gpu_uuid,pid': 'lane0, 1\nlane0, 2\nlane1, 1\nlane1, 3\n',
        '--query-gpu=index,uuid,memory.free': '0, lane0, 16000\n1, lane1, 16000\n',
    }
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda args, **kwargs: responses[args[1]])
    assert runner.available([0, 1]) == []
    assert runner.available([0, 1], training_cap=True) == [0]
