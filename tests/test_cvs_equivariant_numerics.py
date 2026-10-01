import pytest
import torch
from experiments.cvs_equivariant_numerics.audit import precision_mode, stage_error, capture
from experiments.cvs_equivariant_identity.model import build, rotate_pair


def test_precision_toggle_restores_even_on_failure():
    original = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    with pytest.raises(RuntimeError):
        with precision_mode(torch, 'cudnn_tf32_off_fp32') as dtype:
            assert dtype == torch.float32
            assert not torch.backends.cudnn.allow_tf32
            assert torch.backends.cuda.matmul.allow_tf32 == matmul
            raise RuntimeError('intentional')
    assert torch.backends.cudnn.allow_tf32 == original


def test_stage_measurement_distinguishes_charge_one_and_invariant():
    x = torch.randn(3, 2, 4, 16)
    phase = .37
    rotated = rotate_pair(x, x.new_tensor(phase))
    assert stage_error('time.0', x, rotated, phase)['max_abs_error'] == 0
    assert stage_error('features', x, rotated, phase)['max_abs_error'] > .01
    assert stage_error('features', x, x, phase)['max_abs_error'] == 0


def test_capture_observes_actual_sinc_and_removes_hooks_without_changing_output():
    torch.set_num_threads(2)
    model = build('equivariant_memory').eval().requires_grad_(False)
    x = torch.randn(3, 2, 256)
    with torch.no_grad():
        baseline = model(x)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        trace = capture(model, x)
    assert torch.equal(trace['logits'], baseline)
    assert trace['id_backbone.sinc#0'].shape == (3, 48, 256)
    assert {'readout#0', 'readout#1', 'time.0.conv#0', 'behavior.2#0', 'features', 'unit_features', 'logits'} <= set(trace)
    assert not any(module._forward_hooks for module in model.modules())
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
