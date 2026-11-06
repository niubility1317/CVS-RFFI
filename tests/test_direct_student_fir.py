import torch
import numpy as np
import pytest

from experiments.phase1_dura_orthogonal.direct_student_fir import DirectStudentFIR, TAPS
from experiments.phase1_dura_orthogonal import train_source


def test_identity_at_initialization_and_frozen_student():
    model = DirectStudentFIR(width=8).train()
    received = torch.randn(4, 2, 256)
    with torch.no_grad():
        base = model.student(received)
    output = model.refine(received, base)
    assert torch.equal(output, base)
    output.square().mean().backward()
    assert model.tap_head.weight.grad is not None
    assert torch.count_nonzero(model.tap_head.weight.grad)
    assert not model.student.training
    assert all(not p.requires_grad and p.grad is None for p in model.student.parameters())


def test_complex_fir_residual_and_received_only_inference():
    model = DirectStudentFIR(width=8).eval()
    with torch.no_grad():
        model.tap_head.bias[TAPS + TAPS//2] = .2
    received = torch.randn(2, 2, 256)
    with torch.no_grad():
        base = model.student(received)
        out = model(received, torch.randn_like(received), torch.randn_like(received), torch.ones(2))
    factor = .5*torch.tanh(torch.tensor(.2))
    torch.testing.assert_close(out[:, 0], base[:, 0] - factor*base[:, 1])
    torch.testing.assert_close(out[:, 1], base[:, 1] + factor*base[:, 0])


def test_parent_contract_rejects_reordered_or_target_contaminated_source(tmp_path, monkeypatch):
    monkeypatch.setattr(train_source, 'WIDE_FIT_SIZE', 4)
    folder = tmp_path/'runs'/train_source.FIR_SOURCE
    folder.mkdir(parents=True)
    metadata = dict(commit=train_source.FIR_SOURCE_COMMIT, architecture='direct_spectral_student',
                    objective='wide_source_distill', epochs=train_source.DUAL_DOMAIN_EPOCHS,
                    DURA_training_reads=1, source_iq_teacher='dura_v1_R65_E20_4940',
                    source_iq_teacher_weight=.5, inference_received_iq_only=True,
                    TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                    target_reads=0, DURA_frozen=True, source_fit_size=4)
    train_source.core.write(folder/'source_summary.json', dict(status='SOURCE_TRAINED', fit_ids=4, **metadata))
    state = dict(schema=train_source.SCHEMA, fit_ids=np.array([1, 2, 3, 4]), model={}, **metadata)
    torch.save(state, folder/'final.pt')
    assert train_source.load_fir_parent(tmp_path, np.array([1, 2, 3, 4]), np.array([9]), False) == {}
    with pytest.raises(ValueError):
        train_source.load_fir_parent(tmp_path, np.array([2, 1, 3, 4]), np.array([9]), False)
    state['target_reads'] = 1
    torch.save(state, folder/'final.pt')
    with pytest.raises(ValueError):
        train_source.load_fir_parent(tmp_path, np.array([1, 2, 3, 4]), np.array([9]), False)
