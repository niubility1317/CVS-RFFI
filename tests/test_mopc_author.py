"""Synthetic/no-query checks for the explicit author-network input adaptation."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_phase2_practical.mopc_author import (
    BasicBlock,
    ConvBlockAttention,
    MoPCAuthorWiSig256,
    iq_to_author_layout,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_single_observation_layout_keeps_each_iq_sample_and_zero_gradient_finite():
    iq = torch.tensor([[[3., 0., -5.], [4., 0., 12.]]], requires_grad=True)
    before = iq.detach().clone()
    layout = iq_to_author_layout(iq)
    assert layout.shape == (1, 3, 1, 3)
    torch.testing.assert_close(layout[:, :2, 0], iq, rtol=0, atol=0)
    torch.testing.assert_close(layout[:, 2, 0], torch.tensor([[5., 0., 13.]]), rtol=0, atol=0)
    layout.sum().backward()
    assert torch.isfinite(iq.grad).all()
    torch.testing.assert_close(iq.detach(), before, rtol=0, atol=0)


def test_author_head_and_all_ten_residual_attention_blocks_receive_256_point_gradients():
    torch.manual_seed(71)
    model = MoPCAuthorWiSig256()
    blocks = [m for m in model.modules() if isinstance(m, BasicBlock)]
    attention = [m for m in model.modules() if isinstance(m, ConvBlockAttention)]
    assert len(blocks) == len(attention) == 10
    assert [m.conv2.out_channels for m in blocks] == [8, 8, 32, 32, 64, 64, 128, 128, 256, 256]
    assert [m.conv2.kernel_size for m in blocks] == [(8,), (8,), (8,), (8,), (4,), (4,), (4,), (4,), (2,), (2,)]
    assert model.fc.in_features == model.out_dim == 256
    assert model.fc.out_features == 6
    assert isinstance(model.avg_pooling, nn.AdaptiveAvgPool2d)
    assert model.avg_pooling.output_size == (1, 256)
    recorded = []
    def capture(module, inputs, output):
        output.retain_grad()
        recorded.append((module, output))
    watched = [model.conv1_, model.conv1, model.avg_pooling]
    watched += [m for block in blocks for m in (block.conv1, block.conv2, block.attention.ChannelGate,
                                               block.attention.SpatialGate)]
    handles = [m.register_forward_hook(capture) for m in watched]
    x = torch.randn(3, 2, 256, requires_grad=True)
    try:
        result = model(x)
        assert result['features'].shape == (3, 256)
        assert result['tx_logits'].shape == (3, 6)
        F.cross_entropy(result['tx_logits'], torch.tensor([0, 3, 5])).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert len(recorded) == len(watched)
    assert recorded[0][1].shape == (3, 1, 1, 256)
    assert recorded[1][1].shape == (3, 2, 250)
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)
    for module, output in recorded:
        assert output.grad is not None and torch.isfinite(output.grad).all(), type(module)
        assert torch.count_nonzero(output.grad), type(module)
        parameters = list(module.parameters())
        if parameters:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            assert any(torch.count_nonzero(p.grad) for p in parameters), type(module)


def test_eval_is_per_record_and_classifier_is_the_native_linear_head():
    torch.manual_seed(19)
    model = MoPCAuthorWiSig256().eval()
    iq = torch.randn(4, 2, 256)
    with torch.no_grad():
        result = model(iq)
        singles = torch.cat([model(iq[i:i + 1])['tx_logits'] for i in range(len(iq))])
        altered = iq.clone()
        altered[1:] *= 100
        torch.testing.assert_close(result['features'], model.features(iq))
        torch.testing.assert_close(result['tx_logits'], model.classify_features(result['features']))
        torch.testing.assert_close(result['tx_logits'], model.fc(result['features']), rtol=0, atol=0)
        torch.testing.assert_close(result['tx_logits'], singles, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(result['tx_logits'][:1], model(altered)['tx_logits'][:1], rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(2, 512), (2, 4, 256), (2, 2, 2, 256), (2, 2, 18)])
def test_invalid_layout_cannot_concatenate_observations_into_author_input(shape):
    with pytest.raises(ValueError, match='IQ'):
        MoPCAuthorWiSig256().features(torch.zeros(shape))


def _reference_directory():
    configured = os.environ.get('MOPC_AUTHOR_REFERENCE_DIR')
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'github_publish/CVS-RFFI-repo/local_artifacts/external_refs/MoPC-HR'
        if (candidate / 'ads_resnet.py').is_file():
            return candidate
    pytest.skip('Set MOPC_AUTHOR_REFERENCE_DIR to the unmodified author checkout for parity verification')


def _author_reference(monkeypatch):
    directory = _reference_directory()
    spec = importlib.util.spec_from_file_location('ConvBlockAttention', directory / 'ConvBlockAttention.py')
    attention = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'ConvBlockAttention', attention)
    spec.loader.exec_module(attention)
    path = directory / 'ads_resnet.py'
    parsed = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    # resnet101 is an unused import; removing it avoids a torchvision dependency
    # without modifying any executed author-network operation or parameter.
    parsed.body = [node for node in parsed.body
                   if not (isinstance(node, ast.ImportFrom) and node.module == 'torchvision.models')]
    module = types.ModuleType('_mopc_author_reference')
    exec(compile(parsed, str(path), 'exec'), module.__dict__)
    return module.resnet1d_ads()


def test_1024_legal_author_layout_matches_unmodified_residual_attention_forward(monkeypatch):
    torch.manual_seed(107)
    adapted = MoPCAuthorWiSig256().eval()
    reference = _author_reference(monkeypatch).eval()
    # The declared old-class-count adaptation is the only parameter-shape change.
    reference.fc = nn.Linear(256, 6)
    assert set(reference.state_dict()) == set(adapted.state_dict())
    reference.load_state_dict(adapted.state_dict(), strict=True)
    iq = torch.randn(2, 2, 1024)
    layout = iq_to_author_layout(iq)
    assert layout.shape == (2, 3, 1, 1024)
    with torch.no_grad():
        expected = reference.feature_extract_stage2(layout)
        actual = adapted(iq)
        torch.testing.assert_close(actual['features'], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual['tx_logits'], reference(layout), rtol=0, atol=0)


def test_native_registration_uses_256_dimensional_prototypes_and_expands_the_same_fc():
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite import registration as cil
    torch.manual_seed(8)
    model = MoPCAuthorWiSig256().eval()
    metadata = cil.build_registration_base_metadata(
        'mopc_hr', model, torch.randn(12, 2, 256), torch.arange(6).repeat_interleave(2))
    assert metadata['old_prototypes'].shape == (6, 256)
    anchor = cil.RegistrationState('mopc_hr', model, tuple(range(6)), 6,
                                   prototypes=metadata['old_prototypes'], base_metadata=metadata)
    old_weights = model.fc.weight.detach().clone()
    support = torch.randn(4, 2, 256)
    result = cil.register_new_classes(anchor, support, torch.tensor([6, 6, 7, 7]), (6, 7),
        config={'mopc_epochs': 1, 'mopc_batch_size': 4, 'mopc_stage_size': 5,
                'mopc_semantics': 'public_code'})
    assert model.fc.out_features == 6
    torch.testing.assert_close(model.fc.weight, old_weights, rtol=0, atol=0)
    assert result.model.fc.in_features == 256 and result.model.fc.out_features == 8
    assert result.prototypes.shape == (8, 256)
    assert result.score(support[:1]).shape == (1, 8)
    assert torch.isfinite(result.score(support)).all()
