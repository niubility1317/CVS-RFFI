import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_adapter import (  # noqa: E402
    ResidualMetaAdapter,
    adapter_parameter_budget,
    adapter_step_size_by_parameter,
    iter_inner_adapter_parameters,
)
from model import build_model  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _tiny_model(**kwargs):
    model_variant = kwargs.pop("model_variant", "lite_h")
    return build_model(
        num_classes=3,
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant=model_variant,
        **kwargs,
    )


def _tiny_dual_model(**kwargs):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        **kwargs,
    )


def test_builder_accepts_meta_adapter_rank_and_sites():
    model = _tiny_model(meta_adapter_rank=4, meta_adapter_sites="time")
    assert isinstance(model.meta_adapter_time, ResidualMetaAdapter)


def test_rank_zero_has_no_meta_adapter_state_keys_and_preserves_legacy_load():
    torch.manual_seed(7)
    legacy = _tiny_model()
    torch.manual_seed(7)
    rank_zero = _tiny_model(meta_adapter_rank=0)

    assert not any("meta_adapter" in key for key in rank_zero.state_dict())
    rank_zero.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(2, 2, 64)
    legacy.eval()
    rank_zero.eval()
    with torch.no_grad():
        assert torch.allclose(legacy(x), rank_zero(x), atol=1e-6, rtol=1e-6)


def test_rank_four_tri_sites_create_exactly_three_adapters():
    model = _tiny_model(meta_adapter_rank=4, meta_adapter_sites="time,freq,fusion")
    sites = {
        name
        for name, module in model.named_modules()
        if isinstance(module, ResidualMetaAdapter)
    }
    assert sites == {"meta_adapter_time", "meta_adapter_freq", "meta_adapter_fusion"}
    assert all("meta_adapter" in key for key in model.state_dict() if "meta_adapter" in key)


def test_adapter_is_shape_preserving_and_near_identity_at_step_zero():
    torch.manual_seed(11)
    adapter = ResidualMetaAdapter(dim=8, rank=4)
    x = torch.randn(3, 5, 8)
    y = adapter(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-3, rtol=1e-3)


def test_inner_parameter_iterator_excludes_step_size_and_non_adapter_parameters():
    model = _tiny_model(meta_adapter_rank=4, meta_adapter_sites="time,freq,fusion")
    items = list(iter_inner_adapter_parameters(model))
    names = [name for name, _ in items]
    assert len(items) == 15
    assert all(name.startswith("meta_adapter_") for name in names)
    assert all(name.endswith(("down.weight", "down.bias", "up.weight", "up.bias", "gate")) for name in names)
    assert all("log_step_size" not in name for name in names)
    assert all("cls_head" not in name and "fuse" not in name for name in names)


def test_step_size_mapping_shares_differentiable_site_step_tensor():
    model = _tiny_model(meta_adapter_rank=4, meta_adapter_sites="time,freq,fusion")
    mapping = adapter_step_size_by_parameter(model)
    names = [name for name, _ in iter_inner_adapter_parameters(model)]
    assert set(mapping) == set(names)
    by_site = {}
    for name in names:
        site = name.rsplit(".", 2)[0]
        by_site.setdefault(site, []).append(mapping[name])
    for site, values in by_site.items():
        assert all(value is values[0] for value in values)
        assert values[0].requires_grad
        assert torch.allclose(values[0], getattr(model, site).step_size())


def test_parameter_budget_uses_real_model_total_and_inner_ratio_is_below_one_percent():
    model = _tiny_model(
        model_variant="base",
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
    )
    budget = adapter_parameter_budget(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    inner = sum(parameter.numel() for _, parameter in iter_inner_adapter_parameters(model))
    assert budget["total_parameters"] == total
    assert budget["inner_parameters"] == inner
    assert budget["adapter_parameters"] >= inner
    assert budget["inner_ratio"] == pytest.approx(inner / total)
    assert budget["inner_ratio"] <= 0.01


def test_dual_builder_transmits_meta_adapter_options_to_both_adv3b02_backbones():
    model = _tiny_dual_model(meta_adapter_rank=4, meta_adapter_sites="time,freq,fusion")
    for backbone in (model.id_backbone, model.dom_backbone):
        sites = {
            name
            for name, module in backbone.named_modules()
            if isinstance(module, ResidualMetaAdapter)
        }
        assert sites == {"meta_adapter_time", "meta_adapter_freq", "meta_adapter_fusion"}


def test_dual_builder_default_remains_legacy_state_and_forward_compatible():
    torch.manual_seed(19)
    legacy = _tiny_dual_model()
    torch.manual_seed(19)
    explicit_zero = _tiny_dual_model(meta_adapter_rank=0, meta_adapter_sites="")
    assert not any("meta_adapter" in key for key in explicit_zero.state_dict())
    explicit_zero.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(2, 2, 64)
    legacy.eval()
    explicit_zero.eval()
    with torch.no_grad():
        assert torch.allclose(legacy(x), explicit_zero(x), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"meta_adapter_rank": -1},
        {"meta_adapter_rank": 4, "meta_adapter_sites": "time,time"},
        {"meta_adapter_rank": 4, "meta_adapter_sites": "unknown"},
        {"meta_adapter_rank": 4, "meta_adapter_sites": ""},
    ],
)
def test_invalid_meta_adapter_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        _tiny_model(**kwargs)
