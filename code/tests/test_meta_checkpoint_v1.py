from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_adapter import iter_inner_adapter_parameters  # noqa: E402
from cvsrffi.meta_checkpoint import (  # noqa: E402
    META_BUNDLE_SCHEMA,
    REQUIRED_META_BUNDLE_KEYS,
    load_legacy_base_for_meta,
    load_meta_bundle_strict,
    save_meta_bundle,
)
from model import build_model  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _model_args(*, variant: str = "base", adapters: bool = True) -> dict[str, object]:
    return {
        "num_classes": 3,
        "dataset": "wisig",
        "input_len": 64,
        "sample_rate_hz": 25e6,
        "model_variant": variant,
        "meta_adapter_rank": 4 if adapters else 0,
        "meta_adapter_sites": "time,freq,fusion" if adapters else "",
    }


def _model(*, variant: str = "base", adapters: bool = True):
    return build_model(**_model_args(variant=variant, adapters=adapters))


def _legacy_payload(*, container: str | None = "model") -> dict[str, object]:
    state = _model(adapters=False).state_dict()
    if container is None:
        return dict(state)
    return {container: dict(state)}


def _config(*, variant: str = "base") -> dict[str, object]:
    return {
        "model_args": _model_args(variant=variant),
        "meta_adapter_config": {
            "rank": 4,
            "sites": ["time", "freq", "fusion"],
            "phase2_steps": 3,
        },
        "base_checkpoint": {
            "id": "ADV3B02_CORE90_SOFT_E200:test",
            "role": "source_only",
        },
        "class_mapping": {"0": "tx_a", "1": "tx_b", "2": "tx_c"},
        "prototypes": {
            "0": torch.tensor([1.0, 0.0]),
            "1": torch.tensor([0.0, 1.0]),
            "2": torch.tensor([-1.0, 0.0]),
        },
    }


def _selection() -> dict[str, object]:
    return {
        "source_split": "V_select",
        "criterion": "max_min_source_holdout_delta",
        "seed": 7,
    }


def _save_valid_bundle(path: Path, *, variant: str = "base") -> dict[str, object]:
    model = _model(variant=variant)
    config = _config(variant=variant)
    save_meta_bundle(path, model, config, _selection())
    return config


def test_legacy_base_migration_allows_only_adapter_missing_and_keeps_adapter_init():
    model = _model()
    adapter_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if "meta_adapter_" in name
    }

    audit = load_legacy_base_for_meta(model, _legacy_payload())

    assert audit.checkpoint_load_strict is False
    assert audit.unexpected_keys == ()
    assert audit.missing_keys
    assert all("meta_adapter_" in name for name in audit.missing_keys)
    for name, value in adapter_before.items():
        assert torch.equal(model.state_dict()[name], value)


@pytest.mark.parametrize("container", [None, "model", "model_state_dict", "state_dict"])
def test_legacy_loader_supports_repository_state_dict_containers(container):
    model = _model()
    audit = load_legacy_base_for_meta(model, _legacy_payload(container=container))
    assert audit.unexpected_keys == ()
    assert all("meta_adapter_" in name for name in audit.missing_keys)


def test_legacy_loader_rejects_conflicting_containers_instead_of_guessing():
    first = _legacy_payload(container="model")
    second = copy.deepcopy(first["model"])
    key = next(iter(second))
    second[key] = second[key].clone()
    second[key].view(-1)[0] += 1.0
    first["state_dict"] = second

    with pytest.raises(ValueError, match="conflicting state dict containers"):
        load_legacy_base_for_meta(_model(), first)


def test_legacy_loader_rejects_unexpected_and_non_adapter_missing_keys():
    unexpected = _legacy_payload()
    unexpected["model"]["unexpected.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unexpected"):
        load_legacy_base_for_meta(_model(), unexpected)

    missing = _legacy_payload()
    missing["model"].pop("cls_head.head.weight")
    with pytest.raises(ValueError, match="missing"):
        load_legacy_base_for_meta(_model(), missing)


def test_legacy_loader_rejects_adapter_values_in_a_legacy_payload():
    payload = _legacy_payload()
    payload["model"]["meta_adapter_time.down.weight"] = torch.zeros(4, 64)
    with pytest.raises(ValueError, match="legacy payload must not contain adapter"):
        load_legacy_base_for_meta(_model(), payload)


def test_legacy_loader_rejects_missing_unauthorized_adapter_site():
    class UnauthorizedAdapterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 2)
            self.meta_adapter_fake = torch.nn.Linear(2, 2)

    model = UnauthorizedAdapterModel()
    payload = {
        "model": {
            "base.weight": model.base.weight.detach().clone(),
            "base.bias": model.base.bias.detach().clone(),
        }
    }
    with pytest.raises(ValueError, match="non-adapter"):
        load_legacy_base_for_meta(model, payload)


def test_save_bundle_writes_fixed_schema_cpu_state_and_never_overwrites(tmp_path):
    path = tmp_path / "meta_bundle.pth"
    model = _model()
    config = _config()
    save_meta_bundle(path, model, config, _selection())

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) == REQUIRED_META_BUNDLE_KEYS
    assert payload["schema"] == META_BUNDLE_SCHEMA
    assert all(value.device.type == "cpu" for value in payload["model_state"].values())
    assert payload["model_args"] == config["model_args"]
    assert payload["meta_adapter_config"] == config["meta_adapter_config"]
    assert payload["base_checkpoint"] == config["base_checkpoint"]
    assert payload["class_mapping"] == config["class_mapping"]
    for key, value in config["prototypes"].items():
        assert torch.equal(payload["prototypes"][key], value)

    with pytest.raises(FileExistsError):
        save_meta_bundle(path, model, config, _selection())


def test_fusion_only_bundle_saves_and_strictly_reloads_exact_adapter_profile(tmp_path):
    path = tmp_path / "fusion_only_meta_bundle.pth"
    model_args = _model_args()
    model_args["meta_adapter_sites"] = "fusion"
    model = build_model(**model_args)
    config = _config()
    config["model_args"] = model_args
    config["meta_adapter_config"]["sites"] = ["fusion"]

    save_meta_bundle(path, model, config, _selection())
    loaded, audit = load_meta_bundle_strict(path, "cpu")

    expected = tuple(sorted(name for name, _ in iter_inner_adapter_parameters(loaded)))
    assert expected
    assert all(name.startswith("meta_adapter_fusion.") for name in expected)
    assert not any("meta_adapter_time" in name or "meta_adapter_freq" in name for name in loaded.state_dict())
    assert audit.trainable_names == expected
    assert audit.trainable_fraction <= 0.01


@pytest.mark.parametrize(
    "sites",
    [["time"], ["freq"], ["time", "fusion"], ["freq", "fusion"]],
)
def test_bundle_rejects_unregistered_authorized_site_subsets(tmp_path, sites):
    config = _config()
    config["model_args"]["meta_adapter_sites"] = ",".join(sites)
    config["meta_adapter_config"]["sites"] = sites

    with pytest.raises(ValueError, match="registered adapter site profile"):
        save_meta_bundle(
            tmp_path / ("unregistered_" + "_".join(sites) + ".pth"),
            build_model(**config["model_args"]),
            config,
            _selection(),
        )


def test_save_bundle_rejects_target_query_selection_information(tmp_path):
    with pytest.raises(ValueError, match="Phase1 source selection"):
        save_meta_bundle(
            tmp_path / "forbidden.pth",
            _model(),
            _config(),
            {"target_query_accuracy": 0.99},
        )


@pytest.mark.parametrize("field", ["model_args", "meta_adapter_config", "selection", "base_checkpoint", "class_mapping", "prototypes"])
def test_save_bundle_rejects_nested_renamed_or_embedded_fields(tmp_path, field):
    config = _config()
    selection = _selection()
    if field == "model_args":
        config["model_args"]["num_classes_renamed"] = config["model_args"].pop("num_classes")
    elif field == "meta_adapter_config":
        config["meta_adapter_config"]["renamed_phase2_limit"] = 3
    elif field == "selection":
        selection["source_only"] = {"renamed_metric": 0.9}
    elif field == "base_checkpoint":
        config["base_checkpoint"]["source_samples"] = ["sample-id"]
    elif field == "class_mapping":
        config["class_mapping"]["0"] = {"source_samples": ["sample-id"]}
    elif field == "prototypes":
        config["prototypes"]["0"] = {"query_truth": [1]}

    with pytest.raises(ValueError):
        save_meta_bundle(tmp_path / f"invalid_{field}.pth", _model(), config, selection)


def test_save_bundle_validates_class_mapping_and_prototype_shapes(tmp_path):
    config = _config()
    config["class_mapping"] = {"0": "tx_a", "2": "tx_c"}
    with pytest.raises(ValueError, match="class_mapping"):
        save_meta_bundle(tmp_path / "gapped_classes.pth", _model(), config, _selection())

    config = _config()
    config["prototypes"]["1"] = torch.ones(3)
    with pytest.raises(ValueError, match="prototypes"):
        save_meta_bundle(tmp_path / "inconsistent_prototypes.pth", _model(), config, _selection())


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_field", "required fields"),
        ("schema", "schema"),
        ("extra_field", "top-level fields"),
        ("missing_state", "missing"),
        ("unexpected_state", "unexpected"),
        ("shape", "shape"),
    ],
)
def test_strict_loader_fails_closed_on_bundle_schema_or_state_drift(
    tmp_path, mutation, message
):
    source = tmp_path / "source.pth"
    _save_valid_bundle(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if mutation == "missing_field":
        payload.pop("selection")
    elif mutation == "schema":
        payload["schema"] = "wrong.schema"
    elif mutation == "extra_field":
        payload["extra"] = True
    elif mutation == "missing_state":
        payload["model_state"].pop(next(iter(payload["model_state"])))
    elif mutation == "unexpected_state":
        payload["model_state"]["unexpected.weight"] = torch.ones(1)
    elif mutation == "shape":
        key = next(iter(payload["model_state"]))
        payload["model_state"][key] = torch.zeros(1)
    broken = tmp_path / f"broken_{mutation}.pth"
    torch.save(payload, broken)

    with pytest.raises(ValueError, match=message):
        load_meta_bundle_strict(broken, "cpu")


@pytest.mark.parametrize("field", ["model_args", "meta_adapter_config", "selection", "base_checkpoint", "class_mapping", "prototypes"])
def test_strict_loader_rejects_nested_field_drift(tmp_path, field):
    source = tmp_path / "source_nested.pth"
    _save_valid_bundle(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if field == "model_args":
        payload[field]["num_classes_renamed"] = payload[field].pop("num_classes")
    elif field == "meta_adapter_config":
        payload[field]["renamed_phase2_limit"] = 3
    elif field == "selection":
        payload[field]["source_only"] = {"renamed_metric": 0.9}
    elif field == "base_checkpoint":
        payload[field]["source_samples"] = ["sample-id"]
    elif field == "class_mapping":
        payload[field]["0"] = {"source_samples": ["sample-id"]}
    elif field == "prototypes":
        payload[field]["0"] = {"query_truth": [1]}
    broken = tmp_path / f"broken_nested_{field}.pth"
    torch.save(payload, broken)

    with pytest.raises(ValueError):
        load_meta_bundle_strict(broken, "cpu")


def test_strict_loader_requests_weights_only_true(monkeypatch, tmp_path):
    path = tmp_path / "weights_only.pth"
    _save_valid_bundle(path)
    original_load = torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    load_meta_bundle_strict(path, "cpu")
    assert calls and calls[0].get("weights_only") is True


def test_strict_loader_fails_explicitly_when_weights_only_is_unsupported(monkeypatch, tmp_path):
    path = tmp_path / "unsupported_weights_only.pth"
    _save_valid_bundle(path)

    def unsupported_load(*args, **kwargs):
        if kwargs.get("weights_only") is True:
            raise TypeError("weights_only is unsupported")
        raise AssertionError("must not fall back to pickle loading")

    monkeypatch.setattr(torch, "load", unsupported_load)
    with pytest.raises(ValueError, match="weights_only"):
        load_meta_bundle_strict(path, "cpu")


def test_strict_load_rebuilds_true_model_with_exact_state_and_freeze_allowlist(tmp_path):
    path = tmp_path / "meta_bundle.pth"
    config = _save_valid_bundle(path)
    original = torch.load(path, map_location="cpu", weights_only=False)["model_state"]

    loaded, audit = load_meta_bundle_strict(path, "cpu")

    assert audit.schema == META_BUNDLE_SCHEMA
    assert audit.checkpoint_load_strict is True
    assert audit.missing_keys == ()
    assert audit.unexpected_keys == ()
    assert audit.base_checkpoint_id == config["base_checkpoint"]["id"]
    assert dict(audit.class_mapping) == config["class_mapping"]
    assert dict(audit.selection) == _selection()
    for key, value in config["prototypes"].items():
        assert torch.equal(audit.prototypes[key], value)
    assert set(loaded.state_dict()) == set(original)
    for key, value in original.items():
        assert torch.equal(loaded.state_dict()[key].cpu(), value)

    expected = tuple(sorted(name for name, _ in iter_inner_adapter_parameters(loaded)))
    trainable = tuple(sorted(name for name, parameter in loaded.named_parameters() if parameter.requires_grad))
    assert trainable == expected
    assert all(name.endswith(("down.weight", "down.bias", "up.weight", "up.bias", "gate")) for name in trainable)
    assert "log_step_size" not in " ".join(trainable)
    assert all(
        not parameter.requires_grad
        for name, parameter in loaded.named_parameters()
        if name not in expected
    )
    assert audit.trainable_names == expected
    assert audit.trainable_count == sum(loaded.get_parameter(name).numel() for name in expected)
    assert audit.trainable_fraction <= 0.01


def test_strict_loader_rejects_default_lite_model_when_real_budget_exceeds_one_percent(tmp_path):
    path = tmp_path / "over_budget.pth"
    _save_valid_bundle(path, variant="lite_h")
    with pytest.raises(ValueError, match="1%"):
        load_meta_bundle_strict(path, "cpu")


def test_strict_loader_rejects_forbidden_trainable_name(monkeypatch, tmp_path):
    import cvsrffi.meta_checkpoint as meta_checkpoint

    path = tmp_path / "forbidden_name.pth"
    _save_valid_bundle(path)
    original_iterator = meta_checkpoint.iter_inner_adapter_parameters

    def poisoned_iterator(model):
        yield "cls_head.head.weight", model.cls_head.head.weight

    monkeypatch.setattr(meta_checkpoint, "iter_inner_adapter_parameters", poisoned_iterator)
    with pytest.raises(ValueError, match="forbidden trainable state"):
        load_meta_bundle_strict(path, "cpu")
    monkeypatch.setattr(meta_checkpoint, "iter_inner_adapter_parameters", original_iterator)


def test_strict_loader_dispatches_real_dual_adv3b02_builder(tmp_path):
    model_args = {
        "num_classes": 3,
        "num_domains": 2,
        "model_size": "M",
        "dataset": "wisig",
        "input_len": 64,
        "sample_rate_hz": 25e6,
        "model_variant": "base",
        "id_feature_key": "feat_joint",
        "dom_feature_key": "feat_imp",
        "meta_adapter_rank": 4,
        "meta_adapter_sites": "time,freq,fusion",
    }
    model = build_dual_model(**model_args)
    config = _config()
    config["model_args"] = model_args
    path = tmp_path / "dual_meta_bundle.pth"
    save_meta_bundle(path, model, config, _selection())

    loaded, audit = load_meta_bundle_strict(path, "cpu")

    assert set(loaded.state_dict()) == set(model.state_dict())
    assert audit.checkpoint_load_strict is True
    assert audit.trainable_fraction <= 0.01


def test_strict_loader_dispatches_real_dual_fusion_only_profile(tmp_path):
    model_args = {
        "num_classes": 3,
        "num_domains": 2,
        "model_size": "M",
        "dataset": "wisig",
        "input_len": 64,
        "sample_rate_hz": 25e6,
        "model_variant": "base",
        "id_feature_key": "feat_joint",
        "dom_feature_key": "feat_imp",
        "meta_adapter_rank": 4,
        "meta_adapter_sites": "fusion",
    }
    model = build_dual_model(**model_args)
    config = _config()
    config["model_args"] = model_args
    config["meta_adapter_config"]["sites"] = ["fusion"]
    path = tmp_path / "dual_fusion_meta_bundle.pth"

    save_meta_bundle(path, model, config, _selection())
    loaded, audit = load_meta_bundle_strict(path, "cpu")

    assert set(loaded.state_dict()) == set(model.state_dict())
    assert audit.checkpoint_load_strict is True
    assert audit.trainable_fraction <= 0.01
    assert audit.trainable_names
    assert all("meta_adapter_fusion" in name for name in audit.trainable_names)
