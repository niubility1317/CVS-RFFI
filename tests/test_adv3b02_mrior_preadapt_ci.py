import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper_reproduction.cvs_aligned.adv3b02_mrior_preadapt_ci import (
    MRIORPreadaptInputBinding,
    MRIORPreadaptResult,
    expected_mrior_preadapt_method_lock,
    fit_mrior_preadapted_backbone,
    load_verified_mrior_preadapt_artifact,
    preadapt_key,
    write_mrior_preadapt_artifact,
)


def test_preadapt_key_excludes_new_count_and_downstream_method() -> None:
    assert preadapt_key("20-1", 713101, 5, "leo_rain_weak") == (
        "rx_20_1__seed_713101__k_5__scene_leo_rain_weak"
    )


def _binding(**overrides: object) -> MRIORPreadaptInputBinding:
    values: dict[str, object] = {
        "checkpoint_sha256": "a" * 64,
        "source_cache_sha256": "b" * 64,
        "support_token_sha256": "c" * 64,
        "target_package_seal_sha256": "d" * 64,
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 5,
        "scene": "leo_rain_weak",
    }
    values.update(overrides)
    return MRIORPreadaptInputBinding.from_verified_values(**values)


def _scalar_losses(*, estimate_steps: float = 7.0) -> dict[str, float]:
    return {
        "loss": 0.25,
        "source_ce": 0.10,
        "target_support_ce": 0.11,
        "weighted_ce": 0.12,
        "dvkl": 0.13,
        "target_ce_weight": 1.0,
        "dvkl_weight": 0.005,
        "mu": 0.5,
        "estimate_loss": 0.14,
        "estimate_zeta": 0.15,
        "estimate_steps": estimate_steps,
    }


def _result() -> MRIORPreadaptResult:
    losses = _scalar_losses()
    trace_steps = [1, *range(20, 201, 20)]
    return MRIORPreadaptResult(
        model_state={"id_backbone.weight": torch.tensor([[1.0, -2.0]])},
        loss_trace=[
            {
                "method": "mrior_sda",
                "scenario": "sealed_by_caller",
                "phase": "target_support_adaptation",
                "step": step,
                "total_steps": 200,
                **losses,
            }
            for step in trace_steps
        ],
        resource={
            "adapt_steps": 200,
            "final_adaptation_losses": losses,
            "optimizer": "Adam_minimax",
            "learning_rate": 0.0006,
            "adv3b02_gradient_updates": 200,
        },
        input_binding=_binding(),
        method_lock=expected_mrior_preadapt_method_lock(),
        is_formal=True,
    )


def _write_artifact(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "mrior_preadapt"
    manifest = write_mrior_preadapt_artifact(
        root,
        _result(),
    )
    return root, manifest


def _load_artifact(root: Path, **overrides: object) -> MRIORPreadaptResult:
    result = _result()
    expected: dict[str, object] = {
        "expected_input_binding_sha256": result.input_binding_sha256,
        "expected_method_lock_sha256": result.method_lock_sha256,
    }
    expected.update(overrides)
    return load_verified_mrior_preadapt_artifact(root, **expected)


def test_frozen_artifact_round_trip_has_only_preadapt_identity_and_query_unopened_receipt(
    tmp_path: Path,
) -> None:
    root, manifest = _write_artifact(tmp_path)

    assert manifest["artifact_id"] == "rx_20_1__seed_713101__k_5__scene_leo_rain_weak"
    assert "new_class_count" not in manifest
    assert "downstream_method" not in manifest
    loaded = _load_artifact(root)
    assert torch.equal(
        loaded.model_state["id_backbone.weight"], torch.tensor([[1.0, -2.0]])
    )
    assert loaded.input_binding == _binding()
    assert loaded.input_binding_sha256 == _binding().canonical_sha256
    assert loaded.method_lock == expected_mrior_preadapt_method_lock()
    assert loaded.method_lock_sha256 == _result().method_lock_sha256
    assert loaded.query_unopened_receipt == {
        "query_opened_before_model_lock": False,
        "query_rows_used_for_training": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "query_class_quota_access": False,
        "query_global_reassignment_access": False,
    }

    stored = torch.load(
        root / str(manifest["state_filename"]), map_location="cpu", weights_only=False
    )
    assert set(stored) == {
        "schema",
        "model_state",
        "loss_trace",
        "resource",
        "input_binding",
        "input_binding_sha256",
        "method_lock",
        "method_lock_sha256",
        "query_unopened_receipt",
    }
    assert "query_rows" not in stored
    assert "query_truth" not in stored
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("checkpoint_sha256", "e" * 64),
        ("source_cache_sha256", "f" * 64),
        ("support_token_sha256", "0" * 64),
        ("target_package_seal_sha256", "1" * 64),
        ("receiver", "7-7"),
        ("seed", 713102),
        ("k_shot", 10),
        ("scene", "leo_clear_weak"),
    ],
)
def test_loader_rejects_any_canonical_input_binding_drift(
    tmp_path: Path, field: str, wrong_value: object
) -> None:
    root, _ = _write_artifact(tmp_path)

    with pytest.raises(ValueError):
        _load_artifact(
            root,
            expected_input_binding_sha256=_binding(**{field: wrong_value}).canonical_sha256,
        )


def test_loader_rejects_canonical_method_lock_drift(tmp_path: Path) -> None:
    root, _ = _write_artifact(tmp_path)

    with pytest.raises(ValueError):
        _load_artifact(root, expected_method_lock_sha256="1" * 64)


@pytest.mark.parametrize("surface", ["resource", "loss_trace"])
def test_writer_rejects_neutral_key_opaque_rows_outside_exact_summary_schema(
    tmp_path: Path, surface: str
) -> None:
    result = _result()
    if surface == "resource":
        result.resource["audit"] = ["qid_opaque"]
    else:
        result.loss_trace[0]["receipt"] = "qid_opaque"
    root = tmp_path / f"forbidden_{surface}"

    with pytest.raises(ValueError, match="schema"):
        write_mrior_preadapt_artifact(root, result)
    assert not root.exists()


def test_loader_rejects_query_data_in_a_hash_updated_tampered_state(
    tmp_path: Path,
) -> None:
    root, manifest = _write_artifact(tmp_path)
    state_path = root / str(manifest["state_filename"])
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state["resource"]["audit"] = ["qid_opaque"]
    torch.save(state, state_path)
    manifest["state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema"):
        _load_artifact(root)


class _TinyIdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 3)
        self.classifier = nn.Linear(3, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        y: torch.Tensor | None = None,
        return_aux: bool = True,
        domain_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del y, return_aux, domain_labels
        features = torch.tanh(self.projection(x))
        return {"feat_joint": features, "logits": self.classifier(features)}


class _TinyADV3B02:
    def __init__(self) -> None:
        self.id_backbone = _TinyIdentityBackbone()
        self.id_feature_key = "feat_joint"
        self.emb_dim = 3


def test_fit_reuses_mrior_minimax_path_with_target_old_support_only() -> None:
    torch.manual_seed(713101)
    source_x = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.8]])
    source_y = torch.tensor([0, 0, 1, 1])
    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=2, shuffle=False)
    target_old_x = torch.tensor([[0.9, 0.0], [0.0, 0.9]])
    target_old_y = torch.tensor([0, 1])

    result = fit_mrior_preadapted_backbone(
        _TinyADV3B02(),
        source_loader,
        target_old_x,
        target_old_y,
        binding=_binding(k_shot=1),
        seed=713101,
        adapt_steps=1,
        estimate_steps=1,
        _test_only_allow_nonfrozen_params=True,
    )

    assert result.resource["adapt_steps"] == 1
    assert result.resource["optimizer"] == "Adam_minimax"
    assert result.loss_trace[0]["phase"] == "target_support_adaptation"
    assert any(name.startswith("id_backbone.") for name in result.model_state)
    assert result.query_unopened_receipt["query_rows_used_for_training"] == 0
    assert result.input_binding == _binding(k_shot=1)
    assert result.method_lock["adapt_steps"] == 1
    assert result.is_formal is False


def test_fit_accepts_the_exact_identity_backbone_returned_by_predictor_loader() -> None:
    """The production loader returns CVSincNet itself, not its dual-model parent."""

    source_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source_y = torch.tensor([0, 1])
    identity_backbone = _TinyIdentityBackbone()
    identity_backbone.emb_dim = 3
    result = fit_mrior_preadapted_backbone(
        identity_backbone,
        DataLoader(TensorDataset(source_x, source_y), batch_size=2, shuffle=False),
        source_x,
        source_y,
        binding=_binding(k_shot=1),
        seed=713101,
        adapt_steps=1,
        estimate_steps=1,
        _test_only_allow_nonfrozen_params=True,
    )

    assert any(name.startswith("id_backbone.") for name in result.model_state)


def test_formal_fit_rejects_one_step_even_with_nominal_verified_binding() -> None:
    source_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source_y = torch.tensor([0, 1])
    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="frozen"):
        fit_mrior_preadapted_backbone(
            _TinyADV3B02(),
            source_loader,
            source_x,
            source_y,
            binding=_binding(k_shot=1),
            seed=713101,
            adapt_steps=1,
            estimate_steps=1,
        )


def test_fit_rejects_target_support_k_mismatch_to_verified_binding() -> None:
    source_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source_y = torch.tensor([0, 1])
    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="K-shot"):
        fit_mrior_preadapted_backbone(
            _TinyADV3B02(),
            source_loader,
            source_x,
            source_y,
            binding=_binding(k_shot=2),
            seed=713101,
            adapt_steps=1,
            estimate_steps=1,
            _test_only_allow_nonfrozen_params=True,
        )


def test_writer_rejects_one_step_nominal_method_lock(tmp_path: Path) -> None:
    result = _result()
    result.method_lock["adapt_steps"] = 1

    with pytest.raises(ValueError, match="method lock"):
        write_mrior_preadapt_artifact(tmp_path / "one_step", result)


def test_verified_binding_requires_target_package_seal() -> None:
    with pytest.raises(ValueError, match="target_package_seal_sha256"):
        _binding(target_package_seal_sha256="")


def test_test_only_nonfrozen_fit_result_cannot_be_written_as_formal_artifact(
    tmp_path: Path,
) -> None:
    source_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source_y = torch.tensor([0, 1])
    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=2, shuffle=False)
    result = fit_mrior_preadapted_backbone(
        _TinyADV3B02(),
        source_loader,
        source_x,
        source_y,
        binding=_binding(k_shot=1),
        seed=713101,
        adapt_steps=1,
        estimate_steps=1,
        _test_only_allow_nonfrozen_params=True,
    )

    with pytest.raises(ValueError, match="formal"):
        write_mrior_preadapt_artifact(tmp_path / "test_only", result)
