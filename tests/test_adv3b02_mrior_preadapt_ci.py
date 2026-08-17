import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper_reproduction.cvs_aligned.adv3b02_mrior_preadapt_ci import (
    MRIORPreadaptResult,
    fit_mrior_preadapted_backbone,
    load_verified_mrior_preadapt_artifact,
    preadapt_key,
    write_mrior_preadapt_artifact,
)


def test_preadapt_key_excludes_new_count_and_downstream_method() -> None:
    assert preadapt_key("20-1", 713101, 5, "leo_rain_weak") == (
        "rx_20_1__seed_713101__k_5__scene_leo_rain_weak"
    )


def _result() -> MRIORPreadaptResult:
    return MRIORPreadaptResult(
        model_state={"id_backbone.weight": torch.tensor([[1.0, -2.0]])},
        loss_trace=[{"step": 1, "loss": 0.25}],
        resource={"adapt_steps": 200, "optimizer": "Adam_minimax"},
    )


def _write_artifact(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "mrior_preadapt"
    manifest = write_mrior_preadapt_artifact(
        root,
        _result(),
        checkpoint_sha256="a" * 64,
        source_cache_sha256="b" * 64,
        support_token_sha256="c" * 64,
        receiver="20-1",
        seed=713101,
        k_shot=5,
        scene="leo_rain_weak",
        method_lock_sha256="d" * 64,
    )
    return root, manifest


def _load_artifact(root: Path, **overrides: object) -> MRIORPreadaptResult:
    expected: dict[str, object] = {
        "expected_checkpoint_sha256": "a" * 64,
        "expected_source_cache_sha256": "b" * 64,
        "expected_support_token_sha256": "c" * 64,
        "expected_receiver": "20-1",
        "expected_seed": 713101,
        "expected_k_shot": 5,
        "expected_scene": "leo_rain_weak",
        "expected_method_lock_sha256": "d" * 64,
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
    assert loaded.loss_trace == [{"step": 1, "loss": 0.25}]
    assert loaded.input_digests == {
        "checkpoint_sha256": "a" * 64,
        "source_cache_sha256": "b" * 64,
        "support_token_sha256": "c" * 64,
        "method_lock_sha256": "d" * 64,
    }
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
        "input_digests",
        "query_unopened_receipt",
    }
    assert "query_rows" not in stored
    assert "query_truth" not in stored
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("expected_checkpoint_sha256", "e" * 64),
        ("expected_source_cache_sha256", "f" * 64),
        ("expected_support_token_sha256", "0" * 64),
        ("expected_receiver", "7-7"),
        ("expected_seed", 713102),
        ("expected_k_shot", 10),
        ("expected_scene", "leo_clear_weak"),
        ("expected_method_lock_sha256", "1" * 64),
    ],
)
def test_loader_rejects_any_artifact_binding_drift(
    tmp_path: Path, field: str, wrong_value: object
) -> None:
    root, _ = _write_artifact(tmp_path)

    with pytest.raises(ValueError):
        _load_artifact(root, **{field: wrong_value})


@pytest.mark.parametrize("surface", ["resource", "loss_trace"])
def test_writer_rejects_query_data_even_when_nested_in_allowed_result_fields(
    tmp_path: Path, surface: str
) -> None:
    result = _result()
    if surface == "resource":
        result.resource["query_rows"] = ["qid_opaque"]
    else:
        result.loss_trace[0]["query_truth"] = "forbidden"
    root = tmp_path / f"forbidden_{surface}"

    with pytest.raises(ValueError, match="query"):
        write_mrior_preadapt_artifact(
            root,
            result,
            checkpoint_sha256="a" * 64,
            source_cache_sha256="b" * 64,
            support_token_sha256="c" * 64,
            receiver="20-1",
            seed=713101,
            k_shot=5,
            scene="leo_rain_weak",
            method_lock_sha256="d" * 64,
        )
    assert not root.exists()


def test_loader_rejects_query_data_in_a_hash_updated_tampered_state(
    tmp_path: Path,
) -> None:
    root, manifest = _write_artifact(tmp_path)
    state_path = root / str(manifest["state_filename"])
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state["resource"]["query_rows"] = ["qid_opaque"]
    torch.save(state, state_path)
    manifest["state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="query"):
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
        seed=713101,
        adapt_steps=1,
        estimate_steps=1,
    )

    assert result.resource["adapt_steps"] == 1
    assert result.resource["optimizer"] == "Adam_minimax"
    assert result.loss_trace[0]["phase"] == "target_support_adaptation"
    assert any(name.startswith("id_backbone.") for name in result.model_state)
    assert result.query_unopened_receipt["query_rows_used_for_training"] == 0
