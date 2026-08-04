from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import cvsrffi.stage2_d92_pr160_target125 as target
from test_stage2_d109_target125 import _common


METHOD_LOCK = Path(__file__).resolve().parents[1] / "configs" / "d138_d92_lite_pr160_r2.json"
METHOD = {
    "method_lock_path": METHOD_LOCK,
    "expected_method_lock_sha256": target.core.METHOD_LOCK_SHA256,
}


def _unit_feature(token: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    row = generator.standard_normal(160).astype(np.float64)
    row /= np.linalg.norm(row)
    return row.astype(np.float32)


class FakePR160Materializer:
    feature_width = 160

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.device = "cpu"

        def forbidden_d92_fit(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("D138 core must not invoke the legacy d92_fit hook")

        self.d92_fit = forbidden_d92_fit

    def __call__(self, request: dict[str, object]) -> dict[str, object]:
        phase = str(request["phase"])
        outer_id = str(request["outer_id"])
        scene = str(request["scene"])
        active_k = int(request["k_shot"])
        new_count = int(request["new_count"])
        old = tuple(f"old_{index}" for index in range(6))
        new = tuple(f"new_{index}" for index in range(new_count))
        classes = old if phase == "before" else old + new
        labels: list[str] = []
        physical_ids: list[str] = []
        features: list[np.ndarray] = []
        for label in classes:
            for shot in range(active_k):
                token = f"{outer_id}:{scene}:support:{label}:{shot}"
                labels.append(label)
                physical_ids.append(token)
                features.append(_unit_feature(token))
        query_ids = tuple(
            f"{outer_id}:{scene}:query:{index}" for index in range(3)
        )
        return {
            "support_features": np.stack(features).astype(np.float32),
            "support_labels": labels,
            "registered_classes": classes,
            "support_physical_ids": physical_ids,
            "query_features": np.stack([_unit_feature(item) for item in query_ids]),
            "query_physical_ids": query_ids,
        }


def test_pr160_adapter_exercises_typed_smoke_and_full_eight_shard_merge(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(target.FORMAL_ISOLATION_ENV, "1")
    monkeypatch.setattr(target, "PR160StateMaterializer", FakePR160Materializer)
    common = _common(tmp_path)
    smoke = target.smoke_d92_pr160_target125_prepared_state(
        **METHOD,
        **common,
        extractor_runtime_path=tmp_path / "test-extractor.pt",
        output_dir=tmp_path / "smoke",
    )
    assert smoke["candidate_id"] == target.CANDIDATE_ID
    assert smoke["status"].endswith("SMOKE_PASS")
    smoke_predictions = json.loads(
        Path(smoke["smoke_predictions"]).read_text(encoding="utf-8")
    )
    assert smoke_predictions["truth_open"] is False

    manifests: list[Path] = []
    for shard_index in range(target.SHARD_COUNT):
        result = target.predict_d92_pr160_target125(
            **METHOD,
            **common,
            extractor_runtime_path=tmp_path / "test-extractor.pt",
            output_dir=tmp_path / f"shard_{shard_index}",
            shard_index=shard_index,
            device="cpu",
        )
        manifests.append(Path(result["prediction_shard_manifest"]))

    merged = target.predict_d92_pr160_target125(
        **METHOD,
        shard_manifest_paths=manifests,
        output_dir=tmp_path / "merged",
    )
    assert merged["outer_job_count"] == 125
    assert merged["arm_pair_count"] == 375
    assert merged["surface_count"] == 750
    validated = target.validate_d92_pr160_prediction_manifest(
        **METHOD,
        prediction_manifest_path=Path(merged["prediction_manifest"]),
        expected_prediction_manifest_file_sha256=merged[
            "prediction_manifest_file_sha256"
        ],
    )
    assert validated["surface_count"] == 750
