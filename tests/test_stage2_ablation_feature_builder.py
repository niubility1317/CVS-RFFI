from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch

from cvsrffi.stage2_ablation_feature_builder import (
    _deployment_prototypes,
    build_feature_cache_from_sealed_row_pair,
)
from cvsrffi.stage2_d81_query_evaluation import (
    _require_cross_state_lock,
)


def _handle(value: str) -> str:
    return "cls_" + hashlib.sha256(value.encode()).hexdigest()


def test_feature_builder_has_no_truth_or_dataset_input_surface() -> None:
    parameters = inspect.signature(
        build_feature_cache_from_sealed_row_pair
    ).parameters
    assert not any(
        token in name
        for name in parameters
        for token in (
            "truth",
            "dataset",
            "clean",
            "source_sample",
        )
    )


def test_phase1_identity_prototypes_are_padded_to_288(
    tmp_path: Path,
) -> None:
    path = tmp_path / "phase2_zid_prototypes.pt"
    prototypes = torch.arange(
        6 * 160, dtype=torch.float32
    ).reshape(6, 160) + 1.0
    torch.save({"prototypes": prototypes}, path)
    result = _deployment_prototypes(
        path,
        expected_old_class_count=6,
    )
    assert result.shape == (6, 288)
    assert np.allclose(
        np.linalg.norm(result, axis=1),
        np.ones(6),
    )
    assert np.count_nonzero(result[:, 160:]) == 0


def test_d81_cross_state_lock_accepts_registered_k2() -> None:
    old = [
        {
            "class_index": index,
            "class_handle": _handle(f"old-{index}"),
        }
        for index in range(6)
    ]
    new = [
        {
            "class_index": index + 6,
            "class_handle": _handle(f"new-{index}"),
        }
        for index in range(5)
    ]
    common = {
        "receiver": "20-1",
        "seed": 840001,
        "k_shot": 2,
        "phase1_checkpoint_sha256": "a" * 64,
        "feature_runtime_sha256": "b" * 64,
        "method_lock_sha256": "c" * 64,
        "row_handle": "row-1",
        "row_manifest_sha256": "d" * 64,
    }
    before_enrollment = {
        **common,
        "registration_state": "before",
        "registered_classes": old,
    }
    before_apply = {
        **common,
        "registration_state": "before",
    }
    after_enrollment = {
        **common,
        "registration_state": "after",
        "registered_classes": old + new,
    }
    after_apply = {
        **common,
        "registration_state": "after",
    }
    old_handles, all_handles, k_shot = _require_cross_state_lock(
        before_enrollment,
        before_apply,
        after_enrollment,
        after_apply,
    )
    assert k_shot == 2
    assert len(old_handles) == 6
    assert len(all_handles) == 11
