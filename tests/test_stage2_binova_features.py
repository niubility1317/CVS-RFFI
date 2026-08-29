from __future__ import annotations

import numpy as np
import pytest
import torch

from cvsrffi.stage2_binova_features import (
    BiNOVAFeatureError,
    BiNOVAFeatures,
    BiNOVAQuery,
    BiNOVASupport,
    class_balanced_domain_context,
    extract_binova_features,
)


def _context() -> dict[str, str]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-a",
        "split_id": "split-a",
    }


def _features(rows: int = 6) -> BiNOVAFeatures:
    base = np.arange(rows, dtype=np.float32)[:, None]
    return BiNOVAFeatures(
        identity160=np.repeat(base, 160, axis=1),
        late_time160=np.repeat(base + 1.0, 160, axis=1),
        domain160=np.repeat(base + 2.0, 160, axis=1),
        fft96=np.repeat(base + 3.0, 96, axis=1),
        physical6=np.repeat(base + 4.0, 6, axis=1),
        physical_ids=tuple(f"p-{index}" for index in range(rows)),
    )


def test_support_rejects_duplicate_physical_ids_and_forbidden_context() -> None:
    features = _features()
    with pytest.raises(BiNOVAFeatureError, match="physical IDs must be unique"):
        BiNOVAFeatures(
            identity160=features.identity160,
            late_time160=features.late_time160,
            domain160=features.domain160,
            fft96=features.fft96,
            physical6=features.physical6,
            physical_ids=("same",) * 6,
        )

    forbidden = {**_context(), "query_truth": "forbidden"}
    with pytest.raises(BiNOVAFeatureError, match="forbidden Phase2 context"):
        BiNOVASupport(
            features=features,
            labels=np.arange(6, dtype=np.int64),
            ranks=np.zeros(6, dtype=np.int64),
            context=forbidden,
        )


def test_query_has_no_label_or_role_constructor_surface() -> None:
    query = BiNOVAQuery(features=_features(2), context=_context())
    assert not hasattr(query, "labels")
    assert not hasattr(query, "roles")
    with pytest.raises(TypeError):
        BiNOVAQuery(
            features=_features(2),
            context=_context(),
            labels=np.zeros(2, dtype=np.int64),
        )


def test_class_balanced_context_is_invariant_to_class_row_replication() -> None:
    class_points = np.asarray([[0.0], [2.0], [5.0]], dtype=np.float32)
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    original = class_balanced_domain_context(class_points, labels)

    repeated = np.concatenate([class_points[:2], np.repeat(class_points[2:], 20, axis=0)])
    repeated_labels = np.concatenate(
        [labels[:2], np.repeat(labels[2:], 20, axis=0)]
    )
    replicated = class_balanced_domain_context(repeated, repeated_labels)
    np.testing.assert_allclose(replicated, original, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(original, np.asarray([2.0], dtype=np.float32), atol=1.0e-5)


class _ToyFrozenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(
        self,
        rows: torch.Tensor,
        y_tx: None = None,
        grl_lambda: float = 1.0,
        return_aux: bool = True,
    ) -> dict[str, object]:
        del y_tx, grl_lambda, return_aux
        mean = rows.mean(dim=(1, 2), keepdim=False)[:, None]
        return {
            "z_id": mean.repeat(1, 160),
            "z_dom": (mean + 1.0).repeat(1, 160),
            "aux_id": {"t_emb": (mean + 2.0).repeat(1, 160)},
        }


def test_extract_uses_one_received_iq_and_returns_fixed_geometry() -> None:
    model = _ToyFrozenModel().eval()
    iq = torch.zeros(3, 2, 256)
    iq[:, 0, :] = torch.linspace(-1.0, 1.0, 256)
    iq[:, 1, :] = torch.linspace(1.0, -1.0, 256)
    result = extract_binova_features(
        model,
        iq,
        physical_ids=("a", "b", "c"),
        device="cpu",
    )
    assert result.identity160.shape == (3, 160)
    assert result.late_time160.shape == (3, 160)
    assert result.domain160.shape == (3, 160)
    assert result.fft96.shape == (3, 96)
    assert result.physical6.shape == (3, 6)
    assert np.isfinite(result.physical6).all()


def test_feature_geometry_rejects_nonfinite_rows() -> None:
    values = _features(2)
    broken = values.identity160.copy()
    broken[0, 0] = np.nan
    with pytest.raises(BiNOVAFeatureError, match="finite"):
        BiNOVAFeatures(
            identity160=broken,
            late_time160=values.late_time160,
            domain160=values.domain160,
            fft96=values.fft96,
            physical6=values.physical6,
            physical_ids=values.physical_ids,
        )
