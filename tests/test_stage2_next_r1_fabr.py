from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi import stage2_next_r1_fabr as fabr


def _asset(block_id: str = "t1_norm_affine") -> fabr.FABRAsset:
    basis = np.zeros((fabr.BLOCK_DIMENSIONS[block_id], fabr.RANK), dtype=np.int8)
    basis[0, 0] = 64
    basis[1, 1] = 64
    return fabr.FABRAsset(
        checkpoint_sha256="1" * 64,
        phase1_seal_sha256="2" * 64,
        phase1_selection_sha256="3" * 64,
        block_id=block_id,
        basis_qint8=basis,
        basis_scale_fp16=np.asarray([1.0 / 64.0, 1.0 / 64.0], dtype=np.float16),
        fisher_k_fp16=np.asarray([[1.0, 0.2], [0.2, 1.5]], dtype=np.float16),
        forward_jitter_tolerance_fp16=np.asarray([0.0], dtype=np.float16),
    )


def _binding(
    asset: fabr.FABRAsset,
    *,
    checkpoint_sha256: str | None = None,
    phase1_seal_sha256: str | None = None,
    representation_rule_sha256: str | None = None,
) -> fabr.FABRRuntimeBinding:
    return fabr.FABRRuntimeBinding(
        actual_checkpoint_sha256=checkpoint_sha256 or asset.checkpoint_sha256,
        phase1_seal_sha256=phase1_seal_sha256 or asset.phase1_seal_sha256,
        representation_rule_sha256=(
            representation_rule_sha256 or asset.representation_rule_sha256
        ),
    )


def _case(k: int, seed: int = 31) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    # Deliberately non-lexical registry order: FABR must preserve it.
    classes = ("c2", "c0", "c1")
    centers = rng.normal(size=(len(classes), fabr.Z_DIM))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for class_index, label in enumerate(classes):
        rows.append(centers[class_index][None, :] + 0.03 * rng.normal(size=(k, fabr.Z_DIM)))
        labels.extend([label] * k)
        # Reverse declaration order to prove physical-ID canonicalisation.
        physical_ids.extend(f"pid-{label}-{k - index - 1}" for index in range(k))
    base = np.concatenate(rows, axis=0).astype(np.float32)
    direction = rng.normal(size=(fabr.RANK, base.shape[0], fabr.Z_DIM)).astype(np.float32)
    direction[0, :, 0] += np.linspace(-2.0, 2.0, base.shape[0]).astype(np.float32)
    direction[1, :, 1] += np.repeat(np.arange(len(classes)), k).astype(np.float32)
    direction *= np.float32(0.08)
    return {"base": base, "direction": direction, "physical_ids": tuple(physical_ids)}, tuple(labels), classes, tuple(physical_ids)


def _forward(token: dict[str, object], coefficient: np.ndarray) -> fabr.FABRForwardBatch:
    return fabr.FABRForwardBatch(
        features=np.ascontiguousarray(
            token["base"] + np.einsum("r,rnd->nd", coefficient, token["direction"]),
            dtype=np.float32,
        ),
        physical_ids=token["physical_ids"],
    )


def test_signed_pre_relu160_uses_positive_then_signed_and_exact_zero_fails() -> None:
    pre = np.zeros((3, fabr.Z_DIM), dtype=np.float32)
    pre[0, :2] = (3.0, -4.0)
    pre[1, :2] = (-3.0, -4.0)
    pre[2, 0] = 1.0
    observed = np.maximum(pre, np.float32(0.0))
    z = fabr.signed_pre_relu160(pre, observed_post_relu=observed)
    assert z.dtype == np.float32
    assert np.allclose(np.linalg.norm(z.astype(np.float64), axis=1), 1.0)
    assert z[0, 0] > 0 and z[0, 1] == 0
    assert z[1, 0] < 0 and z[1, 1] < 0
    mismatch = observed.copy()
    mismatch[0, 1] = 1.0
    with pytest.raises(fabr.FABRError, match="does not bind"):
        fabr.signed_pre_relu160(pre, observed_post_relu=mismatch)
    zero = pre.copy()
    zero[2] = 0.0
    with pytest.raises(fabr.FABRError, match="exactly zero"):
        fabr.signed_pre_relu160(zero)


def test_strict_top_tie_rejects_without_class_or_id_fallback() -> None:
    logits = np.asarray([[1.0, 0.5, -1.0], [0.1, 0.9, 0.8]], dtype=np.float32)
    assert np.array_equal(fabr.strict_top1_predictions(logits), np.asarray([0, 1]))
    tied = logits.copy()
    tied[0, 1] = tied[0, 0]
    with pytest.raises(fabr.FABRTieError, match="TIE_UNRESOLVED"):
        fabr.strict_top1_predictions(tied)
    with pytest.raises(fabr.FABRError, match="exact float32 aliases"):
        fabr.require_exact_logit_alias(logits, tied)
    fabr.require_exact_logit_alias(logits, logits.copy())


def test_canonical_order_preserves_registry_and_sorts_only_physical_ids_within_class() -> None:
    classes = ("z", "a")
    labels = ("a", "z", "a", "z", "a", "z", "a", "z", "a", "z")
    ids = ("a-2", "z-4", "a-0", "z-2", "a-4", "z-0", "a-3", "z-3", "a-1", "z-1")
    order = fabr.canonical_support_order(labels, classes, ids)
    assert tuple(order[:5]) == (5, 9, 3, 7, 1)
    assert tuple(order[5:]) == (2, 8, 0, 6, 4)


def test_phase1_fisher_geometry_is_finite_and_uses_positive_eps() -> None:
    gradients = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    geometry = fabr.phase1_fisher_geometry(gradients, ("r0", "r0", "r1", "r1"))
    assert geometry.epsilon_f > 0.0
    assert np.all(np.linalg.eigvalsh(geometry.fisher) > 0.0)
    with pytest.raises(fabr.FABRError, match=r"tr\(G\)"):
        fabr.phase1_fisher_geometry(np.zeros((4, 2), dtype=np.float32), ("r0", "r0", "r1", "r1"))


def test_fit_api_has_no_query_or_forbidden_runtime_inputs() -> None:
    parameters = set(inspect.signature(fabr.fit_fabr_support).parameters)
    assert parameters == {
        "asset",
        "support_token",
        "labels",
        "registered_classes",
        "forward_with_coeff",
        "support_physical_ids",
        "runtime_binding",
    }
    forbidden = {"query", "truth", "role", "old_class_count", "quota", "source", "clean"}
    assert not any(any(token in name for token in forbidden) for name in parameters)


@pytest.mark.parametrize("k", [1, 5])
def test_fabr_fit_uses_exact_six_forwards_and_k_specific_objective(k: int) -> None:
    token, labels, classes, physical_ids = _case(k)
    calls: list[np.ndarray] = []

    def forward(value: dict[str, object], coefficient: np.ndarray) -> fabr.FABRForwardBatch:
        calls.append(np.asarray(coefficient).copy())
        return _forward(value, coefficient)

    asset = _asset()
    state = fabr.fit_fabr_support(
        asset,
        token,
        labels,
        classes,
        forward,
        support_physical_ids=physical_ids,
        runtime_binding=_binding(asset),
    )
    assert state.registered_classes == classes
    assert len(calls) == 6
    assert np.array_equal(calls[0], np.zeros(2, dtype=np.float32))
    assert state.resource_receipt.additional_support_forward_calls == 5
    assert state.resource_receipt.protocol_closed
    assert state.support_receipt["physical_loo_compactness_active"] is (k == 5)
    assert state.support_receipt["query_rows_used_for_fit"] == 0
    assert state.support_receipt["actual_checkpoint_sha256"] == asset.checkpoint_sha256
    assert state.support_receipt["phase1_seal_sha256"] == asset.phase1_seal_sha256
    assert state.support_receipt["representation_rule_sha256"] == fabr.REPRESENTATION_RULE_SHA256
    h_fabr = np.asarray(state.support_receipt["h_fabr"], dtype=np.float64)
    assert h_fabr.shape == (fabr.RANK, fabr.RANK)
    assert np.allclose(h_fabr, h_fabr.T)
    assert np.all(np.linalg.eigvalsh(h_fabr) >= 0.0)
    coeff = state.coeff_fp16.astype(np.float64)
    fisher_k = fabr.decode_fabr_fisher_k(asset)
    assert float(coeff @ fisher_k @ coeff) <= fabr.RHO**2


def test_k1_compactness_is_exactly_zero_and_k5_is_physical_loo() -> None:
    token1, labels1, classes1, ids1 = _case(1)
    total1, separation1, compactness1 = fabr.fabr_support_objective(
        token1["base"], labels=labels1, registered_classes=classes1, physical_ids=ids1
    )
    assert compactness1 == 0.0
    assert total1 == pytest.approx(separation1)
    token5, labels5, classes5, ids5 = _case(5)
    _total5, _separation5, compactness5 = fabr.fabr_support_objective(
        token5["base"], labels=labels5, registered_classes=classes5, physical_ids=ids5
    )
    assert compactness5 >= 0.0


def test_fabr_rejects_callback_physical_row_reordering() -> None:
    token, labels, classes, physical_ids = _case(5)
    count = 0

    def reordered(value: dict[str, object], coefficient: np.ndarray) -> fabr.FABRForwardBatch:
        nonlocal count
        count += 1
        batch = _forward(value, coefficient)
        if count == 2:
            return fabr.FABRForwardBatch(batch.features[::-1], tuple(reversed(batch.physical_ids)))
        return batch

    asset = _asset()
    with pytest.raises(fabr.FABRError, match="physical IDs do not exactly match"):
        fabr.fit_fabr_support(
            asset,
            token,
            labels,
            classes,
            reordered,
            support_physical_ids=physical_ids,
            runtime_binding=_binding(asset),
        )


def test_fit_requires_all_three_explicit_runtime_identities() -> None:
    token, labels, classes, physical_ids = _case(1)
    asset = _asset()
    calls = 0

    def counted(value: dict[str, object], coefficient: np.ndarray) -> fabr.FABRForwardBatch:
        nonlocal calls
        calls += 1
        return _forward(value, coefficient)

    with pytest.raises(fabr.FABRError, match="explicit immutable runtime binding"):
        fabr.fit_fabr_support(
            asset, token, labels, classes, counted, support_physical_ids=physical_ids
        )
    mismatches = (
        (_binding(asset, checkpoint_sha256="4" * 64), "actual checkpoint"),
        (_binding(asset, phase1_seal_sha256="5" * 64), "Phase1 seal"),
        (_binding(asset, representation_rule_sha256="6" * 64), "representation-rule"),
    )
    for binding, message in mismatches:
        with pytest.raises(fabr.FABRError, match=message):
            fabr.fit_fabr_support(
                asset,
                token,
                labels,
                classes,
                counted,
                support_physical_ids=physical_ids,
                runtime_binding=binding,
            )
    assert calls == 0


def test_functional_override_uses_joint_linear_tap_and_never_writes_state() -> None:
    torch = pytest.importorskip("torch")

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.joint_proj = torch.nn.Sequential(torch.nn.Linear(4, fabr.Z_DIM))

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cls_head = Head()

        def forward(self, value: object, **_kwargs: object) -> dict[str, object]:
            pre = self.cls_head.joint_proj[0](value)
            return {"feat_joint": torch.relu(pre)}

    model = Toy().eval()
    with torch.no_grad():
        model.cls_head.joint_proj[0].weight.zero_()
        model.cls_head.joint_proj[0].bias.fill_(-1.0)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    asset = _asset("joint_proj_bias")
    result = fabr.functional_forward_signed_pre_relu160(
        model,
        torch.zeros((2, 4), dtype=torch.float32),
        ("p0", "p1"),
        asset,
        np.zeros(2, dtype=np.float32),
        functional_kwargs={},
        runtime_binding=_binding(asset),
    )
    assert result.receipt.signed_rows == 2
    assert result.receipt.positive_rows == 0
    assert result.receipt.actual_checkpoint_sha256 == asset.checkpoint_sha256
    assert result.receipt.phase1_seal_sha256 == asset.phase1_seal_sha256
    assert result.receipt.representation_rule_sha256 == fabr.REPRESENTATION_RULE_SHA256
    assert result.receipt.state_before_sha256 == result.receipt.state_after_sha256
    assert result.batch.features.shape == (2, fabr.Z_DIM)
    assert np.all(result.batch.features < 0.0)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key])


def test_functional_boundary_rejects_checkpoint_identity_mismatch_before_forward() -> None:
    torch = pytest.importorskip("torch")

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.joint_proj = torch.nn.Sequential(torch.nn.Linear(4, fabr.Z_DIM))

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cls_head = Head()

        def forward(self, value: object, **_kwargs: object) -> dict[str, object]:
            pre = self.cls_head.joint_proj[0](value)
            return {"feat_joint": torch.relu(pre)}

    model = Toy().eval()
    asset = _asset("joint_proj_bias")
    with pytest.raises(fabr.FABRError, match="actual checkpoint"):
        fabr.functional_forward_signed_pre_relu160(
            model,
            torch.zeros((1, 4), dtype=torch.float32),
            ("p0",),
            asset,
            np.zeros(2, dtype=np.float32),
            functional_kwargs={},
            runtime_binding=_binding(asset, checkpoint_sha256="7" * 64),
        )
