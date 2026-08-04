from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_next_r1_assets as assets
from cvsrffi import stage2_next_r1_fabr as fabr
from cvsrffi import stage2_next_r1_tsl as tsl


def _unit(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    rows /= np.sqrt(np.sum(rows * rows, axis=1, keepdims=True))
    return np.ascontiguousarray(rows, dtype=np.float32)


def _sha(character: str) -> str:
    return character * 64


def _fixture(seed: int = 612) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    receivers = tuple(f"r{index}" for index in range(assets.FROZEN_RECEIVER_COUNT))
    classes = tuple(f"c{index}" for index in range(assets.FROZEN_CLASS_COUNT))
    held_receiver = receivers[-1]
    held_class = classes[-1]
    active_receivers = receivers[:-1]
    active_classes = classes[:-1]
    cells: list[tsl.Phase1Cell] = []
    cell_by_pair: dict[tuple[str, str], tsl.Phase1Cell] = {}
    receiver_rows: list[str] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for receiver_index, receiver in enumerate(active_receivers):
        for class_index, class_label in enumerate(active_classes):
            rows = rng.normal(scale=0.025, size=(assets.FROZEN_PHYSICAL_PER_CELL, fabr.Z_DIM))
            rows[:, class_index] += 1.0
            rows[:, 31 + receiver_index] += 0.15
            rows = _unit(rows)
            ids = tuple(f"pid-{receiver}-{class_label}-{index:02d}" for index in range(len(rows)))
            cell = tsl.Phase1Cell(receiver, class_label, ids, rows)
            cells.append(cell)
            cell_by_pair[(receiver, class_label)] = cell
            receiver_rows.extend([receiver] * len(rows))
            labels.extend([class_label] * len(rows))
            physical_ids.extend(ids)
    receiver_rows_t = tuple(receiver_rows)
    labels_t = tuple(labels)
    physical_ids_t = tuple(physical_ids)
    assert len(labels_t) == 420

    # Each receiver lies on exactly one frozen axis.  The paired axes retain
    # zero cross-moment after any one receiver is held out, so the intended
    # rank-two subspace is stable before and after actual INT8 quantisation.
    receiver_vectors = np.asarray(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.0, 0.5), (0.0, -0.5)),
        dtype=np.float64,
    )
    latent = np.asarray(
        [receiver_vectors[active_receivers.index(item)] for item in receiver_rows_t], dtype=np.float64
    )
    blocks: list[assets.Phase1GradientBlock] = []
    for block_index, block_id in enumerate(fabr.BLOCK_TIE_ORDER):
        dimension = fabr.BLOCK_DIMENSIONS[block_id]
        direction, _ = np.linalg.qr(rng.normal(size=(dimension, fabr.RANK)))
        # The same zero-mean, direction-orthogonal residual bank is reused by
        # every receiver.  Thus F is full rank without rotating the intended
        # generalized receiver subspace when one receiver is left out.
        half = rng.normal(size=(35, dimension))
        half -= (half @ direction) @ direction.T
        residual = np.concatenate((half, -half), axis=0)
        residual = np.concatenate((residual,) * len(active_receivers), axis=0)
        gradients = 3.0 * latent @ direction.T + 0.8 * residual
        blocks.append(
            assets.Phase1GradientBlock(
                block_id=block_id,
                gradients=np.ascontiguousarray(gradients, dtype=np.float32),
                phase1_receiver_ids=receiver_rows_t,
                phase1_physical_ids=physical_ids_t,
            )
        )
    loo_bindings: list[assets.TSLPhysicalLOOBinding] = []
    for receiver in active_receivers:
        support_rows: list[np.ndarray] = []
        support_labels: list[str] = []
        support_ids: list[str] = []
        for class_label in active_classes:
            cell = cell_by_pair[(receiver, class_label)]
            support_rows.extend(cell.z160[:5])
            support_labels.extend([class_label] * 5)
            support_ids.extend(cell.physical_ids[:5])
        for class_label in active_classes:
            cell = cell_by_pair[(receiver, class_label)]
            fold = tsl.Phase1PhysicalLOOFold(
                fold_id=f"loo-{receiver}-{class_label}",
                support_z160=np.ascontiguousarray(np.stack(support_rows), dtype=np.float32),
                support_labels=tuple(support_labels),
                registered_classes=active_classes,
                support_physical_ids=tuple(support_ids),
                validation_z160=np.ascontiguousarray(cell.z160[5:], dtype=np.float32),
                validation_labels=(class_label,) * 9,
                validation_physical_ids=cell.physical_ids[5:],
            )
            loo_bindings.append(assets.TSLPhysicalLOOBinding(receiver, class_label, fold))
    fit_root = assets.phase1_fit_physical_id_root(receiver_rows_t, labels_t, physical_ids_t)
    cell_root = tsl.phase1_cell_physical_id_root(cells)
    seal = assets.Phase1FoldSeal(
        fold_id="r6-c5",
        held_receiver=held_receiver,
        held_class=held_class,
        checkpoint_sha256=_sha("a"),
        representation_rule_sha256=_sha("b"),
        row_phase1_seal_sha256=_sha("c"),
        phase1_fit_physical_id_root_sha256=fit_root,
        phase1_cell_physical_id_root_sha256=cell_root,
    )
    return {
        "blocks": tuple(blocks),
        "labels": labels_t,
        "seal": seal,
        "receivers": receivers,
        "classes": classes,
        "cells": tuple(cells),
        "loo_bindings": tuple(loo_bindings),
    }


def _validation(
    block_id: str, basis: np.ndarray, coefficient: np.ndarray, phase1_labels: tuple[str, ...]
) -> assets.Phase1DirectionalValidation:
    del block_id, phase1_labels
    action = float(np.max(np.abs(basis @ coefficient.astype(np.float64))))
    values = {"c0": 8, "c1": 7, "c2": 6, "c3": 8, "c4": 7, "c5": 5}
    totals = {key: 10 for key in values}
    return assets.Phase1DirectionalValidation(
        basis_sha256=assets._basis_sha256(basis),
        coefficient_sha256=assets._coefficient_sha256(coefficient),
        baseline_total_correct=sum(values.values()),
        perturbed_total_correct=sum(values.values()),
        baseline_per_class_correct=values,
        perturbed_per_class_correct=values,
        per_class_total=totals,
        forward_action_max_abs_delta=action,
        repeated_forward_jitter_max_abs_delta=action / 8.0,
        validation_seal_sha256=_sha("d"),
    )


def _build() -> assets.NextR1Phase1AssetBundle:
    value = _fixture()
    return assets.build_next_r1_phase1_assets(
        value["blocks"],
        value["labels"],
        value["seal"],
        value["receivers"],
        value["classes"],
        value["cells"],
        value["loo_bindings"],
        _validation,
    )


def test_builds_deterministic_immutable_cross_bound_fabr_tsl_bundle() -> None:
    first = _build()
    second = _build()
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.receipt["complete_tsl_cell_grid"] is True
    assert first.receipt["complete_unique_tsl_physical_loo_coverage"] is True
    assert first.fabr_asset.checkpoint_sha256 == first.tsl_prior.checkpoint_sha256
    assert first.fabr_asset.phase1_seal_sha256 == first.receipt["row_phase1_seal_sha256"]
    assert first.tsl_prior.representation_rule_sha256 == first.receipt["representation_rule_sha256"]
    assert first.fabr_asset.phase1_selection_sha256 == first.receipt["selection_sha256"]
    assert first.receipt["selected_minimum_principal_cosine"] >= 0.9
    assert np.linalg.matrix_rank(fabr.decode_fabr_basis(first.fabr_asset)) == fabr.RANK
    k = fabr.decode_fabr_fisher_k(first.fabr_asset)
    assert np.all(np.linalg.eigvalsh(k) > 0.0)
    assert np.linalg.cond(k) <= fabr.MAX_CONDITION
    assert first.fabr_asset.basis_qint8.flags.writeable is False
    assert first.tsl_prior.q_logv0.flags.writeable is False
    assert len(first.phase1_receiver_registry_sha256) == 64
    assert len(first.phase1_class_registry_sha256) == 64


def test_api_has_explicit_phase1_labels_and_no_runtime_forbidden_inputs() -> None:
    parameters = tuple(inspect.signature(assets.build_next_r1_phase1_assets).parameters)
    assert "phase1_labels" in parameters
    forbidden = {"target", "query", "truth", "role", "quota", "source", "clean"}
    assert not any(any(token in parameter for token in forbidden) for parameter in parameters)


def test_validation_is_called_for_all_four_signed_rank_directions_on_actual_basis() -> None:
    fixture = _fixture()
    seen: list[tuple[str, bytes, bytes]] = []

    def callback(block_id: str, basis: np.ndarray, coefficient: np.ndarray, phase1_labels: tuple[str, ...]):
        seen.append((block_id, basis.tobytes(order="C"), coefficient.tobytes(order="C")))
        return _validation(block_id, basis, coefficient, phase1_labels)

    bundle = assets.build_next_r1_phase1_assets(
        fixture["blocks"], fixture["labels"], fixture["seal"], fixture["receivers"], fixture["classes"],
        fixture["cells"], fixture["loo_bindings"], callback,
    )
    assert len(seen) == 16
    selected_basis = fabr.decode_fabr_basis(bundle.fabr_asset)
    selected_bytes = selected_basis.tobytes(order="C")
    assert sum(basis == selected_bytes for _block, basis, _coefficient in seen) == 4
    selected_coefficients = [coefficient for _block, basis, coefficient in seen if basis == selected_bytes]
    expected = []
    for direction in range(fabr.RANK):
        for sign in (-1.0, 1.0):
            value = np.zeros(fabr.RANK, dtype=np.float32)
            value[direction] = np.float32(sign * fabr.DELTA)
            expected.append(value.tobytes(order="C"))
    assert set(selected_coefficients) == set(expected)


def test_records_floor_regression_without_blocking_real_performance_test() -> None:
    fixture = _fixture()

    def callback(block_id: str, basis: np.ndarray, coefficient: np.ndarray, phase1_labels: tuple[str, ...]):
        result = _validation(block_id, basis, coefficient, phase1_labels)
        if block_id == "t1_norm_affine":
            changed = dict(result.perturbed_per_class_correct)
            changed["c2"] = 0
            return assets.Phase1DirectionalValidation(
                basis_sha256=result.basis_sha256,
                coefficient_sha256=result.coefficient_sha256,
                baseline_total_correct=result.baseline_total_correct,
                perturbed_total_correct=sum(changed.values()),
                baseline_per_class_correct=result.baseline_per_class_correct,
                perturbed_per_class_correct=changed,
                per_class_total=result.per_class_total,
                forward_action_max_abs_delta=result.forward_action_max_abs_delta,
                repeated_forward_jitter_max_abs_delta=result.repeated_forward_jitter_max_abs_delta,
                validation_seal_sha256=result.validation_seal_sha256,
            )
        return result

    bundle = assets.build_next_r1_phase1_assets(
        fixture["blocks"], fixture["labels"], fixture["seal"], fixture["receivers"], fixture["classes"],
        fixture["cells"], fixture["loo_bindings"], callback,
    )
    assert bundle.receipt["phase1_performance_gate_used"] is False
    if bundle.fabr_asset.block_id == "t1_norm_affine":
        assert bundle.receipt["selected_phase1_floor_non_decrease_all"] is False


def test_tie_uses_frozen_t1_then_t2_then_t3_then_joint_order() -> None:
    tied = tuple(
        SimpleNamespace(block_id=block_id, primary_eigenvalue=0.25)
        for block_id in fabr.BLOCK_TIE_ORDER
    )
    assert assets._select_candidate(tied).block_id == "t1_norm_affine"


def test_rejects_missing_tsl_grid_coverage_and_cross_bound_cell_root() -> None:
    fixture = _fixture()
    missing = fixture["loo_bindings"][:-1]
    with pytest.raises(assets.NextR1AssetError, match="coverage"):
        assets.build_next_r1_phase1_assets(
            fixture["blocks"], fixture["labels"], fixture["seal"], fixture["receivers"], fixture["classes"],
            fixture["cells"], missing, _validation,
        )
    bad_seal = assets.Phase1FoldSeal(
        fold_id=fixture["seal"].fold_id,
        held_receiver=fixture["seal"].held_receiver,
        held_class=fixture["seal"].held_class,
        checkpoint_sha256=fixture["seal"].checkpoint_sha256,
        representation_rule_sha256=fixture["seal"].representation_rule_sha256,
        row_phase1_seal_sha256=fixture["seal"].row_phase1_seal_sha256,
        phase1_fit_physical_id_root_sha256=fixture["seal"].phase1_fit_physical_id_root_sha256,
        phase1_cell_physical_id_root_sha256=_sha("e"),
    )
    with pytest.raises(assets.NextR1AssetError, match="cell physical-ID root"):
        assets.build_next_r1_phase1_assets(
            fixture["blocks"], fixture["labels"], bad_seal, fixture["receivers"], fixture["classes"],
            fixture["cells"], fixture["loo_bindings"], _validation,
        )


def test_rejects_validation_that_only_moves_at_jitter() -> None:
    fixture = _fixture()

    def callback(block_id: str, basis: np.ndarray, coefficient: np.ndarray, phase1_labels: tuple[str, ...]):
        result = _validation(block_id, basis, coefficient, phase1_labels)
        return assets.Phase1DirectionalValidation(
            basis_sha256=result.basis_sha256,
            coefficient_sha256=result.coefficient_sha256,
            baseline_total_correct=result.baseline_total_correct,
            perturbed_total_correct=result.perturbed_total_correct,
            baseline_per_class_correct=result.baseline_per_class_correct,
            perturbed_per_class_correct=result.perturbed_per_class_correct,
            per_class_total=result.per_class_total,
            forward_action_max_abs_delta=result.repeated_forward_jitter_max_abs_delta,
            repeated_forward_jitter_max_abs_delta=result.repeated_forward_jitter_max_abs_delta,
            validation_seal_sha256=result.validation_seal_sha256,
        )

    with pytest.raises(assets.NextR1AssetSelectionError):
        assets.build_next_r1_phase1_assets(
            fixture["blocks"], fixture["labels"], fixture["seal"], fixture["receivers"], fixture["classes"],
            fixture["cells"], fixture["loo_bindings"], callback,
        )
