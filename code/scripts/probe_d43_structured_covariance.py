#!/usr/bin/env python3
"""Diagnostic D43 covariance-structure probe over the locked D42 runner.

The probe changes only the shared LDA covariance structure and removes the
class-common affine score before D42's existing int8/FP16 compiler.  It is not
a formal candidate or a replacement runner; each arm writes a separate D42
development artifact plus explicit probe metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


ARM_STRUCTURES = {
    "full_centered_control": "full_auto_shrinkage_control",
    "block3_centered": "three_block_z160_fft96_rf32",
    "diagonal_centered": "diagonal_auto_shrinkage_variance",
}
ARMS = tuple(ARM_STRUCTURES)
_BOOTSTRAPPED = False
ALLOW_FP32_CENTERING_ARGMAX_DRIFT = False
FP32_CENTERING_ARGMAX_AUDIT: list[dict[str, Any]] = []
RUNTIME_LEGACY_SHA256 = (
    "f4becfa61f23bd88448e9a9673f9068cdf5d5e613999992de4b9f35cc29fefa9"
)
RUNTIME_MODULE_SHA256 = {
    "cvsrffi": "db0119aff842e1af0991535c9681b59ac404950dbbae3772487aa74ec0fc9c4d",
    "cvsrffi.phase1_int8_prototype_bundle": (
        "50deb1f18203d40aeaea1976d5dc146af5157bfcb1f96ded1f5e835ee1401623"
    ),
    "cvsrffi.phase2_runtime_contract": (
        "ec2351da61cd18c5a177575427efd57cd84c5ad61c7fb921306874b50da0bcca"
    ),
    "cvsrffi.somph_diagnostic_bundle_loader": (
        "2256f4206f6b0078e4a78b3393cbd638419ab0ac5b91ec2d033c9b654cbe1774"
    ),
    "cvsrffi.somph_predictor_bundle": (
        "49a05c6f1f809fc221e3cb64fffe0c2f11b1b252e6cdbe86449303f8fb5def48"
    ),
    "cvsrffi.somph_predictor_runtime": (
        "dfc2e57ec9055000fc55ebd5334ac947dcf935578ebb779c2839c93fa219e64f"
    ),
    "cvsrffi.somph_runtime_trust": (
        "4b1dee1d8ffdc793f48c46c21a11b0fdf8b6ef6e3b253807cc1138011dc1f9fc"
    ),
    "cvsrffi.stage2_ciaf": (
        "974bdf62f8eaf1cfaa3c919d3d9292d8b5033955fffc97b16b3eec8d3db3fabf"
    ),
    "cvsrffi.stage2_dali": (
        "41579a0adb9773e7fb2fbd36b6eee70814f74aa886dc1078ffdcab80711ac179"
    ),
    "cvsrffi.stage2_diag_cosine_exploration": (
        "6e33fe992415b7d113d4eab1898d2d66929adf01066fb0458165983771eff2de"
    ),
    "cvsrffi.stage2_predictor_bundle": (
        "bb27beaa94c4245b2135b5493e1be305985e05ff9f88c01bc0b9f60955944aa9"
    ),
    "cvsrffi.stage2_predictor_runtime": (
        "58d7e3aadb7ffeec7f5a88fb4d2cc34e2fd70a105e28e95b30e1cada58cf5747"
    ),
}


class D43ProbeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _structured_covariance(
    covariance: np.ndarray, arm: str, block_slices: tuple[slice, ...]
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise D43ProbeError("D43 covariance must be square")
    if arm == "full_centered_control":
        structured = matrix.copy()
    elif arm == "block3_centered":
        structured = np.zeros_like(matrix)
        for block in block_slices:
            structured[block, block] = matrix[block, block]
    elif arm == "diagonal_centered":
        structured = np.diag(np.diag(matrix))
    else:
        raise D43ProbeError(f"unsupported D43 arm: {arm}")
    if not np.isfinite(structured).all():
        raise D43ProbeError("D43 structured covariance became non-finite")
    return structured


def _center_affine_scores(
    coefficients: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coef = np.asarray(coefficients, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    if coef.ndim != 2 or bias.shape != (coef.shape[0],):
        raise D43ProbeError("D43 affine score shape drift")
    return coef - coef.mean(axis=0, keepdims=True), bias - bias.mean()


def build_structured_fit(
    d42: Any, arm: str
) -> Callable[[np.ndarray, np.ndarray, int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Build the monkeypatched D42 fit used only by this diagnostic probe."""

    if arm not in ARMS:
        raise D43ProbeError(f"unsupported D43 arm: {arm}")
    original_fit = d42._fit_equal_prior_lda

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rows = np.asarray(transformed, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.int64)
        means = np.stack(
            [rows[labels == index].mean(axis=0) for index in range(class_count)]
        )
        residuals = rows - means[labels]
        residual_energy = float(np.sum(residuals**2))
        residual_rank = int(np.linalg.matrix_rank(residuals))
        fallback = bool(
            int(k_shot) <= 2
            or residual_rank == 0
            or not np.isfinite(residual_energy)
            or residual_energy <= float(d42.ENERGY_EPSILON)
        )
        if fallback:
            coefficients, intercept, audit = original_fit(
                transformed, targets, class_count, k_shot
            )
            centered_coef, centered_intercept = _center_affine_scores(
                coefficients, intercept
            )
            centered_coef32 = centered_coef.astype(np.float32)
            centered_intercept32 = centered_intercept.astype(np.float32)
            original_scores32 = (
                rows.astype(np.float32) @ np.asarray(coefficients, dtype=np.float32).T
                + np.asarray(intercept, dtype=np.float32)[None, :]
            )
            centered_scores32 = (
                rows.astype(np.float32) @ centered_coef32.T
                + centered_intercept32[None, :]
            )
            fp32_argmax_equivalent = bool(np.array_equal(
                np.argmax(original_scores32, axis=1),
                np.argmax(centered_scores32, axis=1),
            ))
            fp32_argmax_changed_count = int(np.sum(
                np.argmax(original_scores32, axis=1)
                != np.argmax(centered_scores32, axis=1)
            ))
            FP32_CENTERING_ARGMAX_AUDIT.append(
                {
                    "arm": arm,
                    "fallback": True,
                    "support_rows": int(len(rows)),
                    "changed_count": fp32_argmax_changed_count,
                    "equivalent": fp32_argmax_equivalent,
                    "drift_allowed": ALLOW_FP32_CENTERING_ARGMAX_DRIFT,
                }
            )
            if not fp32_argmax_equivalent and not ALLOW_FP32_CENTERING_ARGMAX_DRIFT:
                raise D43ProbeError("D43 fallback FP32 centering changed support argmax")
            result_audit = dict(audit)
            result_audit.update(
                {
                    "d43_probe_arm": arm,
                    "d43_covariance_structure": "unit_fallback",
                    "d43_k_le2_unit_covariance_fallback": int(k_shot) <= 2,
                    "d43_class_common_affine_omitted": True,
                    "d43_centered_score_fp64_algebraically_equivalent": True,
                    "d43_centered_support_fp32_argmax_equivalent": fp32_argmax_equivalent,
                    "d43_centered_support_fp32_argmax_changed_count": (
                        fp32_argmax_changed_count
                    ),
                    "d43_centered_support_fp32_argmax_drift_allowed": (
                        ALLOW_FP32_CENTERING_ARGMAX_DRIFT
                    ),
                    "d43_centered_support_fp32_pairwise_drift_max": float(
                        np.max(
                            np.abs(
                                (original_scores32[:, :, None] - original_scores32[:, None, :])
                                - (centered_scores32[:, :, None] - centered_scores32[:, None, :])
                            )
                        )
                    ),
                    "d43_centered_coefficient_mean_max_abs": float(
                        np.max(np.abs(centered_coef.mean(axis=0)))
                    ),
                    "d43_centered_intercept_mean_abs": float(
                        abs(centered_intercept.mean())
                    ),
                }
            )
            return (
                centered_coef32,
                centered_intercept32,
                result_audit,
            )

        priors = np.full(class_count, 1.0 / class_count, dtype=np.float64)
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=priors, store_covariance=True
        )
        estimator.fit(rows, labels)
        expected_classes = np.arange(class_count, dtype=np.int64)
        if not np.array_equal(np.asarray(estimator.classes_), expected_classes):
            raise D43ProbeError("D43 sklearn class order drift")
        fitted_means = np.asarray(estimator.means_, dtype=np.float64)
        covariance = _structured_covariance(
            np.asarray(estimator.covariance_, dtype=np.float64),
            arm,
            tuple(d42.BLOCK_SLICES),
        )
        coefficients = np.linalg.lstsq(covariance, fitted_means.T, rcond=None)[0].T
        intercept = -0.5 * np.diag(fitted_means @ coefficients.T) + np.log(priors)
        uncentered_predictions = np.argmax(
            rows @ coefficients.T + intercept[None, :], axis=1
        )
        centered_coef, centered_intercept = _center_affine_scores(
            coefficients, intercept
        )
        centered_predictions = np.argmax(
            rows @ centered_coef.T + centered_intercept[None, :], axis=1
        )
        if not np.array_equal(uncentered_predictions, centered_predictions):
            raise D43ProbeError("D43 common-score removal changed argmax")
        sklearn_equivalent = None
        if arm == "full_centered_control":
            sklearn_equivalent = bool(
                np.array_equal(
                    centered_predictions,
                    np.asarray(estimator.predict(rows), dtype=np.int64),
                )
            )
            if not sklearn_equivalent:
                raise D43ProbeError("D43 full control drifted from D42 sklearn fit")
        uncentered_coef32 = coefficients.astype(np.float32)
        uncentered_intercept32 = intercept.astype(np.float32)
        centered_coef32 = centered_coef.astype(np.float32)
        centered_intercept32 = centered_intercept.astype(np.float32)
        rows32 = rows.astype(np.float32)
        uncentered_scores32 = (
            rows32 @ uncentered_coef32.T + uncentered_intercept32[None, :]
        )
        centered_scores32 = rows32 @ centered_coef32.T + centered_intercept32[None, :]
        fp32_argmax_equivalent = bool(np.array_equal(
            np.argmax(uncentered_scores32, axis=1),
            np.argmax(centered_scores32, axis=1),
        ))
        fp32_argmax_changed_count = int(np.sum(
            np.argmax(uncentered_scores32, axis=1)
            != np.argmax(centered_scores32, axis=1)
        ))
        FP32_CENTERING_ARGMAX_AUDIT.append(
            {
                "arm": arm,
                "fallback": False,
                "support_rows": int(len(rows)),
                "changed_count": fp32_argmax_changed_count,
                "equivalent": fp32_argmax_equivalent,
                "drift_allowed": ALLOW_FP32_CENTERING_ARGMAX_DRIFT,
            }
        )
        if not fp32_argmax_equivalent and not ALLOW_FP32_CENTERING_ARGMAX_DRIFT:
            raise D43ProbeError("D43 FP32 centering changed support argmax")
        fp32_pairwise_drift = float(
            np.max(
                np.abs(
                    (
                        uncentered_scores32[:, :, None]
                        - uncentered_scores32[:, None, :]
                    )
                    - (
                        centered_scores32[:, :, None]
                        - centered_scores32[:, None, :]
                    )
                )
            )
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(np.min(eigenvalues)) <= 0.0:
            raise D43ProbeError("D43 structured covariance is not positive definite")
        structure_name = ARM_STRUCTURES[arm]
        audit = {
            "solver": "lsqr",
            "shrinkage": "auto",
            "prior_policy": "equal_1_over_registered_class_count",
            # Keep the D42 state schema valid; the explicit D43 fields below
            # carry the diagnostic covariance intervention.
            "covariance_policy": "sklearn_lsqr_auto_shrinkage_equal_prior",
            "unit_covariance_fallback": False,
            "within_class_residual_rank": residual_rank,
            "within_class_residual_energy": residual_energy,
            "support_rows": int(len(rows)),
            "class_count": int(class_count),
            "k_shot": int(k_shot),
            "coefficient_source": (
                "d43_structured_sigma_inverse_mu_then_class_common_affine_omitted"
            ),
            "covariance_equation_residual_max": float(
                np.max(np.abs(covariance @ coefficients.T - fitted_means.T))
            ),
            "sklearn_prediction_equivalent": sklearn_equivalent,
            "d43_probe_arm": arm,
            "d43_covariance_structure": structure_name,
            "d43_class_common_affine_omitted": True,
            "d43_centered_score_fp64_algebraically_equivalent": True,
            "d43_centered_support_fp32_argmax_equivalent": fp32_argmax_equivalent,
            "d43_centered_support_fp32_argmax_changed_count": (
                fp32_argmax_changed_count
            ),
            "d43_centered_support_fp32_argmax_drift_allowed": (
                ALLOW_FP32_CENTERING_ARGMAX_DRIFT
            ),
            "d43_centered_support_fp32_pairwise_drift_max": fp32_pairwise_drift,
            "d43_centered_coefficient_mean_max_abs": float(
                np.max(np.abs(centered_coef.mean(axis=0)))
            ),
            "d43_centered_intercept_mean_abs": float(abs(centered_intercept.mean())),
            "d43_covariance_eigenvalue_min": float(np.min(eigenvalues)),
            "d43_covariance_eigenvalue_max": float(np.max(eigenvalues)),
            "d43_covariance_condition_number": float(
                np.max(eigenvalues) / np.min(eigenvalues)
            ),
        }
        return (
            centered_coef32,
            centered_intercept32,
            audit,
        )

    return fit


def _bootstrap(runtime_root: Path, probe_root: Path) -> tuple[Any, Any, tuple[str, ...]]:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        raise D43ProbeError("D43 bootstrap is single-use; launch one process per arm")
    _BOOTSTRAPPED = True
    runtime_root = runtime_root.resolve()
    probe_root = probe_root.resolve()
    runtime_code = runtime_root / "code"
    runtime_scripts = runtime_code / "scripts"
    runtime_package = runtime_code / "cvsrffi"
    probe_package = probe_root / "code" / "cvsrffi"
    runner = probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
    for required in (runtime_scripts, runtime_package, probe_package, runner):
        if not required.exists():
            raise D43ProbeError(f"missing D43 bootstrap path: {required}")
    legacy_path = runtime_scripts / "run_d19_support_only_ciaf.py"
    if _sha256(legacy_path) != RUNTIME_LEGACY_SHA256:
        raise D43ProbeError("D43 locked run_d19 runtime hash drift")
    for module_name, expected in RUNTIME_MODULE_SHA256.items():
        file_name = "__init__.py" if module_name == "cvsrffi" else f"{module_name.rsplit('.', 1)[1]}.py"
        actual = _sha256(runtime_package / file_name)
        if actual != expected:
            raise D43ProbeError(
                f"D43 runtime hash drift for {module_name}: expected {expected}, got {actual}"
            )
    sys.path.insert(0, str(runtime_scripts))
    sys.path.insert(0, str(runtime_code))
    legacy = importlib.import_module("run_d19_support_only_ciaf")
    if (
        Path(legacy.__file__).resolve() != legacy_path.resolve()
        or _sha256(Path(legacy.__file__).resolve()) != RUNTIME_LEGACY_SHA256
    ):
        raise D43ProbeError("D43 loaded run_d19 from an unverified path")
    package = importlib.import_module("cvsrffi")
    original_package_path = tuple(str(value) for value in package.__path__)
    loaded_runtime_names = {
        name
        for name, module in sys.modules.items()
        if name.startswith("cvsrffi") and getattr(module, "__file__", None)
    }
    if loaded_runtime_names != set(RUNTIME_MODULE_SHA256):
        raise D43ProbeError(
            "D43 run_d19 preloaded runtime-module closure drift: "
            f"{sorted(loaded_runtime_names)}"
        )
    for module_name, expected_hash in RUNTIME_MODULE_SHA256.items():
        module = sys.modules[module_name]
        loaded_path = Path(module.__file__).resolve()
        file_name = "__init__.py" if module_name == "cvsrffi" else f"{module_name.rsplit('.', 1)[1]}.py"
        expected_path = (runtime_package / file_name).resolve()
        if loaded_path != expected_path or _sha256(loaded_path) != expected_hash:
            raise D43ProbeError(f"D43 loaded unverified runtime module: {module_name}")
    package_path = str(probe_package)
    if package_path not in package.__path__:
        package.__path__.insert(0, package_path)
    sys.path.insert(0, str(probe_root / "code" / "scripts"))
    sys.path.insert(0, str(probe_root / "code"))
    d42 = importlib.import_module("cvsrffi.stage2_d42_unified_shrinkage_lda")
    expected_d42 = (probe_package / "stage2_d42_unified_shrinkage_lda.py").resolve()
    if Path(d42.__file__).resolve() != expected_d42:
        raise D43ProbeError("D43 loaded the D42 core from an unverified path")
    return d42, package, original_package_path


def _argument_value(arguments: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise D43ProbeError(f"D43 runner arguments require exactly one {name}")
    return str(arguments[positions[0] + 1])


def _runner_output(arguments: list[str]) -> Path:
    return Path(_argument_value(arguments, "--output")).resolve()


def _require_locked_runner_arguments(arguments: list[str]) -> None:
    if _argument_value(arguments, "--candidate-set") != "d42_v1":
        raise D43ProbeError("D43 probe requires --candidate-set d42_v1")
    if (
        _argument_value(arguments, "--mode")
        != "development_select_unverified_component"
    ):
        raise D43ProbeError("D43 probe requires the locked development mode")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise D43ProbeError(f"D43 expected a JSON object: {path}")
    return value


def _verify_probe_output(
    output: Path, arm: str, probe_script_sha256: str
) -> dict[str, Any]:
    receipt_path = output / "RECEIPT.json"
    receipt = _read_json(receipt_path)
    required_receipt = {
        "candidate_set": "d42_v1",
        "candidate_count": 7,
        "folds_per_candidate": 15,
        "mode": "development_select_unverified_component",
        "query_opened": False,
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "selected_candidate_id": "Z0_SUPPORT_ONLY",
        "pre_full_k10_selected_candidate_id": "Z0_SUPPORT_ONLY",
        "selected_positive_route": False,
        "training_log_row_count": 105,
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE",
        "source_closure_unchanged_after_support": True,
        "support_query_disjointness_status": "SUPPORT_ONLY_NO_QUERY_CLAIM",
        "d19_control_helper_sha256": RUNTIME_LEGACY_SHA256,
        "ciaf_sha256": RUNTIME_MODULE_SHA256["cvsrffi.stage2_ciaf"],
        "diag_cosine_feature_operator_sha256": RUNTIME_MODULE_SHA256[
            "cvsrffi.stage2_diag_cosine_exploration"
        ],
    }
    for name, expected in required_receipt.items():
        if receipt.get(name) != expected:
            raise D43ProbeError(
                f"D43 base receipt drift for {name}: expected {expected!r}, "
                f"got {receipt.get(name)!r}"
            )
    hash_fields = {
        "training_log.jsonl": "training_log_sha256",
        "support_audit.json": "support_audit_sha256",
        "selection.json": "selection_sha256",
        "resource_audit.json": "resource_audit_sha256",
        "geometry_audit.json": "geometry_audit_sha256",
    }
    artifact_hashes: dict[str, str] = {}
    for file_name, field in hash_fields.items():
        actual = _sha256(output / file_name)
        if actual != str(receipt.get(field, "")):
            raise D43ProbeError(f"D43 receipt hash mismatch for {file_name}")
        artifact_hashes[file_name] = actual
    support = _read_json(output / "support_audit.json")
    if (
        support.get("query_opened") is not False
        or int(support.get("query_rows_opened", -1)) != 0
        or int(support.get("query_labels_opened", -1)) != 0
        or support.get("formal_metric_claim_allowed") is not False
        or support.get("performance_claim_allowed") is not False
    ):
        raise D43ProbeError("D43 support/query claim boundary drift")
    candidate_lock = support.get("candidate_lock")
    if not isinstance(candidate_lock, dict):
        raise D43ProbeError("D43 composite candidate lock missing")
    source_closure = candidate_lock.get("source_closure")
    probe_lock = candidate_lock.get("d43_probe_lock")
    lock_payload = dict(candidate_lock)
    stated_lock_sha256 = str(lock_payload.pop("sha256", ""))
    recomputed_lock_sha256 = hashlib.sha256(
        _canonical_bytes(lock_payload)
    ).hexdigest()
    if (
        not isinstance(source_closure, dict)
        or source_closure.get("d19_control_helper_sha256")
        != RUNTIME_LEGACY_SHA256
        or source_closure.get("ciaf_sha256")
        != RUNTIME_MODULE_SHA256["cvsrffi.stage2_ciaf"]
        or source_closure.get("diag_cosine_feature_operator_sha256")
        != RUNTIME_MODULE_SHA256["cvsrffi.stage2_diag_cosine_exploration"]
        or source_closure.get("d43_probe_script_sha256") != probe_script_sha256
        or source_closure.get("d43_runtime_legacy_sha256")
        != RUNTIME_LEGACY_SHA256
        or source_closure.get("d43_preloaded_runtime_module_sha256")
        != RUNTIME_MODULE_SHA256
        or not isinstance(probe_lock, dict)
        or probe_lock.get("arm") != arm
        or probe_lock.get("formal_candidate") is not False
        or probe_lock.get("forced_nonpromotable") is not True
        or probe_lock.get("selected_only_full_k10_refit_allowed") is not False
        or stated_lock_sha256 != recomputed_lock_sha256
        or stated_lock_sha256 != receipt.get("candidate_lock_sha256")
    ):
        raise D43ProbeError("D43 composite source/candidate lock drift")
    selection = _read_json(output / "selection.json")
    if (
        selection.get("selected_candidate_id") != "Z0_SUPPORT_ONLY"
        or selection.get("pre_full_k10_selected_candidate_id") != "Z0_SUPPORT_ONLY"
        or selection.get("selected_positive_route") is not False
        or selection.get("candidate_lock_sha256") != stated_lock_sha256
    ):
        raise D43ProbeError("D43 diagnostic selection guard drift")
    decisions = selection.get("candidate_decisions")
    if not isinstance(decisions, list):
        raise D43ProbeError("D43 candidate decision evidence missing")
    d42_decision = next(
        (
            row
            for row in decisions
            if isinstance(row, dict) and row.get("candidate_id") == "D42-USLDA-INT8"
        ),
        None,
    )
    if (
        not isinstance(d42_decision, dict)
        or d42_decision.get("eligible_positive_route") is not False
        or d42_decision.get("d43_probe_forced_nonpromotable") is not True
    ):
        raise D43ProbeError("D43 nonpromotable decision evidence missing")
    training_rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    d43_rows = [
        row
        for row in training_rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(training_rows) != 105 or len(d43_rows) != 30:
        raise D43ProbeError("D43 training-row closure drift")
    expected_structure = ARM_STRUCTURES[arm]
    for row in d43_rows:
        if row.get("query_opened") is not False:
            raise D43ProbeError("D43 row opened query")
        geometry = row.get("geometry_summary", {})
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = geometry.get(field, {})
            if (
                audit.get("d43_probe_arm") != arm
                or audit.get("d43_covariance_structure") != expected_structure
                or audit.get("d43_class_common_affine_omitted") is not True
            ):
                raise D43ProbeError(f"D43 fit audit missing from {field}")
    return {
        "base_runner_receipt_sha256": _sha256(receipt_path),
        "base_artifact_sha256": artifact_hashes,
        "verified_training_row_count": len(training_rows),
        "verified_d43_fit_row_count": len(d43_rows),
        "verified_covariance_structure": expected_structure,
        "verified_query_opened": False,
        "verified_forced_nonpromotable": True,
    }


def _install_runner_probe_guards(
    runner: Any,
    *,
    arm: str,
    probe_script_sha256: str,
    extra_source_closure: dict[str, Any] | None = None,
) -> None:
    """Force diagnostic-only selection and prohibit selected-only refits."""

    original_select = runner._select_d42_candidate
    original_candidate_lock = runner._candidate_lock

    def candidate_lock(candidates: Any, candidate_set: str = "d25_v4") -> dict[str, Any]:
        lock = dict(original_candidate_lock(candidates, candidate_set))
        lock.pop("sha256", None)
        source_closure = dict(lock.get("source_closure", {}))
        source_closure.update(
            {
                "ciaf_sha256": RUNTIME_MODULE_SHA256["cvsrffi.stage2_ciaf"],
                "d19_control_helper_sha256": RUNTIME_LEGACY_SHA256,
                "diag_cosine_feature_operator_sha256": RUNTIME_MODULE_SHA256[
                    "cvsrffi.stage2_diag_cosine_exploration"
                ],
                "d43_probe_script_sha256": probe_script_sha256,
                "d43_runtime_legacy_sha256": RUNTIME_LEGACY_SHA256,
                "d43_preloaded_runtime_module_sha256": dict(
                    RUNTIME_MODULE_SHA256
                ),
            }
        )
        if extra_source_closure:
            extra = dict(extra_source_closure)
            collisions = sorted(set(source_closure).intersection(extra))
            if collisions:
                raise D43ProbeError(
                    "D43 extra source closure overwrites reserved keys: "
                    + ", ".join(collisions)
                )
            source_closure.update(extra)
        lock["source_closure"] = source_closure
        lock["d43_probe_lock"] = {
            "arm": arm,
            "formal_candidate": False,
            "forced_nonpromotable": True,
            "selected_only_full_k10_refit_allowed": False,
        }
        return {
            **lock,
            "sha256": hashlib.sha256(_canonical_bytes(lock)).hexdigest(),
        }

    def select_diagnostic_only(folds_by_candidate: Any) -> tuple[str, list[dict[str, Any]]]:
        _selected, decisions = original_select(folds_by_candidate)
        guarded: list[dict[str, Any]] = []
        for row in decisions:
            record = dict(row)
            record["d43_probe_pre_guard_eligible_positive_route"] = bool(
                record.get("eligible_positive_route", False)
            )
            record["eligible_positive_route"] = False
            record["d43_probe_forced_nonpromotable"] = True
            guarded.append(record)
        return runner.IDENTITY_CANDIDATE, guarded

    runner._select_d42_candidate = select_diagnostic_only
    runner._full_state_refit_required = lambda *_args, **_kwargs: False
    runner._candidate_lock = candidate_lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d43-arm", required=True, choices=ARMS)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    _require_locked_runner_arguments(runner_arguments)
    output = _runner_output(runner_arguments)
    if output.exists():
        raise D43ProbeError(f"D43 output already exists: {output}")
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_package_path: tuple[str, ...] = ()
    original_fit = None
    runner_module = None
    runner_module_name = "d43_locked_d42_runner"
    probe_script_sha256 = _sha256(Path(__file__).resolve())
    try:
        d42, package, original_package_path = _bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_structured_fit(d42, known.d43_arm)
        runner = (
            known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        )
        spec = importlib.util.spec_from_file_location(runner_module_name, runner)
        if spec is None or spec.loader is None:
            raise D43ProbeError("D43 could not load the locked D42 runner")
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        spec.loader.exec_module(runner_module)
        _install_runner_probe_guards(
            runner_module,
            arm=known.d43_arm,
            probe_script_sha256=probe_script_sha256,
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
        if d42 is not None and original_fit is not None:
            d42._fit_equal_prior_lda = original_fit
        if package is not None:
            package.__path__[:] = list(original_package_path)
        sys.modules.pop(runner_module_name, None)
    if exit_code != 0:
        return exit_code
    receipt = output / "RECEIPT.json"
    if not receipt.is_file():
        raise D43ProbeError("D43 base runner completed without RECEIPT.json")
    evidence = _verify_probe_output(
        output, known.d43_arm, probe_script_sha256
    )
    metadata = {
        "schema": "cvs.phase2.d43.structured_covariance_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d43_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": probe_script_sha256,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        "runtime_legacy_sha256": RUNTIME_LEGACY_SHA256,
        "runtime_module_sha256": dict(RUNTIME_MODULE_SHA256),
        **evidence,
    }
    (output / "D43_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
