#!/usr/bin/env python3
"""Prepare, smoke-test and execute the frozen D92 TPCE Hard11 screen."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from cvsrffi import stage2_d92_tpce_hard11 as _matrix
from cvsrffi.stage2_d92_tpce_hard11 import *  # noqa: F401,F403

try:
    from scripts import run_d92_pareto_distill_hard11 as _base_runner
except ImportError:  # pragma: no cover
    import run_d92_pareto_distill_hard11 as _base_runner  # type: ignore[no-redef]


CODE_ROOT = _base_runner.CODE_ROOT
PREDICTION_ENTRY = _base_runner.PREDICTION_ENTRY
SCORING_ENTRY = _base_runner.SCORING_ENTRY
QUERY_ZERO_FIELDS = tuple(_base_runner.QUERY_ZERO_FIELDS)
subprocess = _base_runner.subprocess


class D92D92TPCEHard11RunnerError(RuntimeError):
    """Raised when TPCE Hard11 evidence or fit receipt would drift."""


D92TPCEHard11RunnerError = D92D92TPCEHard11RunnerError


def _is_full_matrix(manifest: Mapping[str, Any]) -> bool:
    return int(manifest.get("job_count", -1)) == 11 and isinstance(manifest.get("jobs"), list) and len(manifest["jobs"]) == 11


def _finite(value: Any, label: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise D92D92TPCEHard11RunnerError(f"fit audit {label} is not finite")
    result = float(value)
    if lower is not None and result < lower:
        raise D92D92TPCEHard11RunnerError(f"fit audit {label} is below bound")
    return result


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise D92D92TPCEHard11RunnerError(f"fit audit {label} SHA drift")
    return value.lower()


def _validate_tpce_row(row: Mapping[str, Any], *, active: bool, allow_numeric_fallback: bool = False) -> None:
    prefix = "d92_e0d_tpce_"
    required = (
        "code1_byte_exact", "scale1_byte_exact", "scale2_byte_exact", "intercept_byte_exact", "log_diag_byte_exact",
    )
    for name in required:
        if row.get(prefix + name) not in ({True, None} if not active else {True}):
            raise D92D92TPCEHard11RunnerError(f"fit audit {name} state guard drift")
    if row.get(prefix + "old_group_uniform_shift") not in ({False, None} if not active else {False}):
        raise D92D92TPCEHard11RunnerError("fit audit old-group uniform-shift guard drift")
    fallback = row.get(prefix + "fallback_active") is True
    if active and not fallback:
        if row.get(prefix + "active") is not True or row.get(prefix + "fallback_active") is not False or row.get(prefix + "fallback_reason") is not None:
            raise D92D92TPCEHard11RunnerError("fit audit TPCE active/fallback drift")
        if row.get(prefix + "state_postprocess_mode") != "d42_tpce" or row.get(prefix + "direct_state_publish") is not True or row.get(prefix + "requantize_call_count") != 0:
            raise D92D92TPCEHard11RunnerError("fit audit TPCE direct-publish guard drift")
        if row.get(prefix + "quantile") != 0.20 or row.get(prefix + "quantile_method") != "lower":
            raise D92D92TPCEHard11RunnerError("fit audit TPCE quantile drift")
        e0_sha = _require_sha(row.get(prefix + "e0_state_sha256"), "E0 state")
        final_sha = _require_sha(row.get(prefix + "final_state_sha256"), "final state")
        if e0_sha == final_sha:
            raise D92D92TPCEHard11RunnerError("fit audit TPCE code2 state did not change")
        changed = int(_finite(row.get(prefix + "changed_code2_count"), "changed_code2_count", lower=1))
        requested = int(_finite(row.get(prefix + "requested_atomic_exchange_count"), "requested_atomic_exchange_count", lower=1))
        applied = int(_finite(row.get(prefix + "applied_atomic_exchange_count"), "applied_atomic_exchange_count", lower=1))
        if changed <= 0 or requested != applied or row.get(prefix + "aggregate_saturation_count") != 0:
            raise D92D92TPCEHard11RunnerError("fit audit TPCE atomic exchange drift")
        for name in ("old_tail_count_by_class", "old_tail_gain_by_class"):
            values = row.get(prefix + name)
            if not isinstance(values, list) or len(values) != 6:
                raise D92D92TPCEHard11RunnerError(f"fit audit {name} drift")
        tol = _finite(row.get(prefix + "guard_tolerance"), "guard_tolerance", lower=0.0)
        if not all(_finite(value, "old tail gain") > tol for value in row[prefix + "old_tail_gain_by_class"]):
            raise D92D92TPCEHard11RunnerError("fit audit old-tail gain guard drift")
        if _finite(row.get(prefix + "old_tail_min_gain"), "old_tail_min_gain") <= tol or _finite(row.get(prefix + "pooled_new_cross_tail_gain"), "pooled_new_cross_tail_gain") <= tol:
            raise D92D92TPCEHard11RunnerError("fit audit strict positive tail guard drift")
        if _finite(row.get(prefix + "pooled_new_allclass_tail_gain"), "pooled_new_allclass_tail_gain") < -tol or _finite(row.get(prefix + "old_to_new_hinge_delta"), "old_to_new_hinge_delta") > tol or _finite(row.get(prefix + "new_to_old_hinge_delta"), "new_to_old_hinge_delta") > tol:
            raise D92D92TPCEHard11RunnerError("fit audit pooled/hinge guard drift")
        for name in ("support_guard_pass", "class_permutation_equivariant"):
            if row.get(prefix + name) is not True:
                raise D92D92TPCEHard11RunnerError(f"fit audit {name} drift")
        for name in ("support_score_macs_upper_bound", "support_coordinate_comparisons_upper_bound", "support_macs_upper_bound", "support_transient_bytes_upper_bound"):
            _finite(row.get(prefix + name), name, lower=0.0)
        if row.get(prefix + "persistent_state_bytes_delta") != 0 or row.get(prefix + "component_fit_count") != 0:
            raise D92D92TPCEHard11RunnerError("fit audit TPCE resource receipt drift")
    elif allow_numeric_fallback and fallback:
        reason = row.get(prefix + "fallback_reason")
        if row.get(prefix + "active") is not False or not isinstance(reason, str) or not reason:
            raise D92D92TPCEHard11RunnerError("fit audit TPCE numeric fallback identity drift")
        if _require_sha(row.get(prefix + "e0_state_sha256"), "E0 state") != _require_sha(row.get(prefix + "final_state_sha256"), "final state"):
            raise D92D92TPCEHard11RunnerError("fit audit TPCE numeric fallback state drift")
        for name in ("changed_code2_count", "applied_atomic_exchange_count"):
            if row.get(prefix + name) != 0:
                raise D92D92TPCEHard11RunnerError("fit audit TPCE numeric fallback applied-update drift")
        _finite(row.get(prefix + "requested_atomic_exchange_count"), "requested_atomic_exchange_count", lower=0.0)
        saturation = _finite(row.get(prefix + "aggregate_saturation_count"), "aggregate_saturation_count", lower=0.0)
        if reason == "aggregate_saturation" and saturation <= 0:
            raise D92D92TPCEHard11RunnerError("fit audit TPCE saturation fallback diagnostic drift")
        return
    else:
        if row.get(prefix + "active") is not False or row.get(prefix + "fallback_active") is not False or row.get(prefix + "fallback_reason") != "K1_K2_EXACT_D92_FULL_ALIAS":
            raise D92D92TPCEHard11RunnerError("fit audit TPCE K1/K2 alias drift")
        if _require_sha(row.get(prefix + "e0_state_sha256"), "E0 state") != _require_sha(row.get(prefix + "final_state_sha256"), "final state"):
            raise D92D92TPCEHard11RunnerError("fit audit alias state drift")
        for name in ("changed_code2_count", "requested_atomic_exchange_count", "applied_atomic_exchange_count", "aggregate_saturation_count", "component_fit_count"):
            if row.get(prefix + name) != 0:
                raise D92D92TPCEHard11RunnerError("fit audit alias count drift")


def _validate_fit_audit(path: str | Path, *, k_shot: int) -> None:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D92D92TPCEHard11RunnerError(f"fit audit is missing: {source}")
    try:
        rows = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92D92TPCEHard11RunnerError(f"fit audit is invalid: {source}") from error
    if not isinstance(rows, list) or len(rows) != 3 or {str(row.get("scenario")) for row in rows if isinstance(row, Mapping)} != set(SCENES):
        raise D92D92TPCEHard11RunnerError("fit audit scene closure drift")
    active = int(k_shot) > 2
    expected = (2, 1, "full_only", "d42_tpce") if active else (3, 3, "full_only", None)
    for row in rows:
        if not isinstance(row, Mapping) or row.get("arm_id") != ARM_ID or row.get("candidate_id") != CANDIDATE_ID:
            raise D92D92TPCEHard11RunnerError("fit audit arm/candidate identity drift")
        if any(row.get(field) is not False for field in QUERY_ZERO_FIELDS):
            raise D92D92TPCEHard11RunnerError("fit audit query access is not zero")
        inventory = row.get("after_actual_component_inventory", {})
        actual = inventory.get("actual_component_fit_count", row.get("actual_fit_count", -1)) if isinstance(inventory, Mapping) else -1
        total = row.get("after_total_component_fit_count", row.get("fit_count", -1))
        mode = row.get("after_registered_d_mode_effective", row.get("registered_d_mode", ""))
        post = row.get("after_state_postprocess_mode", row.get("state_postprocess_mode"))
        try:
            observed = (int(total), int(actual), str(mode), post)
        except (TypeError, ValueError) as error:
            raise D92D92TPCEHard11RunnerError("fit audit inventory is invalid") from error
        if observed != expected:
            raise D92D92TPCEHard11RunnerError("fit audit K/mode inventory drift")
        _validate_tpce_row(row, active=active, allow_numeric_fallback=active)


def _verify_manifest_artifacts(manifest: Mapping[str, Any]) -> None:
    return _base_runner._verify_manifest_artifacts(manifest)


@contextmanager
def _runner_context() -> Iterator[None]:
    old = {name: getattr(_base_runner, name) for name in ("ARM_ID", "CANDIDATE_ID", "CANONICAL_SELECTION_SHA256", "LIVENESS_OUTER_KEY", "SCENES", "SHARD_COUNT", "SMOKE_OUTER_KEY", "build_hard11_manifest", "validate_hard11_manifest", "validate_method_lock", "_validate_fit_audit", "subprocess", "_verify_manifest_artifacts")}
    with _matrix._base_context(disable_validation=False):
        for name, value in {"ARM_ID": ARM_ID, "CANDIDATE_ID": CANDIDATE_ID, "CANONICAL_SELECTION_SHA256": CANONICAL_SELECTION_SHA256, "LIVENESS_OUTER_KEY": LIVENESS_OUTER_KEY, "SCENES": SCENES, "SHARD_COUNT": SHARD_COUNT, "SMOKE_OUTER_KEY": SMOKE_OUTER_KEY, "build_hard11_manifest": build_hard11_manifest, "validate_hard11_manifest": validate_hard11_manifest, "validate_method_lock": validate_method_lock, "_validate_fit_audit": _validate_fit_audit, "subprocess": subprocess, "_verify_manifest_artifacts": _verify_manifest_artifacts}.items():
            setattr(_base_runner, name, value)
        try:
            yield
        finally:
            for name, value in old.items():
                setattr(_base_runner, name, value)


def _rewrite_schema(path: Path, *, to_tpce: bool) -> None:
    if not path.is_file() or path.is_symlink():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict):
        return
    old, new = ("d92_pareto_distill_hard11", "d92_tpce_hard11") if to_tpce else ("d92_tpce_hard11", "d92_pareto_distill_hard11")
    changed = False
    for key in ("schema", "status"):
        value = payload.get(key)
        if isinstance(value, str) and (old in value or (to_tpce and "PARETO_DISTILL" in value) or (not to_tpce and "TPCE" in value)):
            payload[key] = value.replace(old, new)
            payload[key] = payload[key].replace("PARETO_DISTILL", "TPCE" if to_tpce else "PARETO_DISTILL")
            payload[key] = payload[key].replace("TPCE", "TPCE" if to_tpce else "PARETO_DISTILL")
            changed = True
    if not changed:
        return
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(path, stat.S_IREAD)
    except OSError as error:
        raise D92D92TPCEHard11RunnerError(f"cannot rewrite TPCE receipt: {path}") from error


def _rewrite_output(root: Path, *, to_tpce: bool) -> None:
    if root.is_dir():
        for path in root.rglob("*.json"):
            _rewrite_schema(path, to_tpce=to_tpce)


def build_hard11_manifest(**kwargs: Any) -> dict[str, Any]:
    return _matrix.build_hard11_manifest(**kwargs)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    with _runner_context():
        result = _base_runner.prepare(args)
    result["status"] = "TPCE_HARD11_MATRIX_PREPARED"
    _rewrite_output(Path(args.output_root), to_tpce=True)
    return result


def truth_free_smoke(args: argparse.Namespace) -> dict[str, Any]:
    with _runner_context():
        result = _base_runner.truth_free_smoke(args)
    _rewrite_output(Path(args.output_root), to_tpce=True)
    result["status"] = "D92_TPCE_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
    result["schema"] = "cvs.phase2.d92_tpce_hard11.smoke_receipt.v1"
    return result


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.matrix_manifest).resolve().parent
    manifest_root: Path | None = None
    try:
        manifest_root = Path(json.loads(Path(args.matrix_manifest).read_text(encoding="utf-8-sig"))["output_root"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    smoke = manifest_root / "smoke" / "smoke_receipt.json" if manifest_root else None
    if smoke is not None:
        _rewrite_schema(smoke, to_tpce=False)
    with _runner_context():
        result = _base_runner.run_shard(args)
    if manifest_root:
        _rewrite_output(manifest_root, to_tpce=True)
    result["schema"] = "cvs.phase2.d92_tpce_hard11.shard_summary.v1"
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare"); p.add_argument("--context-manifest", required=True); p.add_argument("--method-lock", required=True); p.add_argument("--output-root", required=True)
    p = commands.add_parser("truth-free-smoke", aliases=["smoke"]); p.add_argument("--matrix-manifest", required=True); p.add_argument("--matrix-manifest-sha256", required=True); p.add_argument("--output-root", required=True); p.add_argument("--device", required=True); p.add_argument("--cpu-threads", type=int, default=2)
    p = commands.add_parser("run-shard"); p.add_argument("--matrix-manifest", required=True); p.add_argument("--matrix-manifest-sha256", required=True); p.add_argument("--shard-index", type=int, required=True); p.add_argument("--shard-count", type=int, choices=(SHARD_COUNT,), default=SHARD_COUNT); p.add_argument("--device", required=True); p.add_argument("--cpu-threads", type=int, default=2)
    return result


def main() -> int:
    args = parser().parse_args()
    value = prepare(args) if args.command == "prepare" else truth_free_smoke(args) if args.command in {"truth-free-smoke", "smoke"} else run_shard(args)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return 0 if value["status"] in {"TPCE_HARD11_MATRIX_PREPARED", "D92_TPCE_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
