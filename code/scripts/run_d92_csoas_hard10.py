#!/usr/bin/env python3
"""Prepare, smoke-test and execute the frozen D92 CSOAS Hard9+K1 screen."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d92_csoas_hard10 as _matrix  # noqa: E402
from cvsrffi.stage2_d92_csoas_hard10 import *  # noqa: F401,F403,E402

try:
    from scripts import run_d92_pareto_distill_hard11 as _base_runner  # noqa: E402
except ImportError:  # pragma: no cover
    import run_d92_pareto_distill_hard11 as _base_runner  # type: ignore[no-redef]

PREDICTION_ENTRY = _base_runner.PREDICTION_ENTRY
SCORING_ENTRY = _base_runner.SCORING_ENTRY
subprocess = _base_runner.subprocess
QUERY_ZERO_FIELDS = tuple(_base_runner.QUERY_ZERO_FIELDS)
CSOAS_QUERY_ZERO_FIELDS = tuple(_matrix.CSOAS_QUERY_ZERO_FIELDS)


class D92CSOASHard10RunnerError(RuntimeError):
    """Raised when CSOAS Hard9 evidence or fit receipt drifts."""


D92CSOASHard10Error = D92CSOASHard10RunnerError
D92CSOASHard10RunnerErrorAlias = D92CSOASHard10RunnerError


def _is_full_matrix(manifest: Mapping[str, Any]) -> bool:
    return int(manifest.get("job_count", -1)) == 10 and isinstance(manifest.get("jobs"), list) and len(manifest["jobs"]) == 10


def _finite(value: Any, label: str, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise D92CSOASHard10RunnerError(f"fit audit {label} is not finite")
    result = float(value)
    if lower is not None and result < lower:
        raise D92CSOASHard10RunnerError(f"fit audit {label} is below bound")
    return result


def _validate_fit_audit(path: str | Path, *, k_shot: int) -> None:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D92CSOASHard10RunnerError(f"fit audit is missing: {source}")
    try:
        rows = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92CSOASHard10RunnerError(f"fit audit is invalid: {source}") from error
    if not isinstance(rows, list) or len(rows) != len(SCENES) or any(not isinstance(row, Mapping) for row in rows):
        raise D92CSOASHard10RunnerError("fit audit scene closure drift")
    if {str(row.get("scenario")) for row in rows} != set(SCENES):
        raise D92CSOASHard10RunnerError("fit audit scene identity drift")
    active = int(k_shot) > 2
    for row in rows:
        if row.get("arm_id") != ARM_ID or row.get("candidate_id") != CANDIDATE_ID:
            raise D92CSOASHard10RunnerError("fit audit arm/candidate identity drift")
        if any(row.get(field) is not False for field in QUERY_ZERO_FIELDS + CSOAS_QUERY_ZERO_FIELDS):
            raise D92CSOASHard10RunnerError("fit audit query access is not zero")
        inventory = row.get("after_actual_component_inventory", {})
        actual = inventory.get("actual_component_fit_count", -1) if isinstance(inventory, Mapping) else -1
        total = row.get("after_total_component_fit_count", -1)
        mode = row.get("after_registered_d_mode_effective", "")
        if active:
            expected = (2, 1, "csoas_full")
            prefix = "d92_csoas_"
            if (int(total), int(actual), str(mode)) != expected:
                raise D92CSOASHard10RunnerError("fit audit CSOAS K>2 inventory drift")
            if not isinstance(inventory, Mapping) or int(inventory.get("full_component_fit_count", -1)) != 1 or int(inventory.get("block3_component_fit_count", -1)) != 0:
                raise D92CSOASHard10RunnerError("fit audit CSOAS FULL1/two-state inventory drift")
            if row.get(prefix + "active") is not True or row.get(prefix + "fallback_active") is not False or row.get(prefix + "fallback_reason") is not None:
                raise D92CSOASHard10RunnerError("fit audit CSOAS active/fallback drift")
            if int(row.get(prefix + "candidate_attempt_fit_count", -1)) != 1 or int(row.get(prefix + "fallback_reference_fit_count", -1)) != 0:
                raise D92CSOASHard10RunnerError("fit audit CSOAS fit receipt drift")
            if row.get(prefix + "candidate_statistic_receipt_available") is not True or row.get(prefix + "paired_e0_codec_state_equal") is not None:
                raise D92CSOASHard10RunnerError("fit audit CSOAS statistic/state receipt drift")
            if row.get("d92_e0d_csoas_g0_eligible") is not False or row.get("d92_e0d_csoas_g0_block_reason") != "PENDING_DEPLOYED_CODEC_PAIRED_E0":
                raise D92CSOASHard10RunnerError("fit audit CSOAS G0 block receipt drift")
            if int(row.get("d92_csoas_codec_retry_count", 0) or 0) != 0 or row.get("d92_csoas_codec_numeric_fallback") is True or int(row.get("d92_csoas_codec_fallback_component_execution_count", 0) or 0) != 0 or row.get("d92_csoas_codec_fallback_scope") not in (None, ""):
                raise D92CSOASHard10RunnerError("fit audit CSOAS codec retry/fallback is not formal")
            class_count = row.get("registered_class_count", row.get("class_count"))
            if isinstance(class_count, bool) or not isinstance(class_count, (int, float)) or int(class_count) <= 0 or float(int(class_count)) != float(class_count):
                raise D92CSOASHard10RunnerError("fit audit CSOAS registered-class receipt drift")
            if int(row.get("query_macs", -1)) != int(class_count) * 288:
                raise D92CSOASHard10RunnerError("fit audit CSOAS query MAC receipt drift")
            state_bytes = row.get("after_state_bytes")
            if isinstance(state_bytes, bool) or not isinstance(state_bytes, (int, float)) or int(state_bytes) <= 0 or float(int(state_bytes)) != float(state_bytes):
                raise D92CSOASHard10RunnerError("fit audit CSOAS state-byte receipt drift")
        else:
            expected = (3, 3, "d92_full_alias")
            if (int(total), int(actual), str(mode)) != expected:
                raise D92CSOASHard10RunnerError("fit audit K1 alias inventory drift")
            prefix = "d92_csoas_"
            if row.get(prefix + "active") is not False or row.get(prefix + "fallback_active") is not False or row.get(prefix + "fallback_reason") != "K1_K2_EXACT_D92_FULL_ALIAS":
                raise D92CSOASHard10RunnerError("fit audit K1 exact-alias receipt drift")
            if int(row.get(prefix + "candidate_attempt_fit_count", -1)) != 0 or int(row.get(prefix + "fallback_reference_fit_count", -1)) != 0:
                raise D92CSOASHard10RunnerError("fit audit K1 fit inventory drift")
            if row.get(prefix + "candidate_statistic_receipt_available") is not False or row.get("d92_e0d_csoas_g0_block_reason") != "K1_K2_EXACT_D92_FULL_ALIAS":
                raise D92CSOASHard10RunnerError("fit audit K1 alias block receipt drift")
            class_count = row.get("registered_class_count", row.get("class_count"))
            if isinstance(class_count, bool) or not isinstance(class_count, (int, float)) or int(class_count) <= 0 or float(int(class_count)) != float(class_count):
                raise D92CSOASHard10RunnerError("fit audit K1 registered-class receipt drift")
            if int(row.get("query_macs", -1)) != int(class_count) * 288:
                raise D92CSOASHard10RunnerError("fit audit K1 query MAC receipt drift")


def _verify_manifest_artifacts(manifest: Mapping[str, Any]) -> None:
    # Delegate immutable package/truth hash checks to the proven Hard11 helper.
    _base_runner._verify_manifest_artifacts(manifest)


def _read_json_object(path: str | Path) -> dict[str, Any]:
    return _base_runner._read_json_object(Path(path))


def _rewrite_schema(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict):
        return
    changed = False
    for key in ("schema", "status"):
        value = payload.get(key)
        if isinstance(value, str):
            new = value.replace("d92_pareto_distill_hard11", "d92_csoas_hard10").replace("PARETO_DISTILL", "CSOAS").replace("HARD11", "HARD10")
            if new != value:
                payload[key] = new
                changed = True
    if changed:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(path, stat.S_IREAD)


def _rewrite_output(root: Path) -> None:
    if root.is_dir():
        for path in root.rglob("*.json"):
            _rewrite_schema(path)


def _rewrite_shared_failure_evidence(root: str | Path) -> None:
    _rewrite_output(Path(root))


def _base_smoke_receipt_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload)


@contextmanager
def _runner_context() -> Iterator[None]:
    names = (
        "ARM_ID", "CANDIDATE_ID", "CANONICAL_SELECTION_SHA256", "LIVENESS_OUTER_KEY", "SCENES", "SHARD_COUNT", "SMOKE_OUTER_KEY",
        "build_hard11_manifest", "validate_hard11_manifest", "validate_method_lock", "_validate_fit_audit", "_validate_shared_smoke",
        "subprocess", "_verify_manifest_artifacts", "D92ParetoDistillHard11RunnerError", "D92ParetoDistillHard11Error", "D92ParetoDistillHard11ErrorAlias",
    )
    old = {name: getattr(_base_runner, name) for name in names}
    with _matrix._base_context():
        replacements = {
            "ARM_ID": ARM_ID, "CANDIDATE_ID": CANDIDATE_ID, "CANONICAL_SELECTION_SHA256": CANONICAL_SELECTION_SHA256,
            "LIVENESS_OUTER_KEY": LIVENESS_OUTER_KEY, "SCENES": SCENES, "SHARD_COUNT": SHARD_COUNT, "SMOKE_OUTER_KEY": SMOKE_OUTER_KEY,
            "build_hard11_manifest": build_hard10_manifest, "validate_hard11_manifest": validate_hard10_manifest,
            "validate_method_lock": validate_method_lock, "_validate_fit_audit": _validate_fit_audit,
            "_validate_shared_smoke": _validate_shared_smoke, "subprocess": subprocess,
            "_verify_manifest_artifacts": _verify_manifest_artifacts,
            "D92ParetoDistillHard11RunnerError": D92CSOASHard10RunnerError,
            "D92ParetoDistillHard11Error": D92CSOASHard10RunnerError,
            "D92ParetoDistillHard11ErrorAlias": D92CSOASHard10RunnerError,
        }
        for name, value in replacements.items():
            setattr(_base_runner, name, value)
        try:
            yield
        finally:
            for name, value in old.items():
                setattr(_base_runner, name, value)


def _smoke_job(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = [job for job in manifest.get("jobs", []) if job.get("outer_key") == manifest.get("smoke_outer_key") and job.get("outer_role") == "performance" and int(job.get("k_shot", -1)) > 2 and job.get("arm_id") == ARM_ID]
    if len(rows) != 1:
        raise D92CSOASHard10RunnerError("K>2 CSOAS smoke row identity drift")
    return rows[0]


def _validate_shared_smoke(manifest: Mapping[str, Any], *, manifest_sha256: str, device: str) -> None:
    if not _is_full_matrix(manifest):
        raise D92CSOASHard10RunnerError("CSOAS Hard9 matrix identity drift")
    _verify_manifest_artifacts(manifest)
    smoke_root = Path(str(manifest["output_root"])).resolve() / "smoke"
    receipt = _read_json_object(smoke_root / "smoke_receipt.json")
    job = _smoke_job(manifest)
    prediction_root = smoke_root / "diag"
    paths = _base_runner._prediction_closure_paths(prediction_root)
    identity = (
        receipt.get("schema") == "cvs.phase2.d92_csoas_hard10.smoke_receipt.v1"
        and receipt.get("status") == "D92_CSOAS_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
        and str(receipt.get("matrix_manifest_sha256", "")).lower() == str(manifest_sha256).lower()
        and receipt.get("selection_sha256") == CANONICAL_SELECTION_SHA256
        and receipt.get("smoke_outer_key") == SMOKE_OUTER_KEY
        and receipt.get("job_id") == job.get("job_id")
        and receipt.get("arm_id") == ARM_ID
        and receipt.get("candidate") == CANDIDATE_ID
        and int(receipt.get("k_shot", -1)) > 2
        and receipt.get("truth_open") is False
        and receipt.get("query_truth_joined_only_after_immutable_predictions") is True
        and receipt.get("prediction_and_scorer_processes_isolated") is True
        and all(receipt.get(field) is False for field in QUERY_ZERO_FIELDS)
    )
    if not identity:
        raise D92CSOASHard10RunnerError("CSOAS smoke receipt identity/protocol drift")
    _validate_fit_audit(paths["after_fit_audit"], k_shot=int(job["k_shot"]))
    if _base_runner._prediction_closure_status(prediction_root)[0] != "closed":
        raise D92CSOASHard10RunnerError("CSOAS smoke prediction closure drift")


def prepare(args: Any) -> dict[str, Any]:
    output = Path(args.output_root)
    if output.exists():
        raise D92CSOASHard10RunnerError("matrix output already exists")
    with _runner_context():
        result = _base_runner.prepare(args)
    _rewrite_output(output)
    result.update({"status": "CSOAS_HARD10_MATRIX_PREPARED", "schema": "cvs.phase2.d92_csoas_hard10.matrix.v1", "job_count": 10, "scene_arm_count": 30})
    return result


def truth_free_smoke(args: Any) -> dict[str, Any]:
    try:
        with _runner_context():
            result = _base_runner.truth_free_smoke(args)
    finally:
        _rewrite_output(Path(args.output_root))
        _rewrite_shared_failure_evidence(Path(args.output_root).parent)
    result.update({"status": "D92_CSOAS_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS", "schema": "cvs.phase2.d92_csoas_hard10.smoke_receipt.v1"})
    return result


def run_shard(args: Any) -> dict[str, Any]:
    manifest = None
    try:
        payload = json.loads(Path(args.matrix_manifest).read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            manifest = payload
    except (OSError, ValueError):
        pass
    try:
        with _runner_context():
            result = _base_runner.run_shard(args)
    finally:
        if manifest is not None:
            _rewrite_output(Path(str(manifest["output_root"])))
            _rewrite_shared_failure_evidence(manifest["output_root"])
    result["schema"] = "cvs.phase2.d92_csoas_hard10.shard_summary.v1"
    return result


def parser() -> Any:
    result = __import__("argparse").ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare"); p.add_argument("--context-manifest", required=True); p.add_argument("--method-lock", required=True); p.add_argument("--output-root", required=True)
    p = commands.add_parser("truth-free-smoke", aliases=["smoke"]); p.add_argument("--matrix-manifest", required=True); p.add_argument("--matrix-manifest-sha256", required=True); p.add_argument("--output-root", required=True); p.add_argument("--device", required=True); p.add_argument("--cpu-threads", type=int, default=2)
    p = commands.add_parser("run-shard"); p.add_argument("--matrix-manifest", required=True); p.add_argument("--matrix-manifest-sha256", required=True); p.add_argument("--shard-index", type=int, required=True); p.add_argument("--shard-count", type=int, choices=(SHARD_COUNT,), default=SHARD_COUNT); p.add_argument("--device", required=True); p.add_argument("--cpu-threads", type=int, default=2)
    return result


def main() -> int:
    args = parser().parse_args()
    value = prepare(args) if args.command == "prepare" else truth_free_smoke(args) if args.command in {"truth-free-smoke", "smoke"} else run_shard(args)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return 0 if value.get("status") in {"CSOAS_HARD10_MATRIX_PREPARED", "D92_CSOAS_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
