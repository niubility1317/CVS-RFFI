#!/usr/bin/env python3
"""Prepare, smoke-test and execute the frozen D92 TCRA Hard10 screen."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import sys
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d92_tcra_hard10 as _matrix
from cvsrffi.stage2_d92_tcra_hard10 import *  # noqa: F401,F403

try:
    from scripts import run_d92_pareto_distill_hard11 as _base_runner
except ImportError:  # pragma: no cover
    import run_d92_pareto_distill_hard11 as _base_runner  # type: ignore[no-redef]


CODE_ROOT = _base_runner.CODE_ROOT
PREDICTION_ENTRY = _base_runner.PREDICTION_ENTRY
SCORING_ENTRY = _base_runner.SCORING_ENTRY
QUERY_ZERO_FIELDS = tuple(_base_runner.QUERY_ZERO_FIELDS)
subprocess = _base_runner.subprocess
_BASE_VERIFY_MANIFEST_ARTIFACTS = _base_runner._verify_manifest_artifacts


class D92TCRAHard10RunnerError(RuntimeError):
    """Raised when TCRA Hard10 evidence or fit receipt would drift."""


D92TCRAHard10RunnerErrorAlias = D92TCRAHard10RunnerError


def _is_full_matrix(manifest: Mapping[str, Any]) -> bool:
    return int(manifest.get("job_count", -1)) == 10 and isinstance(manifest.get("jobs"), list) and len(manifest["jobs"]) == 10


def _finite(value: Any, label: str, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise D92TCRAHard10RunnerError(f"fit audit {label} is not finite")
    result = float(value)
    if lower is not None and result < lower:
        raise D92TCRAHard10RunnerError(f"fit audit {label} is below bound")
    return result


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise D92TCRAHard10RunnerError(f"fit audit {label} SHA drift")
    return value.lower()

_sha = _require_sha


def _validate_tcra_row(row: Mapping[str,Any], active: bool) -> None:
    p='d92_e0d_tcra_'
    for n in ('code1_byte_exact','scale1_byte_exact','scale2_byte_exact','intercept_byte_exact','log_diag_byte_exact'):
        if row.get(p+n) not in ({True} if active else {None,True}): raise D92TCRAHard10RunnerError(f"fit audit {n} state guard drift")
    if active:
        if row.get(p+'active') is not True:
            raise D92TCRAHard10RunnerError('fit audit TCRA candidate did not activate')
        if row.get(p+'active') is not True or row.get(p+'fallback_active') is not False or row.get(p+'fallback_reason') is not None: raise D92TCRAHard10RunnerError('fit audit TCRA active/fallback drift')
        if row.get(p+'final_gate_revision')!='safe_directional_v2' or row.get(p+'state_postprocess_mode')!='d42_tcra' or row.get(p+'direct_state_publish') is not True or row.get(p+'requantize_call_count')!=0: raise D92TCRAHard10RunnerError('fit audit TCRA revision/publish drift')
        if _sha(row.get(p+'e0_state_sha256'),'E0 state') == _sha(row.get(p+'final_state_sha256'),'final state'): raise D92TCRAHard10RunnerError('fit audit TCRA state did not change')
        counts=[int(_finite(row.get(p+n),n,0)) for n in ('requested_atomic_ascent_count','applied_atomic_ascent_count','generated_atomic_ascent_count','selected_atomic_ascent_count','rejected_atomic_ascent_count','prefix_guard_rejected_count','greedy_step_count')]
        requested,applied,generated,selected,rejected,prefix_rejected,steps=counts
        saturation=int(_finite(row.get(p+'aggregate_saturation_count'),'aggregate_saturation_count',0))
        if requested!=applied or requested!=selected or generated!=selected+rejected or prefix_rejected>rejected or steps!=selected+prefix_rejected or selected<1 or saturation<0 or saturation>rejected: raise D92TCRAHard10RunnerError('fit audit TCRA atomic receipt drift')
        if row.get(p+'modified_state_field_names')!=['coef2_qint8'] or row.get(p+'competitor_code_decrement_count')!=0: raise D92TCRAHard10RunnerError('fit audit TCRA state-field drift')
        for n in ('old_tail_count_by_class','old_tail_gain_by_class'):
            if not isinstance(row.get(p+n),list) or len(row[p+n])!=6: raise D92TCRAHard10RunnerError(f'fit audit {n} drift')
        tol=_finite(row.get(p+'guard_tolerance'),'guard_tolerance',0)
        if not all(_finite(v,'old tail gain')>=-tol for v in row[p+'old_tail_gain_by_class']) or _finite(row.get(p+'old_tail_gain_sum'),'old_tail_gain_sum')<=tol or _finite(row.get(p+'old_tail_min_gain'),'old_tail_min_gain') < -tol or _finite(row.get(p+'pooled_new_cross_tail_gain'),'pooled_new_cross_tail_gain') < -tol or _finite(row.get(p+'pooled_new_allclass_tail_gain'),'pooled_new_allclass_tail_gain') < -tol or _finite(row.get(p+'old_to_new_hinge_delta'),'old_to_new_hinge_delta') > tol or _finite(row.get(p+'new_to_old_hinge_delta'),'new_to_old_hinge_delta') > tol: raise D92TCRAHard10RunnerError('fit audit safe-directional guard drift')
        for n in ('support_guard_pass','safe_directional_pass','class_permutation_equivariant','row_permutation_invariant','true_class_row_only'):
            if row.get(p+n) is not True: raise D92TCRAHard10RunnerError(f'fit audit {n} drift')
        if row.get(p+'persistent_state_bytes_delta')!=0 or row.get(p+'component_fit_count')!=0: raise D92TCRAHard10RunnerError('fit audit resource receipt drift')
    else:
        if row.get(p+'active') is not False or row.get(p+'fallback_active') is not False or row.get(p+'fallback_reason')!='K1_K2_EXACT_D92_FULL_ALIAS': raise D92TCRAHard10RunnerError('fit audit alias drift')
        if _sha(row.get(p+'e0_state_sha256'),'E0 state') != _sha(row.get(p+'final_state_sha256'),'final state'): raise D92TCRAHard10RunnerError('fit audit alias state drift')

def _validate_fit_audit(path: str|Path, *, k_shot: int) -> None:
    try: rows=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except Exception as exc: raise D92TCRAHard10RunnerError('fit audit is invalid') from exc
    if not isinstance(rows,list) or len(rows)!=3 or {str(r.get('scenario')) for r in rows if isinstance(r,Mapping)} != set(SCENES): raise D92TCRAHard10RunnerError('fit audit scene closure drift')
    active=int(k_shot)>2; expected=(2,1,'full_only','d42_tcra') if active else (3,3,'d92_full_alias',None)
    for row in rows:
        if not isinstance(row,Mapping) or row.get('arm_id')!=ARM_ID or row.get('candidate_id')!=CANDIDATE_ID: raise D92TCRAHard10RunnerError('fit audit arm/candidate identity drift')
        if any(row.get(f) is not False for f in QUERY_ZERO_FIELDS): raise D92TCRAHard10RunnerError('fit audit query access is not zero')
        inv=row.get('after_actual_component_inventory',{}); actual=inv.get('actual_component_fit_count',-1) if isinstance(inv,Mapping) else -1; total=row.get('after_total_component_fit_count',-1); mode=row.get('after_registered_d_mode_effective',''); post=row.get('after_state_postprocess_mode')
        if (int(total),int(actual),str(mode),post)!=expected: raise D92TCRAHard10RunnerError('fit audit K/mode inventory drift')
        _validate_tcra_row(row,active)

def _verify_manifest_artifacts(manifest: Mapping[str, Any]) -> None:
    return _BASE_VERIFY_MANIFEST_ARTIFACTS(manifest)


def _rewrite_shared_failure_evidence(output_root: str | Path) -> None:
    """Translate only immutable shared failure receipts owned by this run."""
    root = Path(output_root)
    _rewrite_schema(root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json", to_tcra=True)
    ledger = root / "systemic_pre_prediction_failures"
    if ledger.is_dir():
        for path in ledger.rglob("*.json"):
            _rewrite_schema(path, to_tcra=True)


def _validate_shared_smoke(manifest: Mapping[str, Any], *, manifest_sha256: str, device: str) -> None:
    """Validate the immutable K>2 smoke closure for the 10-job matrix."""
    if int(manifest.get("job_count", -1)) != 10 or len(manifest.get("jobs", [])) != 10:
        raise D92TCRAHard10RunnerError("TCRA Hard10 matrix identity drift")
    _verify_manifest_artifacts(manifest)
    smoke_root = Path(str(manifest["output_root"])).resolve() / "smoke"
    receipt = _base_runner._read_json_object(smoke_root / "smoke_receipt.json")
    matches = [job for job in manifest["jobs"] if job.get("outer_key") == SMOKE_OUTER_KEY and job.get("outer_role") == "performance" and int(job.get("k_shot", -1)) > 2 and job.get("arm_id") == ARM_ID]
    if len(matches) != 1:
        raise D92TCRAHard10RunnerError("K>2 active-method smoke row identity drift")
    job = matches[0]
    prediction_root = smoke_root / "diag"
    closure_paths = _base_runner._prediction_closure_paths(prediction_root)
    identity = (
        receipt.get("schema") in {"cvs.phase2.d92_tcra_hard10.smoke_receipt.v1", "cvs.phase2.d92_pareto_distill_hard11.smoke_receipt.v1"}
        and receipt.get("status") in {"D92_TCRA_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS", "D92_PARETO_DISTILL_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"}
        and str(receipt.get("matrix_manifest_sha256", "")).lower() == str(manifest_sha256).lower()
        and receipt.get("selection_sha256") == CANONICAL_SELECTION_SHA256
        and receipt.get("smoke_outer_key") == SMOKE_OUTER_KEY
        and receipt.get("job_id") == job.get("job_id") and receipt.get("arm_id") == ARM_ID
        and receipt.get("candidate") == CANDIDATE_ID
        and receipt.get("truth_sidecar_sha256") == job.get("truth_sidecar_sha256")
        and int(receipt.get("k_shot", -1)) > 2
        and receipt.get("outer_role") == "performance" and receipt.get("truth_open") is False
        and receipt.get("query_truth_joined_only_after_immutable_predictions") is True
        and receipt.get("prediction_and_scorer_processes_isolated") is True
        and all(receipt.get(field) is False for field in QUERY_ZERO_FIELDS)
    )
    if not identity:
        raise D92TCRAHard10RunnerError("shared smoke receipt identity/protocol drift")
    expected_command = _base_runner._prediction_command(job, ground_component_dir=str(manifest["ground_component_dir"]), ground_manifest_sha256=str(manifest["ground_manifest_sha256"]), device=str(device), output_root=prediction_root)
    if receipt.get("command") != expected_command:
        raise D92TCRAHard10RunnerError("shared smoke command identity drift")
    hashes = {field: _base_runner._sha256_file(path) for field, path in {"before_prediction_sha256": closure_paths["before_prediction"], "after_prediction_sha256": closure_paths["after_prediction"], "before_commit_sha256": closure_paths["before_commit"], "after_commit_sha256": closure_paths["after_commit"], "before_fit_audit_sha256": closure_paths["before_fit_audit"], "after_fit_audit_sha256": closure_paths["after_fit_audit"], "fit_audit_sha256": closure_paths["after_fit_audit"]}.items()}
    _validate_fit_audit(closure_paths["after_fit_audit"], k_shot=int(job["k_shot"]))
    if any(receipt.get(field) != value for field, value in hashes.items()) or receipt.get("prediction_closure") != hashes or _base_runner._prediction_closure_status(prediction_root)[0] != "closed":
        raise D92TCRAHard10RunnerError("shared smoke prediction closure drift")


@contextmanager
def _runner_context() -> Iterator[None]:
    old = {name: getattr(_base_runner, name) for name in ("ARM_ID", "CANDIDATE_ID", "CANONICAL_SELECTION_SHA256", "LIVENESS_OUTER_KEY", "SCENES", "SHARD_COUNT", "SMOKE_OUTER_KEY", "build_hard11_manifest", "validate_hard11_manifest", "validate_method_lock", "_validate_fit_audit", "_validate_shared_smoke", "subprocess", "_verify_manifest_artifacts", "D92ParetoDistillHard11RunnerError", "D92ParetoDistillHard11Error", "D92ParetoDistillHard11ErrorAlias")}
    with _matrix._base_context(disable_validation=False):
        for name, value in {"ARM_ID": ARM_ID, "CANDIDATE_ID": CANDIDATE_ID, "CANONICAL_SELECTION_SHA256": CANONICAL_SELECTION_SHA256, "LIVENESS_OUTER_KEY": LIVENESS_OUTER_KEY, "SCENES": SCENES, "SHARD_COUNT": SHARD_COUNT, "SMOKE_OUTER_KEY": SMOKE_OUTER_KEY, "build_hard11_manifest": build_hard10_manifest, "validate_hard11_manifest": validate_hard10_manifest, "validate_method_lock": validate_method_lock, "_validate_fit_audit": _validate_fit_audit, "_validate_shared_smoke": _validate_shared_smoke, "subprocess": subprocess, "_verify_manifest_artifacts": _verify_manifest_artifacts, "D92ParetoDistillHard11RunnerError": D92TCRAHard10RunnerError, "D92ParetoDistillHard11Error": D92TCRAHard10RunnerError, "D92ParetoDistillHard11ErrorAlias": D92TCRAHard10RunnerError}.items():
            setattr(_base_runner, name, value)
        try:
            yield
        finally:
            for name, value in old.items():
                setattr(_base_runner, name, value)


def _rewrite_schema(path: Path, *, to_tcra: bool) -> None:
    if not path.is_file() or path.is_symlink():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict):
        return
    old, new = ("d92_pareto_distill_hard11", "d92_tcra_hard10") if to_tcra else ("d92_tcra_hard10", "d92_pareto_distill_hard11")
    changed = False
    for key in ("schema", "status"):
        value = payload.get(key)
        if isinstance(value, str) and (old in value or (to_tcra and "PARETO_DISTILL" in value) or (not to_tcra and "TCRA" in value)):
            payload[key] = value.replace(old, new)
            payload[key] = payload[key].replace("PARETO_DISTILL", "TCRA" if to_tcra else "PARETO_DISTILL")
            payload[key] = payload[key].replace("TCRA", "TCRA" if to_tcra else "PARETO_DISTILL")
            payload[key] = payload[key].replace("HARD11", "HARD10" if to_tcra else "HARD11")
            changed = True
    if not changed:
        return
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(path, stat.S_IREAD)
    except OSError as error:
        raise D92TCRAHard10RunnerError(f"cannot rewrite TCRA receipt: {path}") from error


def _rewrite_output(root: Path, *, to_tcra: bool) -> None:
    if root.is_dir():
        for path in root.rglob("*.json"):
            _rewrite_schema(path, to_tcra=to_tcra)


def _rewrite_shard_output(
    manifest: Mapping[str, Any], *, shard_index: int
) -> None:
    """Rewrite only evidence exclusively owned by one completed shard."""

    for job in manifest.get("jobs", []):
        if (
            isinstance(job, Mapping)
            and int(job.get("planned_shard_index", -1)) == int(shard_index)
        ):
            _rewrite_output(Path(str(job["output_root"])), to_tcra=True)
    _rewrite_schema(
        Path(str(manifest["output_root"]))
        / "summaries"
        / f"shard_{int(shard_index)}.json",
        to_tcra=True,
    )


def _base_smoke_receipt_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate TCRA smoke identity in memory for the proven base validator."""

    result = dict(payload)
    if result.get("schema") == "cvs.phase2.d92_tcra_hard10.smoke_receipt.v1":
        result["schema"] = (
            "cvs.phase2.d92_pareto_distill_hard11.smoke_receipt.v1"
        )
    if result.get("status") == "D92_TCRA_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS":
        result["status"] = (
            "D92_PARETO_DISTILL_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
        )
    return result


def build_hard10_manifest(**kwargs: Any) -> dict[str, Any]:
    return _matrix.build_hard10_manifest(**kwargs)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    with _runner_context():
        result = _base_runner.prepare(args)
    result["status"] = "TCRA_HARD10_MATRIX_PREPARED"
    result["job_count"] = 10
    result["scene_arm_count"] = 30
    _rewrite_output(Path(args.output_root), to_tcra=True)
    return result


def truth_free_smoke(args: argparse.Namespace) -> dict[str, Any]:
    try:
        with _runner_context():
            result = _base_runner.truth_free_smoke(args)
    finally:
        smoke_root = Path(args.output_root)
        _rewrite_output(smoke_root, to_tcra=True)
        _rewrite_shared_failure_evidence(smoke_root.parent)
    result["status"] = "D92_TCRA_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
    result["schema"] = "cvs.phase2.d92_tcra_hard10.smoke_receipt.v1"
    return result


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] | None = None
    try:
        loaded = json.loads(
            Path(args.matrix_manifest).read_text(encoding="utf-8-sig")
        )
        if isinstance(loaded, dict):
            manifest = loaded
    except (OSError, ValueError, KeyError, TypeError):
        pass
    original_read_json_object = _base_runner._read_json_object

    def read_json_object(path: str | Path) -> dict[str, Any]:
        return _base_smoke_receipt_view(original_read_json_object(path))

    try:
        with _runner_context():
            _base_runner._read_json_object = read_json_object
            try:
                result = _base_runner.run_shard(args)
            finally:
                _base_runner._read_json_object = original_read_json_object
    finally:
        if manifest is not None:
            _rewrite_shard_output(manifest, shard_index=int(args.shard_index))
            _rewrite_shared_failure_evidence(manifest["output_root"])
    result["schema"] = "cvs.phase2.d92_tcra_hard10.shard_summary.v1"
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
    return 0 if value["status"] in {"TCRA_HARD10_MATRIX_PREPARED", "D92_TCRA_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
