#!/usr/bin/env python3
"""静态规划并调度PairBiCAD-CV2 Phase1 source-only 24行矩阵。

候选、fold、seed、GPU分配和训练配置在模块加载时一次解析，运行中的worker
只消费已解析的row，不根据任何结果重写候选或配置。默认的真实调度会为每行
创建独立且不可覆盖的目录，并由dispatcher限制每张GPU最多两个直属worker。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvsrffi.phase1_bicad_xr.config import candidate_config, method_lock_payload
from cvsrffi.phase1_bicad_xr.metrics import (
    evaluate_final_checkpoint,
    validate_artifact_closure,
    validate_checkpoint_runtime,
)


RUN_ID_DEFAULT = "phase1_pairbicad_cv2_screen24_seed392002_20260831_r1"
CV2_CANDIDATE_IDS: tuple[str, ...] = (
    "CV2-B0",
    "CV2-B1",
    "CV2-B2",
    "CV2-B3",
    "CV2-D0",
    "CV2-D1",
    "CV2-D2",
    "CV2-D3",
    "CV2-T0",
    "CV2-T1",
    "CV2-T2",
    "CV2-T3",
)
CANDIDATE_IDS = CV2_CANDIDATE_IDS
FOLDS: tuple[int, int] = (1, 8)
FOLD_HELDOUT_RECEIVER: dict[int, int] = {1: 1, 8: 8}
SEED = 392002
GPU_IDS: tuple[int, ...] = tuple(range(8))
MAX_ACTIVE_PER_GPU = 2
SOURCE_RECEIVERS: tuple[int, ...] = (1, 3, 4, 6, 8)
TRAIN_DAYS: tuple[int, ...] = (1, 2, 3)
FORMAL_SCENARIOS: tuple[str, ...] = (
    "clean",
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
EXPECTED_ARTIFACTS: Mapping[str, str] = MappingProxyType(
    {
        "final_checkpoint": "bicad_xr_final.pth",
        "clean": "evaluations/clean.json",
        "leo_clear_weak": "evaluations/leo_clear_weak.json",
        "leo_low_elev_weak": "evaluations/leo_low_elev_weak.json",
        "leo_rain_weak": "evaluations/leo_rain_weak.json",
    }
)
EXECUTION_ACCOUNT = "ordinary_n607"
PLAN_SCHEMA = "pairbicad_cv2_screen24_plan_v1"
WORKER_STATUS_SCHEMA = "pairbicad_cv2_screen24_worker_status_v1"
ARTIFACTS_COMPLETE_STATUS = "ARTIFACTS_COMPLETE"
TECHNICAL_FAILURE_STATUS = "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
SOURCE_ONLY_ACCESS_FLAGS = (
    "target_access",
    "phase2_access",
    "support_access",
    "query_access",
    "truth_access",
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _static_candidate_data() -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Resolve every CV2 config and lock before any plan or worker is built."""

    resolved: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for candidate_id in CV2_CANDIDATE_IDS:
        config = candidate_config(candidate_id)
        configuration = _freeze(asdict(config))
        lock = dict(method_lock_payload(config))
        lock["configuration"] = configuration
        resolved[candidate_id] = (
            configuration,
            _freeze(lock),
        )
    return resolved


_STATIC_CANDIDATE_DATA = _static_candidate_data()


class PlanRow(NamedTuple):
    candidate_id: str
    fold: int
    seed: int
    optimizer_updates: int
    gpu_id: int
    source_receivers: tuple[int, ...]
    train_days: tuple[int, ...]
    configuration: Mapping[str, Any]
    method_lock: Mapping[str, Any]
    expected_artifacts: Mapping[str, str]
    source_only: bool = True
    target_access: bool = False
    phase2_access: bool = False
    support_access: bool = False
    query_access: bool = False
    truth_access: bool = False
    stage: str = "phase1_source_only"

    @property
    def row_id(self) -> str:
        return f"{self.candidate_id}-F{self.fold}-S{self.seed}"

    @property
    def gpu(self) -> int:
        return self.gpu_id

    @property
    def heldout_receiver(self) -> int:
        return FOLD_HELDOUT_RECEIVER[self.fold]


def _build_static_rows() -> tuple[PlanRow, ...]:
    rows: list[PlanRow] = []
    index = 0
    for candidate_id in CV2_CANDIDATE_IDS:
        configuration, method_lock = _STATIC_CANDIDATE_DATA[candidate_id]
        optimizer_updates = int(configuration["optimizer_updates"])
        for fold in FOLDS:
            heldout = FOLD_HELDOUT_RECEIVER[fold]
            source_receivers = tuple(
                receiver for receiver in SOURCE_RECEIVERS if receiver != heldout
            )
            rows.append(
                PlanRow(
                    candidate_id=candidate_id,
                    fold=fold,
                    seed=SEED,
                    optimizer_updates=optimizer_updates,
                    gpu_id=GPU_IDS[index % len(GPU_IDS)],
                    source_receivers=source_receivers,
                    train_days=TRAIN_DAYS,
                    configuration=configuration,
                    method_lock=method_lock,
                    expected_artifacts=EXPECTED_ARTIFACTS,
                )
            )
            index += 1
    return tuple(rows)


_STATIC_ROWS = _build_static_rows()
_STATIC_ROW_BY_ID = {row.row_id: row for row in _STATIC_ROWS}


def _validate_static_matrix(rows: Sequence[PlanRow]) -> None:
    if len(rows) != 24:
        raise RuntimeError("CV2 static matrix must contain exactly 24 rows")
    if {row.candidate_id for row in rows} != set(CV2_CANDIDATE_IDS):
        raise RuntimeError("CV2 static matrix candidate set changed")
    if {(row.fold, row.seed) for row in rows} != {(fold, SEED) for fold in FOLDS}:
        raise RuntimeError("CV2 static matrix fold/seed set changed")
    if len({row.row_id for row in rows}) != len(rows):
        raise RuntimeError("CV2 static matrix row IDs must be unique")
    if any(row.source_receivers not in {(3, 4, 6, 8), (1, 3, 4, 6)} for row in rows):
        raise RuntimeError("CV2 rows must use the declared source receiver folds")
    if any(row.train_days != TRAIN_DAYS for row in rows):
        raise RuntimeError("CV2 rows must use source days 1,2,3")
    if any(
        not row.source_only
        or row.target_access
        or row.phase2_access
        or row.support_access
        or row.query_access
        or row.truth_access
        for row in rows
    ):
        raise RuntimeError("CV2 rows must be source-only")
    if any(row.method_lock.get("dynamic_alias") is not False for row in rows):
        raise RuntimeError("CV2 rows must disable dynamic aliases")
    if any(row.method_lock.get("frozen") is not True for row in rows):
        raise RuntimeError("CV2 rows must carry frozen method locks")


_validate_static_matrix(_STATIC_ROWS)


def build_plan() -> list[PlanRow]:
    """Return a fresh view of the fixed 24-row matrix."""

    return list(_STATIC_ROWS)


def _row_payload(row: PlanRow) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "candidate_id": row.candidate_id,
        "fold": row.fold,
        "seed": row.seed,
        "optimizer_updates": row.optimizer_updates,
        "gpu_id": row.gpu_id,
        "source_receivers": list(row.source_receivers),
        "train_days": list(row.train_days),
        "source_only": row.source_only,
        "stage": row.stage,
        "method_lock": _thaw(row.method_lock),
        "configuration": _thaw(row.configuration),
        "expected_artifacts": _thaw(row.expected_artifacts),
    }


def build_plan_payload(
    rows: Sequence[PlanRow] | None = None,
    *,
    run_id: str = RUN_ID_DEFAULT,
    gpu_capacities: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    selected = list(build_plan() if rows is None else rows)
    _validate_static_matrix(selected) if len(selected) == 24 else None
    capacities = _normalise_gpu_capacities(gpu_capacities)
    return {
        "schema": PLAN_SCHEMA,
        "run_id": str(run_id),
        "execution_account": EXECUTION_ACCOUNT,
        "row_count": len(selected),
        "candidate_ids": list(CV2_CANDIDATE_IDS),
        "folds": list(FOLDS),
        "seed": SEED,
        "gpu_ids": list(GPU_IDS),
        "max_active_per_gpu": MAX_ACTIVE_PER_GPU,
        "gpu_capacities": {str(gpu): capacities[gpu] for gpu in GPU_IDS},
        "queued_rows": max(0, len(selected) - sum(capacities.values())),
        "source_receivers": list(SOURCE_RECEIVERS),
        "train_days": list(TRAIN_DAYS),
        "source_only": True,
        "static_configuration": True,
        "dynamic_alias": False,
        "expected_scenarios": list(FORMAL_SCENARIOS),
        "rows": [_row_payload(row) for row in selected],
    }


def _safe_component(value: str, *, name: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in "\\/\0"):
        raise ValueError(f"{name} must be one path component")
    return value


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    serialized = json.dumps(_thaw(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    return path


def _write_json_atomic_once(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish one JSON artifact atomically without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            _thaw(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return path


def reserve_run_layout(
    output_root: str | Path,
    run_id: str,
    rows: Sequence[PlanRow] | None = None,
) -> Path:
    """Create one fresh run root and one fresh row root per planned row."""

    selected = list(build_plan() if rows is None else rows)
    row_ids = [row.row_id for row in selected]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("rows must have unique row IDs")
    run_root = Path(output_root) / _safe_component(str(run_id), name="run_id")
    if run_root.exists():
        raise FileExistsError(f"run root already exists: {run_root}")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    for row_id in row_ids:
        (run_root / _safe_component(row_id, name="row_id")).mkdir(exist_ok=False)
    return run_root


def write_plan_json(
    run_root: str | Path,
    rows: Sequence[PlanRow],
    *,
    run_id: str,
    gpu_capacities: Mapping[int, int] | None = None,
) -> Path:
    return _write_json_once(
        Path(run_root) / "plan.json",
        build_plan_payload(rows, run_id=run_id, gpu_capacities=gpu_capacities),
    )


def _row_root(row: PlanRow, roots: "LauncherRoots") -> Path:
    return Path(roots.run_root) / row.row_id


def _require_static_row(row: PlanRow) -> None:
    expected = _STATIC_ROW_BY_ID.get(row.row_id)
    if expected is None or row != expected:
        raise ValueError(f"row is not one of the frozen CV2 matrix rows: {row.row_id}")


class LauncherRoots(NamedTuple):
    code_root: Path
    python: Path
    run_root: Path
    wisig_pkl: Path


def build_train_command(
    row: PlanRow,
    roots: LauncherRoots,
    *,
    run_id: str = RUN_ID_DEFAULT,
) -> list[str]:
    """Build the exact source-only training command for one frozen row."""

    _require_static_row(row)
    if not row.source_only or any(
        (
            row.target_access,
            row.phase2_access,
            row.support_access,
            row.query_access,
            row.truth_access,
        )
    ):
        raise ValueError("CV2 training rows must be source-only")
    script = Path(roots.code_root) / "code" / "SSDG" / "train_ssdg.py"
    command = [
        str(roots.python),
        "-u",
        str(script),
        "--phase1_method",
        "bicad_xr",
        "--wisig_pkl",
        str(roots.wisig_pkl),
        "--wisig_equalized",
        "1",
        "--sample_rate_hz",
        "25000000",
        "--wisig_train_days",
        ",".join(map(str, row.train_days)),
        "--wisig_test_days",
        "",
        "--wisig_train_rxs",
        ",".join(map(str, row.source_receivers)),
        "--wisig_test_rxs",
        "",
        "--wisig_allow_shared_days_if_receivers_disjoint",
        "false",
        "--phase1_source_only_eval",
        "true",
        "--split_mode",
        "tx_rx_day_1_7_2",
        "--labeled_ratio",
        "0.07",
        "--unlabeled_ratio",
        "0.63",
        "--source_val_ratio",
        "0.30",
        "--source_cal_ratio",
        "0.15",
        "--source_select_ratio",
        "0.15",
        "--phase1_source_role_protocol",
        "l_s_u_s_v_cal_v_select",
        "--sat_train_scenarios",
        "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "--batch_size",
        str(row.configuration["batch_size"]),
        "--use_tx_rx_balanced_sampler",
        "true",
        "--balanced_sampler_tx_per_batch",
        "6",
        "--balanced_sampler_domain_per_batch",
        "4",
        "--balanced_sampler_samples_per_cell",
        "4",
        "--balanced_sampler_replacement",
        "false",
        "--output_dir",
        str(_row_root(row, roots)),
        "--run_id",
        f"{run_id}-{row.row_id}",
        "--row_key",
        json.dumps(
            {
                "candidate_id": row.candidate_id,
                "fold": row.fold,
                "gpu_id": row.gpu_id,
                "optimizer_updates": row.optimizer_updates,
                "row_id": row.row_id,
                "seed": row.seed,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--candidate_id",
        row.candidate_id,
        "--bicad_optimizer_updates",
        str(row.optimizer_updates),
        "--bicad_loro_receiver",
        str(row.heldout_receiver),
        "--bicad_loro_eval_interval_updates",
        "500",
        "--epochs",
        "200",
        "--from_scratch",
        "true",
        "--checkpoint_selection",
        "final_only",
        "--device",
        "cuda:0",
        "--seed",
        str(row.seed),
    ]
    option_names = {token.lower() for token in command if token.startswith("--")}
    forbidden = {
        option
        for option in option_names
        if any(token in option for token in ("target", "phase2", "support", "query", "truth"))
    }
    if forbidden:
        raise ValueError("source-only command contains forbidden data-role options")
    return command


def build_worker_command(
    row: PlanRow,
    roots: LauncherRoots,
    *,
    run_id: str = RUN_ID_DEFAULT,
) -> list[str]:
    """Build a detached invocation of this launcher for one static row."""

    _require_static_row(row)
    return [
        str(roots.python),
        "-u",
        str(Path(__file__).resolve()),
        "--worker-row-id",
        row.row_id,
        "--run-id",
        str(run_id),
        "--run-root",
        str(Path(roots.run_root)),
        "--code-root",
        str(Path(roots.code_root)),
        "--python",
        str(Path(roots.python)),
        "--wisig-pkl",
        str(Path(roots.wisig_pkl)),
    ]


def _detached_popen_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if flags:
            kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def launch_detached_worker(
    row: PlanRow,
    roots: LauncherRoots,
    *,
    run_id: str,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> Any:
    """Start one row worker in a new process group and return its process handle."""

    _require_static_row(row)
    row_root = _row_root(row, roots)
    if not row_root.is_dir():
        raise FileNotFoundError(f"reserved row root is missing: {row_root}")
    log_path = row_root / "worker.log"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite worker log: {log_path}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(row.gpu_id)
    command = build_worker_command(row, roots, run_id=run_id)
    with log_path.open("x", encoding="utf-8", newline="\n") as handle:
        return popen_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            **_detached_popen_kwargs(),
        )


def dispatch_detached_workers(
    rows: Sequence[PlanRow],
    roots: LauncherRoots,
    *,
    run_id: str,
    worker_launcher: Callable[..., Any] = launch_detached_worker,
    poll_interval: float = 0.25,
    gpu_capacities: Mapping[int, int] | None = None,
) -> dict[str, str]:
    """Run up to two detached workers per GPU and queue the remaining rows."""

    selected = list(rows)
    if len({row.row_id for row in selected}) != len(selected):
        raise ValueError("dispatcher rows must have unique row IDs")
    if any(row.gpu_id not in GPU_IDS for row in selected):
        raise ValueError("dispatcher row GPU must be one of GPU0-7")
    if poll_interval < 0:
        raise ValueError("poll_interval must be non-negative")
    capacities = _normalise_gpu_capacities(gpu_capacities)

    pending = list(selected)
    active: dict[str, tuple[PlanRow, Any]] = {}
    statuses: dict[str, str] = {}
    while pending or active:
        launched = False
        for gpu_id in GPU_IDS:
            active_count = sum(row.gpu_id == gpu_id for row, _ in active.values())
            while active_count < capacities[gpu_id]:
                pending_index = next(
                    (index for index, row in enumerate(pending) if row.gpu_id == gpu_id),
                    None,
                )
                if pending_index is None:
                    break
                row = pending.pop(pending_index)
                process = worker_launcher(row, roots, run_id=run_id)
                active[row.row_id] = (row, process)
                active_count += 1
                launched = True

        completed: list[str] = []
        for row_id, (_row, process) in tuple(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            statuses[row_id] = (
                ARTIFACTS_COMPLETE_STATUS
                if int(returncode) == 0
                else TECHNICAL_FAILURE_STATUS
            )
            completed.append(row_id)
        for row_id in completed:
            del active[row_id]

        if active and not completed and not launched and poll_interval:
            time.sleep(poll_interval)
    return statuses


def write_dispatch_status(
    run_root: str | Path,
    rows: Sequence[PlanRow],
    statuses: Mapping[str, str],
) -> Path:
    expected_ids = {row.row_id for row in rows}
    if set(statuses) != expected_ids:
        raise ValueError("dispatcher status must contain every planned row exactly once")
    invalid = {
        row_id: status
        for row_id, status in statuses.items()
        if status not in {ARTIFACTS_COMPLETE_STATUS, TECHNICAL_FAILURE_STATUS}
    }
    if invalid:
        raise ValueError(f"dispatcher status contains non-terminal values: {invalid}")
    return _write_json_once(
        Path(run_root) / "dispatcher_status.json",
        {
            "schema": WORKER_STATUS_SCHEMA,
            "row_count": len(rows),
            "statuses": dict(statuses),
        },
    )


_FORMAL_EVALUATOR_MODULE: Any | None = None


def _load_formal_evaluator_module(code_root: Path) -> Any:
    """Load the repository's existing formal source-only evaluator context."""

    global _FORMAL_EVALUATOR_MODULE
    if _FORMAL_EVALUATOR_MODULE is not None:
        return _FORMAL_EVALUATOR_MODULE
    path = Path(code_root) / "code" / "scripts" / "launch_phase1_bicad_xr_matrix_20260830.py"
    if not path.is_file():
        raise FileNotFoundError(f"formal BiCAD-XR evaluator is missing: {path}")
    code_path = str(Path(code_root) / "code")
    ssdg_path = str(Path(code_root) / "code" / "SSDG")
    for value in (code_path, ssdg_path):
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("pairbicad_cv2_formal_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load formal BiCAD-XR evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _FORMAL_EVALUATOR_MODULE = module
    return module


def _build_final_evaluation_context(
    row: PlanRow,
    roots: LauncherRoots,
    command: Sequence[str],
) -> Any:
    formal_launcher = _load_formal_evaluator_module(Path(roots.code_root))
    context_type = getattr(formal_launcher, "_FormalEvaluationContext", None)
    if context_type is None:
        raise RuntimeError("formal BiCAD-XR evaluation context is unavailable")
    return context_type(row, roots, command)


def _row_runtime_expectation(
    row: PlanRow,
    *,
    optimizer_updates: int | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "fold": row.fold,
        "seed": row.seed,
        "optimizer_updates": (
            row.optimizer_updates if optimizer_updates is None else optimizer_updates
        ),
        "source_receivers": row.source_receivers,
        "train_days": row.train_days,
    }


def _cv2_runtime_expectation(row_root: str | Path, row: PlanRow) -> dict[str, Any]:
    """Validate dynamic CV2 convergence closure and bind its exact stop update."""

    if not bool(row.configuration.get("coverage_convergence", False)):
        return _row_runtime_expectation(row)

    root = Path(row_root)
    selection = _load_json_mapping(
        root / "source_loro_selection.json",
        label="source_loro_selection.json",
    )
    _validate_source_only_flags(selection, label="source-LORO selection")
    plan = selection.get("cv2_coverage_plan")
    terminal = selection.get("cv2_terminal")
    if not isinstance(plan, Mapping):
        raise ValueError("source-LORO selection has no CV2 coverage plan")
    if not isinstance(terminal, Mapping):
        raise ValueError("source-LORO selection has no terminal CV2 decision")

    required_plan = (
        "unlabeled_physical_count",
        "source_receiver_count",
        "unlabeled_per_four_updates",
        "u_cycle_updates",
        "eval_interval_updates",
        "min_activation_updates",
        "safety_updates",
    )
    if any(isinstance(plan.get(name), bool) or not isinstance(plan.get(name), int) for name in required_plan):
        raise ValueError("CV2 coverage plan fields must be integers")
    if any(int(plan[name]) <= 0 for name in required_plan):
        raise ValueError("CV2 coverage plan fields must be positive")
    if int(plan["source_receiver_count"]) != len(row.source_receivers):
        raise ValueError("CV2 coverage plan source receiver count does not match row")
    expected_four = 3 * 32 + (48 - 6 * len(row.source_receivers))
    if int(plan["unlabeled_per_four_updates"]) != expected_four:
        raise ValueError("CV2 coverage plan does not match structured batch cadence")

    def expected_updates(multiplier: float) -> int:
        target = int(math.ceil(int(plan["unlabeled_physical_count"]) * multiplier))
        full_blocks, remainder = divmod(target, expected_four)
        updates = full_blocks * 4
        if remainder == 0:
            return updates
        for count in (32, 32, 32, 48 - 6 * len(row.source_receivers)):
            updates += 1
            remainder -= count
            if remainder <= 0:
                return updates
        raise AssertionError("unreachable CV2 coverage remainder")

    expected_plan = {
        "u_cycle_updates": expected_updates(1.0),
        "eval_interval_updates": max(500, expected_updates(0.5)),
        "min_activation_updates": expected_updates(3.0),
        "safety_updates": expected_updates(12.0),
    }
    if any(int(plan[name]) != value for name, value in expected_plan.items()):
        raise ValueError("CV2 coverage plan update thresholds are inconsistent")

    stop_update = selection.get("stop_update")
    planned_updates = selection.get("planned_updates")
    interval = selection.get("interval")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (stop_update, planned_updates, interval)
    ):
        raise ValueError("CV2 selection update fields must be integers")
    if planned_updates != int(plan["safety_updates"]):
        raise ValueError("CV2 planned updates do not match the coverage safety budget")
    if interval != int(plan["eval_interval_updates"]):
        raise ValueError("CV2 evaluation interval does not match the coverage plan")
    if not (0 < stop_update <= planned_updates):
        raise ValueError("CV2 stop update is outside the planned budget")
    if stop_update != planned_updates and stop_update % interval != 0:
        raise ValueError("CV2 stop update is outside the source-LORO evaluation clock")
    if selection.get("stopped_early") is not True:
        raise ValueError("CV2 coverage candidate lacks a terminal stop decision")

    status = terminal.get("status")
    scientific = terminal.get("scientifically_converged")
    if status not in {"SCIENTIFICALLY_CONVERGED", "NOT_CONVERGED_SAFETY_STOP"}:
        raise ValueError("CV2 terminal status is invalid")
    if scientific is not (status == "SCIENTIFICALLY_CONVERGED"):
        raise ValueError("CV2 terminal status and scientific flag disagree")
    if terminal.get("artifacts_allowed") is not True:
        raise ValueError("CV2 terminal decision does not allow artifact closure")
    if scientific and stop_update < int(plan["min_activation_updates"]):
        raise ValueError("CV2 scientific stop precedes three U_s coverage cycles")

    curve_path = root / "source_loro_curve.jsonl"
    if not curve_path.is_file() or curve_path.stat().st_size <= 0:
        raise FileNotFoundError("source_loro_curve.jsonl is missing or empty")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        curve_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"source_loro_curve.jsonl line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, Mapping):
            raise ValueError(
                f"source_loro_curve.jsonl line {line_number} is not an object"
            )
        _validate_source_only_flags(record, label=f"source-LORO curve line {line_number}")
        update = record.get("update")
        if isinstance(update, bool) or not isinstance(update, int):
            raise ValueError("source-LORO curve update must be an integer")
        if records and update <= int(records[-1]["update"]):
            raise ValueError("source-LORO curve updates must increase strictly")
        records.append(record)
    if not records or records[-1].get("update") != stop_update:
        raise ValueError("source-LORO curve does not end at CV2 stop update")
    last_decision = records[-1].get("cv2_decision")
    if not isinstance(last_decision, Mapping) or last_decision.get("status") != status:
        raise ValueError("source-LORO curve terminal decision does not match selection")

    expectation = _row_runtime_expectation(row, optimizer_updates=stop_update)
    expectation["planned_optimizer_updates"] = planned_updates
    return expectation


def _assert_finite_json(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-finite number") from exc
        if not finite:
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, label=f"{label}[{index}]")
        return
    raise ValueError(f"{label} contains a non-JSON value: {type(value).__name__}")


def _validate_source_only_flags(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("source_only") is not True:
        raise ValueError(f"{label} must declare source_only=true")
    for name in SOURCE_ONLY_ACCESS_FLAGS:
        if payload.get(name) is not False:
            raise ValueError(f"{label} {name} must be false")


def _validate_evaluation_payload(
    payload: Mapping[str, Any],
    *,
    scenario: str,
    checkpoint_name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"evaluation payload for {scenario} must be a JSON object")
    if payload.get("scenario") != scenario:
        raise ValueError(f"evaluation scenario mismatch for {scenario}")
    if payload.get("checkpoint") != checkpoint_name:
        raise ValueError(f"evaluation checkpoint mismatch for {scenario}")
    if payload.get("checkpoint_load_strict") is not True:
        raise ValueError(f"evaluation checkpoint load is not strict for {scenario}")
    for name in ("missing_keys", "unexpected_keys", "shape_mismatches"):
        if payload.get(name) != []:
            raise ValueError(f"evaluation {name} is not empty for {scenario}")
    for name in ("accuracy", "floor_accuracy"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"evaluation {name} is not numeric for {scenario}")
        if not math.isfinite(float(value)):
            raise ValueError(f"evaluation {name} is not finite for {scenario}")
    per_class = payload.get("per_class_accuracy")
    if not isinstance(per_class, Mapping) or not per_class:
        raise ValueError(f"evaluation per_class_accuracy is empty for {scenario}")
    for class_id, value in per_class.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"evaluation per_class_accuracy[{class_id}] is not numeric for {scenario}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"evaluation per_class_accuracy[{class_id}] is not finite for {scenario}"
            )
    _validate_source_only_flags(payload, label=f"evaluation[{scenario}]")
    _assert_finite_json(payload, label=f"evaluation[{scenario}]")


def _build_atomic_source_only_evaluator(
    context: Any,
    *,
    row_root: Path,
    checkpoint_name: str,
) -> Callable[[Any, str], Mapping[str, Any]]:
    """Wrap the formal callback with fail-closed metadata and atomic JSON output."""

    def evaluate(model: Any, scenario: str) -> Mapping[str, Any]:
        if scenario not in FORMAL_SCENARIOS:
            raise ValueError(f"unsupported formal evaluation scenario: {scenario}")
        metrics = context.evaluate(model, scenario)
        if not isinstance(metrics, Mapping):
            raise TypeError("formal evaluator callback must return a mapping")
        callback_payload = dict(metrics)
        if "source_only" in callback_payload or any(
            name in callback_payload for name in SOURCE_ONLY_ACCESS_FLAGS
        ):
            _validate_source_only_flags(callback_payload, label=f"evaluator[{scenario}]")
        callback_payload["source_only"] = True
        callback_payload.update({name: False for name in SOURCE_ONLY_ACCESS_FLAGS})
        result_payload = dict(callback_payload)
        result_payload.pop("log", None)
        result_payload.update(
            {
                "scenario": scenario,
                "checkpoint": checkpoint_name,
                "checkpoint_load_strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            }
        )
        _validate_evaluation_payload(
            result_payload,
            scenario=scenario,
            checkpoint_name=checkpoint_name,
        )
        _write_json_atomic_once(
            row_root / "evaluations" / f"{scenario}.json",
            result_payload,
        )
        return callback_payload

    return evaluate


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _validate_worker_artifacts(
    row: PlanRow,
    row_root: Path,
    checkpoint: Path,
    evaluation: Mapping[str, Any],
    runtime_expectation: Mapping[str, Any],
) -> dict[str, Any]:
    if evaluation.get("complete") is not True:
        raise ValueError("formal final evaluation did not report complete")
    runtime_artifact = _load_json_mapping(
        row_root / "checkpoint_runtime.json",
        label="checkpoint_runtime.json",
    )
    if runtime_artifact.get("checkpoint_path") != checkpoint.name:
        raise ValueError("checkpoint_runtime.json checkpoint identity mismatch")
    runtime = runtime_artifact.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("checkpoint_runtime.json runtime must be a JSON object")
    _validate_source_only_flags(runtime, label="checkpoint runtime")
    runtime_check = validate_checkpoint_runtime(
        runtime,
        runtime_expectation,
    )
    if runtime_check.get("valid") is not True:
        raise ValueError(f"checkpoint runtime mismatch: {runtime_check}")
    if runtime_artifact.get("strict_reconstruction") is not True:
        raise ValueError("checkpoint reconstruction was not strict")
    if runtime_artifact.get("trainer_runtime_strict") is not True:
        raise ValueError("trainer runtime restoration was not strict")
    for name in ("missing_keys", "unexpected_keys", "shape_mismatches"):
        if runtime_artifact.get(name) != []:
            raise ValueError(f"checkpoint runtime {name} is not empty")

    for scenario in FORMAL_SCENARIOS:
        path = row_root / "evaluations" / f"{scenario}.json"
        payload = _load_json_mapping(path, label=f"evaluations/{scenario}.json")
        _validate_evaluation_payload(
            payload,
            scenario=scenario,
            checkpoint_name=checkpoint.name,
        )
    closure = validate_artifact_closure(row_root)
    if closure.get("complete") is not True:
        raise ValueError(f"formal artifact closure is incomplete: {closure}")
    if set(closure.get("evaluations", {})) != set(FORMAL_SCENARIOS):
        raise ValueError("formal artifact closure is missing one or more scenarios")
    return dict(closure)


def _record_technical_failure(
    row_root: Path,
    *,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    path = row_root / "TECHNICAL_FAILURE.json"
    if path.exists():
        return
    _write_json_once(
        path,
        {
            "status": TECHNICAL_FAILURE_STATUS,
            "reason": str(reason),
            "details": dict(details or {}),
        },
    )


def _write_worker_status(
    row: PlanRow,
    row_root: Path,
    *,
    status: str,
    returncode: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": WORKER_STATUS_SCHEMA,
        "row_id": row.row_id,
        "status": status,
        "expected_artifacts": dict(EXPECTED_ARTIFACTS),
    }
    if returncode is not None:
        payload["returncode"] = int(returncode)
    if details:
        payload["details"] = dict(details)
    _write_json_once(row_root / "worker_status.json", payload)


def _stop_worker(
    row: PlanRow,
    row_root: Path,
    *,
    reason: str,
    details: Mapping[str, Any] | None = None,
    returncode: int | None = None,
) -> str:
    _record_technical_failure(row_root, reason=reason, details=details)
    _write_worker_status(
        row,
        row_root,
        status=TECHNICAL_FAILURE_STATUS,
        returncode=returncode,
        details=details,
    )
    return TECHNICAL_FAILURE_STATUS


def _locate_final_checkpoint(row_root: Path) -> Path:
    for name in ("bicad_xr_final.pth", "final_checkpoint.pt", "final_bicad_xr.pt"):
        checkpoint = row_root / name
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return checkpoint
    raise FileNotFoundError("row final checkpoint is missing or empty")


def run_training_worker(row: PlanRow, roots: LauncherRoots, *, run_id: str) -> str:
    """Train one row, then close its strict source-only four-scenario artifacts."""

    _require_static_row(row)
    row_root = _row_root(row, roots)
    if not row_root.is_dir():
        raise FileNotFoundError(f"reserved row root is missing: {row_root}")
    log_path = row_root / "train.log"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite training log: {log_path}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(row.gpu_id)
    try:
        command = build_train_command(row, roots, run_id=run_id)
        with log_path.open("x", encoding="utf-8", newline="\n") as handle:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
    except Exception as exc:
        return _stop_worker(
            row,
            row_root,
            reason="TRAINING_SUBPROCESS_EXCEPTION",
            details={"exception_type": type(exc).__name__, "message": str(exc)},
        )
    if int(completed.returncode) != 0:
        return _stop_worker(
            row,
            row_root,
            reason="TRAINING_SUBPROCESS_FAILED",
            returncode=int(completed.returncode),
        )

    try:
        checkpoint = _locate_final_checkpoint(row_root)
        runtime_expectation = _cv2_runtime_expectation(row_root, row)
        context = _build_final_evaluation_context(row, roots, command)
        evaluator = _build_atomic_source_only_evaluator(
            context,
            row_root=row_root,
            checkpoint_name=checkpoint.name,
        )
        evaluation = evaluate_final_checkpoint(
            checkpoint,
            expected_runtime=runtime_expectation,
            output_dir=row_root,
            model_builder=context.build_model,
            trainer_runtime_restorer=context.restore_trainer_runtime,
            evaluator=evaluator,
        )
        closure = _validate_worker_artifacts(
            row,
            row_root,
            checkpoint,
            evaluation,
            runtime_expectation,
        )
    except Exception as exc:
        return _stop_worker(
            row,
            row_root,
            reason="FINAL_ARTIFACT_EVALUATION_FAILED",
            details={"exception_type": type(exc).__name__, "message": str(exc)},
            returncode=int(completed.returncode),
        )

    _write_json_atomic_once(row_root / "ARTIFACTS_COMPLETE.json", closure)
    _write_worker_status(
        row,
        row_root,
        status=ARTIFACTS_COMPLETE_STATUS,
        returncode=int(completed.returncode),
    )
    return ARTIFACTS_COMPLETE_STATUS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish and dispatch the fixed PairBiCAD-CV2 Phase1 source-only screen24 matrix."
    )
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--code-root", default="")
    parser.add_argument("--python", dest="python_path", default=sys.executable)
    parser.add_argument("--wisig-pkl", dest="wisig_pkl", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpu-capacities",
        default="",
        help="Preflight-frozen per-GPU worker slots, for example 0:1,1:2,...,7:2.",
    )
    parser.add_argument("--worker-row-id", dest="worker_row_id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--run-root", dest="run_root", default="", help=argparse.SUPPRESS)
    return parser


def _parse_gpu_capacities(raw: str) -> dict[int, int]:
    if not str(raw).strip():
        return _normalise_gpu_capacities(None)
    parsed: dict[int, int] = {}
    for item in str(raw).split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError("gpu-capacities must use gpu:slots comma-separated syntax")
        gpu_id, slots = (int(value) for value in fields)
        if gpu_id in parsed:
            raise ValueError("gpu-capacities contains a duplicate GPU")
        parsed[gpu_id] = slots
    return _normalise_gpu_capacities(parsed)


def _normalise_gpu_capacities(
    capacities: Mapping[int, int] | None,
) -> dict[int, int]:
    resolved = (
        {gpu_id: MAX_ACTIVE_PER_GPU for gpu_id in GPU_IDS}
        if capacities is None
        else dict(capacities)
    )
    if set(resolved) != set(GPU_IDS):
        raise ValueError("gpu capacities must declare GPU0-7 exactly once")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_ACTIVE_PER_GPU
        for value in resolved.values()
    ):
        raise ValueError("each GPU capacity must be an integer in [0,2]")
    if sum(resolved.values()) <= 0:
        raise ValueError("at least one GPU worker slot is required")
    return {gpu_id: int(resolved[gpu_id]) for gpu_id in GPU_IDS}


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    code_root = Path(args.code_root).resolve() if args.code_root else Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root) if args.output_root else code_root / "runs"
    wisig_pkl = (
        Path(args.wisig_pkl)
        if args.wisig_pkl
        else code_root / "Dataset_WigSig" / "ManySig.pkl"
    )
    return code_root, output_root, Path(args.python_path), wisig_pkl


def _worker_main(args: argparse.Namespace) -> int:
    if not args.worker_row_id or not args.run_root:
        raise ValueError("worker mode requires worker-row-id and run-root")
    row = _STATIC_ROW_BY_ID.get(args.worker_row_id)
    if row is None:
        raise ValueError(f"unknown static CV2 row: {args.worker_row_id}")
    code_root, _output_root, python_path, wisig_pkl = _resolve_paths(args)
    roots = LauncherRoots(code_root, python_path, Path(args.run_root), wisig_pkl)
    status = run_training_worker(row, roots, run_id=args.run_id)
    return 0 if status == ARTIFACTS_COMPLETE_STATUS else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_row_id:
        return _worker_main(args)

    rows = build_plan()
    gpu_capacities = _parse_gpu_capacities(args.gpu_capacities)
    if args.dry_run:
        print(
            json.dumps(
                build_plan_payload(
                    rows,
                    run_id=args.run_id,
                    gpu_capacities=gpu_capacities,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if not args.run_id:
        raise ValueError("run-id is required for a real launch")

    code_root, output_root, python_path, wisig_pkl = _resolve_paths(args)
    run_root = reserve_run_layout(output_root, args.run_id, rows)
    write_plan_json(
        run_root,
        rows,
        run_id=args.run_id,
        gpu_capacities=gpu_capacities,
    )
    roots = LauncherRoots(code_root, python_path, run_root, wisig_pkl)
    statuses = dispatch_detached_workers(
        rows,
        roots,
        run_id=args.run_id,
        gpu_capacities=gpu_capacities,
    )
    write_dispatch_status(run_root, rows, statuses)
    return 0 if all(status == ARTIFACTS_COMPLETE_STATUS for status in statuses.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
