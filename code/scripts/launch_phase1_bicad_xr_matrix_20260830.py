#!/usr/bin/env python3
"""Plan and launch the source-only BiCAD-XR Phase1 matrix.

The planning and dry-run paths are side-effect free.  A real launch creates a
new run root, reserves one directory per row, and only passes source-domain
inputs to the BiCAD-XR training entry point.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvsrffi.phase1_bicad_xr.config import candidate_config
from cvsrffi.phase1_bicad_xr.metrics import (
    evaluate_final_checkpoint,
    validate_artifact_closure,
)


RUN_ID_DEFAULT = "phase1_bicad_xr_matrix_20260830_r1"
PAIRBICAD_RUN_ID = "phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1"
FORMAL_SEEDS: tuple[int, int, int] = (392001, 392002, 392003)
QUICK_FOLDS: tuple[int, int] = (1, 8)
CONFIRM_FOLDS: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)
PAIRBICAD_FOLDS: tuple[int, int] = (1, 8)
PAIRBICAD_CONVERGENCE_FOLDS: tuple[int, int] = (1, 8)
PAIRBICAD_FINAL_FOLDS: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)
SOURCE_RECEIVERS: tuple[int, int, int, int, int] = (1, 3, 4, 6, 8)
TRAIN_DAYS: tuple[int, int, int] = (1, 2, 3)
OPTIMIZER_UPDATES = 5000
PAIRBICAD_OPTIMIZER_UPDATES = 4000
PAIRBICAD_CONVERGENCE_OPTIMIZER_UPDATES = 9000
PAIRBICAD_FINAL_MIN_UPDATES = 4000
PAIRBICAD_FINAL_MAX_UPDATES = 9000
PAIRBICAD_UPDATE_INTERVAL = 500
MAX_JOBS_PER_GPU = 3
PAIRBICAD_MAX_JOBS_PER_GPU = 2
QUICK_CANDIDATES: tuple[str, str, str, str] = (
    "D0",
    "D5",
    "E1",
    "ADV3B02-BiCAD-XDC-V1",
)
CONFIRM_CANDIDATES: tuple[str, ...] = tuple(
    [f"D{i}" for i in range(7)]
    + [f"E{i}" for i in range(5)]
    + [f"F{i}" for i in range(4)]
)
PAIRBICAD_CANDIDATES: tuple[str, str, str, str, str] = (
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
)
LORO_HELDOUT_RECEIVER: dict[int, int] = {
    1: 1,
    2: 3,
    3: 4,
    4: 6,
    5: 8,
    8: 8,
}
PAIRBICAD_CONVERGENCE_STAGES = frozenset(
    {"pairbicad_convergence", "pairbicad_final"}
)
FORMAL_STATES = {
    "ARTIFACTS_COMPLETE",
    "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
}

_FORBIDDEN_OPTION_TOKENS = ("target", "phase2", "support", "query", "truth")


class PlanRow(NamedTuple):
    candidate_id: str
    fold: int
    seed: int
    optimizer_updates: int
    gpu_id: int
    source_receivers: tuple[int, ...]
    train_days: tuple[int, ...]
    source_only: bool = True
    target_access: bool = False
    phase2_access: bool = False
    support_access: bool = False
    query_access: bool = False
    truth_access: bool = False
    stage: str = "quick"

    @property
    def row_id(self) -> str:
        return f"{self.candidate_id}-F{self.fold}-S{self.seed}"

    @property
    def gpu(self) -> int:
        """Compatibility alias for launchers that call the assignment ``gpu``."""

        return self.gpu_id


class LauncherRoots(NamedTuple):
    code_root: Path
    python: Path
    run_root: Path
    wisig_pkl: Path


def _unique_ints(
    values: Sequence[int],
    *,
    name: str,
    allowed: Sequence[int] | None = None,
    minimum: int = 1,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    try:
        resolved = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError(f"{name} must be a non-empty unique sequence")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in resolved):
        raise ValueError(f"{name} must contain integers")
    if any(value < minimum for value in resolved):
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{name} must contain {qualifier} integers")
    if allowed is not None and any(value not in allowed for value in resolved):
        raise ValueError(f"{name} must be selected from {tuple(allowed)}")
    return resolved


def _validated_seeds(seeds: Sequence[int] | None) -> tuple[int, ...]:
    selected = FORMAL_SEEDS if seeds is None else _unique_ints(seeds, name="seeds")
    if any(seed not in FORMAL_SEEDS for seed in selected):
        raise ValueError(f"seeds must be selected from {FORMAL_SEEDS}")
    return tuple(selected)


def _validated_gpus(gpus: Sequence[int] | None) -> tuple[int, ...]:
    selected = (
        tuple(range(8))
        if gpus is None
        else _unique_ints(gpus, name="gpu_ids", minimum=0)
    )
    return tuple(selected)


def _canonical_candidate(candidate_id: str) -> str:
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string")
    try:
        return str(candidate_config(candidate_id).candidate_id)
    except ValueError as exc:
        raise ValueError(f"unknown candidate: {candidate_id}") from exc


def _validated_candidates(candidates: Sequence[str] | None, *, stage: str) -> tuple[str, ...]:
    if candidates is None and stage == "quick":
        selected = QUICK_CANDIDATES
    elif candidates is None and stage == "confirm":
        selected = CONFIRM_CANDIDATES
    elif candidates is None and stage == "pairbicad":
        selected = PAIRBICAD_CANDIDATES
    elif candidates is None:
        raise ValueError(f"{stage} requires explicit PairBiCAD candidates")
    else:
        selected = tuple(candidates)
    if isinstance(selected, (str, bytes)) or not selected:
        raise ValueError("candidates must be a non-empty sequence")
    canonical = tuple(_canonical_candidate(value) for value in selected)
    if len(canonical) != len(set(canonical)):
        raise ValueError("candidates must be unique")
    if stage in PAIRBICAD_CONVERGENCE_STAGES:
        if any(value not in PAIRBICAD_CANDIDATES for value in canonical):
            raise ValueError("pairbicad stages require PairBiCAD candidates")
        expected_count = 2 if stage == "pairbicad_convergence" else 1
        if len(canonical) != expected_count:
            raise ValueError(
                f"{stage} requires exactly {expected_count} distinct PairBiCAD candidates"
            )
    return canonical


def build_plan(
    *,
    stage: str = "quick",
    candidates: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    gpu_ids: Sequence[int] | None = None,
    gpus: Sequence[int] | None = None,
    optimizer_updates: int | None = None,
) -> list[PlanRow]:
    """Build a deterministic source-LORO plan without touching the filesystem."""

    key = str(stage).strip().lower()
    if key == "full":
        key = "confirm"
    if key not in {
        "quick",
        "confirm",
        "pairbicad",
        "pairbicad_convergence",
        "pairbicad_final",
    }:
        raise ValueError(f"unknown stage: {stage}")
    if gpu_ids is not None and gpus is not None:
        raise ValueError("pass only one of gpu_ids or gpus")
    selected_gpus = _validated_gpus(gpu_ids if gpu_ids is not None else gpus)
    selected_folds = _unique_ints(
        QUICK_FOLDS
        if folds is None and key == "quick"
        else PAIRBICAD_FOLDS
        if folds is None and key in {"pairbicad", "pairbicad_convergence"}
        else PAIRBICAD_FINAL_FOLDS
        if folds is None and key == "pairbicad_final"
        else CONFIRM_FOLDS
        if folds is None
        else folds,
        name="folds",
        allowed=(
            QUICK_FOLDS
            if key == "quick"
            else PAIRBICAD_FOLDS
            if key in {"pairbicad", "pairbicad_convergence"}
            else PAIRBICAD_FINAL_FOLDS
            if key == "pairbicad_final"
            else CONFIRM_FOLDS
        ),
    )
    selected_seeds = _validated_seeds(seeds)
    if key in PAIRBICAD_CONVERGENCE_STAGES and selected_seeds != FORMAL_SEEDS:
        raise ValueError(
            f"{key} requires seeds exactly {FORMAL_SEEDS}"
        )
    required_folds = (
        PAIRBICAD_CONVERGENCE_FOLDS
        if key == "pairbicad_convergence"
        else PAIRBICAD_FINAL_FOLDS
        if key == "pairbicad_final"
        else None
    )
    if required_folds is not None and selected_folds != required_folds:
        raise ValueError(f"{key} requires folds exactly {required_folds}")
    selected_updates = (
        PAIRBICAD_OPTIMIZER_UPDATES
        if optimizer_updates is None and key == "pairbicad"
        else PAIRBICAD_CONVERGENCE_OPTIMIZER_UPDATES
        if optimizer_updates is None and key == "pairbicad_convergence"
        else None
        if optimizer_updates is None and key == "pairbicad_final"
        else OPTIMIZER_UPDATES
        if optimizer_updates is None
        else optimizer_updates
    )
    if selected_updates is None:
        raise ValueError("pairbicad_final requires an explicit optimizer_updates budget")
    if isinstance(selected_updates, bool) or not isinstance(selected_updates, int):
        raise ValueError("optimizer_updates must be an integer")
    if selected_updates <= 0:
        raise ValueError("optimizer_updates must be positive")
    if key == "pairbicad" and selected_updates != PAIRBICAD_OPTIMIZER_UPDATES:
        raise ValueError(
            f"pairbicad stage requires optimizer_updates={PAIRBICAD_OPTIMIZER_UPDATES}"
        )
    if key == "pairbicad_convergence" and selected_updates != PAIRBICAD_CONVERGENCE_OPTIMIZER_UPDATES:
        raise ValueError(
            "pairbicad_convergence stage requires "
            f"optimizer_updates={PAIRBICAD_CONVERGENCE_OPTIMIZER_UPDATES}"
        )
    if key == "pairbicad_final" and not (
        PAIRBICAD_FINAL_MIN_UPDATES <= selected_updates <= PAIRBICAD_FINAL_MAX_UPDATES
        and selected_updates % PAIRBICAD_UPDATE_INTERVAL == 0
    ):
        raise ValueError(
            "pairbicad_final optimizer_updates must be a 500-multiple in [4000,9000]"
        )
    selected_candidates = _validated_candidates(candidates, stage=key)

    rows: list[PlanRow] = []
    index = 0
    for candidate_id in selected_candidates:
        for fold in selected_folds:
            heldout_receiver = LORO_HELDOUT_RECEIVER[fold]
            source_receivers = tuple(
                receiver for receiver in SOURCE_RECEIVERS if receiver != heldout_receiver
            )
            if heldout_receiver not in SOURCE_RECEIVERS:
                raise ValueError("heldout receiver must belong to the source receiver universe")
            if heldout_receiver in source_receivers:
                raise ValueError("heldout receiver must not overlap source receivers")
            for seed in selected_seeds:
                rows.append(
                    PlanRow(
                        candidate_id=candidate_id,
                        fold=fold,
                        seed=seed,
                        optimizer_updates=selected_updates,
                        gpu_id=selected_gpus[index % len(selected_gpus)],
                        source_receivers=source_receivers,
                        train_days=TRAIN_DAYS,
                        stage=key,
                    )
                )
                index += 1
    if not rows:
        raise ValueError("plan contains no rows")
    return rows


def _validated_max_jobs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_JOBS_PER_GPU:
        raise ValueError(f"max_jobs_per_gpu must be between 1 and {MAX_JOBS_PER_GPU}")
    return value


def pack_rows(
    rows: Sequence[PlanRow],
    *,
    gpu_ids: Sequence[int] = tuple(range(8)),
    max_jobs_per_gpu: int = MAX_JOBS_PER_GPU,
) -> list[PlanRow]:
    """Assign rows round-robin while enforcing the hard per-GPU upper bound."""

    limit = _validated_max_jobs(max_jobs_per_gpu)
    selected_gpus = _validated_gpus(gpu_ids)
    if not rows:
        return []
    if len(rows) > len(selected_gpus) * limit:
        raise ValueError(
            f"{len(rows)} rows exceed safe capacity {len(selected_gpus) * limit}"
        )
    packed: list[PlanRow] = []
    counts = {gpu: 0 for gpu in selected_gpus}
    for index, row in enumerate(rows):
        gpu = selected_gpus[index % len(selected_gpus)]
        counts[gpu] += 1
        if counts[gpu] > limit:
            raise ValueError(f"GPU {gpu} exceeds max_jobs_per_gpu={limit}")
        packed.append(row._replace(gpu_id=gpu))
    return packed


def queue_rows(
    rows: Sequence[PlanRow],
    *,
    gpu_ids: Sequence[int] = tuple(range(8)),
) -> list[PlanRow]:
    """Assign queued rows round-robin without changing legacy pack semantics."""

    selected_gpus = _validated_gpus(gpu_ids)
    if not rows:
        return []
    return [
        row._replace(gpu_id=selected_gpus[index % len(selected_gpus)])
        for index, row in enumerate(rows)
    ]


def safe_gpu_slots(
    inventory: Sequence[Mapping[str, Any]],
    *,
    max_jobs_per_gpu: int = MAX_JOBS_PER_GPU,
    estimated_row_memory_mb: int = 8000,
    reserve_memory_mb: int = 2000,
) -> dict[int, int]:
    """Compute safe slots from a read-only inventory; never mutates processes or input."""

    limit = _validated_max_jobs(max_jobs_per_gpu)
    if isinstance(estimated_row_memory_mb, bool) or estimated_row_memory_mb <= 0:
        raise ValueError("estimated_row_memory_mb must be positive")
    if isinstance(reserve_memory_mb, bool) or reserve_memory_mb < 0:
        raise ValueError("reserve_memory_mb must be non-negative")
    slots: dict[int, int] = {}
    for item in inventory:
        if not isinstance(item, Mapping):
            raise ValueError("GPU inventory entries must be mappings")
        gpu_id = item.get("gpu_id")
        free_memory = item.get("free_memory_mb")
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError("GPU inventory gpu_id must be a non-negative integer")
        if isinstance(free_memory, bool) or not isinstance(free_memory, (int, float)):
            slots[gpu_id] = 0
            continue
        usable = max(0.0, float(free_memory) - float(reserve_memory_mb))
        slots[gpu_id] = min(limit, int(usable // float(estimated_row_memory_mb)))
    return slots


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in "\\/\0"):
        raise ValueError("run_id must be one path component")
    return value


def reserve_run_layout(
    output_root: str | Path,
    run_id: str,
    rows: Sequence[PlanRow],
) -> Path:
    """Create a new run root and one empty, non-overwriting directory per row."""

    run_root = Path(output_root) / _safe_component(str(run_id))
    if run_root.exists():
        raise FileExistsError(f"run root already exists: {run_root}")
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("rows must have unique row_id values")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    for row_id in row_ids:
        (run_root / _safe_component(row_id)).mkdir(exist_ok=False)
    return run_root


def _row_root(row: PlanRow, roots: LauncherRoots) -> Path:
    return Path(roots.run_root) / row.row_id


def _row_expectation(
    row: PlanRow,
    *,
    optimizer_updates: int | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "fold": row.fold,
        "seed": row.seed,
        "optimizer_updates": (
            row.optimizer_updates
            if optimizer_updates is None
            else int(optimizer_updates)
        ),
        "source_receivers": row.source_receivers,
        "train_days": row.train_days,
    }


def build_train_command(row: PlanRow, roots: LauncherRoots, *, run_id: str = RUN_ID_DEFAULT) -> list[str]:
    """Build the source-only BiCAD-XR command for one reserved row."""

    if not row.source_only or any(
        (
            row.target_access,
            row.phase2_access,
            row.support_access,
            row.query_access,
            row.truth_access,
        )
    ):
        raise ValueError("BiCAD-XR rows must be source-only")
    config = candidate_config(row.candidate_id)
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
        "--batch_size",
        str(config.batch_size),
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
                "fold": row.fold,
                "optimizer_updates": row.optimizer_updates,
                "row_id": row.row_id,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--candidate_id",
        row.candidate_id,
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
    if row.stage == "pairbicad_convergence":
        command.extend(
            [
                "--bicad_optimizer_updates",
                str(PAIRBICAD_CONVERGENCE_OPTIMIZER_UPDATES),
                "--bicad_loro_receiver",
                str(LORO_HELDOUT_RECEIVER[row.fold]),
                "--bicad_loro_eval_interval_updates",
                str(PAIRBICAD_UPDATE_INTERVAL),
                "--bicad_loro_min_updates",
                str(PAIRBICAD_FINAL_MIN_UPDATES),
                "--bicad_loro_patience",
                "5",
            ]
        )
    elif row.stage == "pairbicad_final":
        command.extend(
            [
                "--bicad_optimizer_updates",
                str(row.optimizer_updates),
                "--bicad_loro_eval_interval_updates",
                "0",
            ]
        )
    option_names = {token.lower() for token in command if token.startswith("--")}
    forbidden = {
        option for option in option_names if any(token in option for token in _FORBIDDEN_OPTION_TOKENS)
    }
    if forbidden:
        raise ValueError("source-only training command contains forbidden data-role options")
    return command


def _row_payload(row: PlanRow) -> dict[str, Any]:
    # Deliberately omit forbidden data-role fields from the dry-run surface.
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
    }


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    return path


def _write_json_atomic_once(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish a completion marker atomically without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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


def _load_ssdg_module(code_root: Path) -> Any:
    path = Path(code_root) / "code" / "SSDG" / "train_ssdg.py"
    code_path = str(Path(code_root) / "code")
    ssdg_path = str(path.parent)
    for value in (code_path, ssdg_path):
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("bicad_xr_ssdg_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load BiCAD-XR training runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruction_config(
    candidate_id: str,
    *,
    optimizer_updates: int | None = None,
) -> Any:
    """Resolve the candidate config used for strict final-checkpoint restore."""

    from dataclasses import replace as dataclass_replace

    updates = {}
    if optimizer_updates is not None:
        if isinstance(optimizer_updates, bool) or not isinstance(optimizer_updates, int):
            raise ValueError("optimizer_updates must be an integer")
        if optimizer_updates <= 0:
            raise ValueError("optimizer_updates must be positive")
        updates["optimizer_updates"] = optimizer_updates
    return dataclass_replace(
        candidate_config(candidate_id),
        phase1_method="bicad_xr",
        use_fasttrust=False,
        use_mixstyle=False,
        **updates,
    )


class _FormalEvaluationContext:
    """Lazily construct the source-LORO evaluator after training exits cleanly."""

    def __init__(
        self,
        row: PlanRow,
        roots: LauncherRoots,
        command: Sequence[str],
    ) -> None:
        self.row = row
        self.roots = roots
        self.command = tuple(command)
        self.ssdg: Any = None
        self.ssdg_args: Any = None
        self.device: Any = None
        self.loader: Any = None
        self.trainer: Any = None

    def _ensure_runtime(self) -> None:
        if self.ssdg is not None:
            return
        import torch

        self.ssdg = _load_ssdg_module(Path(self.roots.code_root))
        self.ssdg_args = self.ssdg.parse(list(self.command[3:]))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        payload = self.ssdg.load_wisig_compact_pkl(self.ssdg_args.wisig_pkl)
        day_list = list(payload.get("capture_date_list", []))
        receiver_list = list(payload.get("rx_list", []))
        heldout = tuple(set(SOURCE_RECEIVERS) - set(self.row.source_receivers))
        if len(heldout) != 1:
            raise ValueError("row must identify exactly one source-LORO heldout receiver")
        days = self.ssdg._resolve_days(
            day_list, list(self.row.train_days), list(self.row.train_days)
        )
        receivers = self.ssdg._resolve_rxs(
            receiver_list, [heldout[0]], [heldout[0]]
        )
        dataset = self.ssdg.WiSigCompactDataset(
            payload,
            out_len=int(self.ssdg_args.wisig_out_len),
            crop_mode="center",
            normalize=True,
            equalized=1,
            day_keep=days,
            rx_keep=receivers,
            domain="rx_day",
            seed=int(self.row.seed),
            build_index=True,
        )
        self.loader = self.ssdg.make_loader(
            dataset,
            int(self.ssdg_args.eval_batch_size),
            False,
            int(self.ssdg_args.num_workers),
            self.device,
            False,
            int(self.ssdg_args.prefetch_factor),
        )

    def build_model(self, payload: Mapping[str, Any]) -> Any:
        from types import SimpleNamespace

        self._ensure_runtime()
        model_args = payload.get("args")
        if not isinstance(model_args, Mapping):
            raise ValueError("checkpoint is missing recorded training args")
        model = self.ssdg.build_baseline_model(
            SimpleNamespace(**dict(model_args)), self.device
        )
        model.eval()
        return model

    def restore_trainer_runtime(self, model: Any, payload: Mapping[str, Any]) -> None:
        from cvsrffi.phase1_bicad_xr.trainer import BiCADXRTrainer

        self._ensure_runtime()
        runtime = payload.get("bicad_xr_runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError("checkpoint is missing bicad_xr_runtime")
        config = reconstruction_config(
            self.row.candidate_id,
            optimizer_updates=self.row.optimizer_updates,
        )
        self.trainer = BiCADXRTrainer(
            self.ssdg._BiCADXRConcatForward(model),
            config,
            num_receivers=len(self.row.source_receivers),
            num_days=len(self.row.train_days),
            num_channels=2 if config.strict_pair_concat else 4,
        ).to(self.device)
        self.trainer.load_checkpoint_runtime(runtime, strict=True)

    def evaluate(self, model: Any, scenario: str) -> dict[str, Any]:
        import torch

        self._ensure_runtime()
        scenario_index = (
            "clean",
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        ).index(scenario)
        generator = torch.Generator(device=self.device).manual_seed(
            int(self.row.seed) + scenario_index * 1_000_003
        )
        correct = 0
        total = 0
        class_correct: dict[int, int] = {}
        class_total: dict[int, int] = {}
        model.eval()
        with torch.no_grad():
            for raw in self.loader:
                x = raw[0].to(self.device)
                labels = raw[1].to(self.device).reshape(-1).long()
                if scenario != "clean":
                    received = self.ssdg.apply_sat_channel_for_scenario(
                        x,
                        scenario,
                        self.ssdg_args,
                        gen=generator,
                        return_meta=False,
                    )
                    x = received[0] if isinstance(received, tuple) else received
                output = model(x)
                if isinstance(output, Mapping):
                    logits = output.get("tx_logits", output.get("logits"))
                else:
                    logits = output
                if not torch.is_tensor(logits) or logits.ndim != 2:
                    raise ValueError(f"invalid TX logits for {scenario}")
                prediction = logits.argmax(dim=1)
                matches = prediction.eq(labels)
                correct += int(matches.sum().item())
                total += int(labels.numel())
                for class_id in labels.unique().tolist():
                    mask = labels == int(class_id)
                    count = int(mask.sum().item())
                    class_total[int(class_id)] = class_total.get(int(class_id), 0) + count
                    class_correct[int(class_id)] = class_correct.get(int(class_id), 0) + int(
                        matches[mask].sum().item()
                    )
        if total <= 0:
            raise ValueError(f"empty final evaluation loader for {scenario}")
        per_class = {
            str(class_id): class_correct.get(class_id, 0) / count
            for class_id, count in sorted(class_total.items())
        }
        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "per_class_accuracy": per_class,
            "floor_accuracy": min(per_class.values()) if per_class else None,
            "log": (
                f"scenario={scenario} checkpoint=bicad_xr_final.pth "
                f"correct={correct} total={total}\n"
            ),
        }


def write_plan_json(run_root: str | Path, rows: Sequence[PlanRow], *, run_id: str) -> Path:
    root = Path(run_root)
    return _write_json_once(
        root / "plan.json",
        {
            "run_id": str(run_id),
            "protocol": "phase1_source_only_bicad_xr",
            "row_count": len(rows),
            "rows": [_row_payload(row) for row in rows],
        },
    )


def write_final_status(
    run_root: str | Path,
    rows: Sequence[PlanRow],
    statuses: Mapping[str, str],
) -> Path:
    missing = [row.row_id for row in rows if row.row_id not in statuses]
    nonterminal = {
        row.row_id: statuses.get(row.row_id)
        for row in rows
        if statuses.get(row.row_id) not in FORMAL_STATES
    }
    if missing or nonterminal:
        raise ValueError(f"all rows must be terminal; missing={missing}, nonterminal={nonterminal}")
    return _write_json_once(
        Path(run_root) / "final_status.json",
        {
            "row_count": len(rows),
            "statuses": {row.row_id: statuses[row.row_id] for row in rows},
        },
    )


def run_plan(
    rows: Sequence[PlanRow],
    row_runner: Callable[[PlanRow], str],
    *,
    max_active_per_gpu: int = MAX_JOBS_PER_GPU,
) -> dict[str, str]:
    """Run callbacks with a hard per-GPU concurrency cap."""

    limit = _validated_max_jobs(max_active_per_gpu)
    if not rows:
        return {}
    semaphores = {}
    for row in rows:
        semaphores.setdefault(row.gpu_id, threading.BoundedSemaphore(limit))

    def guarded(row: PlanRow) -> tuple[str, str]:
        with semaphores[row.gpu_id]:
            return row.row_id, str(row_runner(row))

    statuses: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        futures = [pool.submit(guarded, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            row_id, status = future.result()
            statuses[row_id] = status
    return statuses


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
            "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
            "reason": str(reason),
            "details": dict(details or {}),
        },
    )


def _locate_final_checkpoint(row_root: Path) -> Path:
    for name in ("bicad_xr_final.pth", "final_checkpoint.pt", "final_bicad_xr.pt"):
        checkpoint = row_root / name
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return checkpoint
    raise FileNotFoundError("row final checkpoint is missing or empty")


_SOURCE_LORO_ACCESS_FLAGS = (
    "target_access",
    "phase2_access",
    "support_access",
    "query_access",
    "truth_access",
)


def _source_loro_row_identity_matches(
    record: Mapping[str, Any],
    row: PlanRow,
    *,
    label: str,
) -> None:
    """Check optional row identity fields without requiring trainer extensions."""

    expected: dict[str, Any] = {
        "candidate_id": row.candidate_id,
        "fold": row.fold,
        "seed": row.seed,
        "source_receivers": tuple(row.source_receivers),
        "train_days": tuple(row.train_days),
    }
    for name, expected_value in expected.items():
        if name not in record:
            continue
        actual_value = record[name]
        if name in {"source_receivers", "train_days"}:
            try:
                matches = tuple(int(value) for value in actual_value) == expected_value
            except (TypeError, ValueError):
                matches = False
        else:
            matches = type(actual_value) is type(expected_value) and actual_value == expected_value
        if not matches:
            raise ValueError(f"{label} field {name} does not match row {row.row_id}")
    for name in ("optimizer_updates", "planned_optimizer_updates"):
        if name in record:
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, int) or value != row.optimizer_updates:
                raise ValueError(f"{label} field {name} does not match planned row budget")


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _convergence_runtime_expectation(
    row_root: str | Path,
    row: PlanRow,
) -> dict[str, Any]:
    """Validate source-LORO closure and return the actual runtime expectation."""

    if row.stage != "pairbicad_convergence":
        raise ValueError("source-LORO runtime expectation requires pairbicad_convergence")
    root = Path(row_root)
    planned_updates = PAIRBICAD_CONVERGENCE_OPTIMIZER_UPDATES
    selection = _load_json_mapping(
        root / "source_loro_selection.json",
        label="source_loro_selection.json",
    )
    _source_loro_row_identity_matches(selection, row, label="source_loro_selection.json")
    required_selection = ("planned_updates", "stop_update", "best_update", "patience", "interval")
    missing = [name for name in required_selection if name not in selection]
    if missing:
        raise ValueError(
            "source_loro_selection.json is missing fields: " + ",".join(missing)
        )
    if selection.get("planned_updates") != planned_updates:
        raise ValueError("source-LORO selection planned_updates does not match row")
    stop_update = selection.get("stop_update")
    best_update = selection.get("best_update")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (stop_update, best_update)
    ):
        raise ValueError("source-LORO selection update fields must be integers")
    if not (
        PAIRBICAD_FINAL_MIN_UPDATES <= stop_update <= planned_updates
        and stop_update % PAIRBICAD_UPDATE_INTERVAL == 0
    ):
        raise ValueError("source-LORO stop_update is outside the planned evaluation clock")
    if not (
        PAIRBICAD_FINAL_MIN_UPDATES <= best_update <= stop_update
        and best_update % PAIRBICAD_UPDATE_INTERVAL == 0
    ):
        raise ValueError("source-LORO best_update is outside the planned evaluation clock")
    if selection.get("patience") != 5 or selection.get("interval") != PAIRBICAD_UPDATE_INTERVAL:
        raise ValueError("source-LORO selection controls do not match convergence settings")
    if selection.get("source_only") is not True:
        raise ValueError("source-LORO selection must be source-only")
    for name in _SOURCE_LORO_ACCESS_FLAGS:
        if selection.get(name) is not False:
            raise ValueError(f"source-LORO selection {name} must be false")

    curve_path = root / "source_loro_curve.jsonl"
    if not curve_path.is_file() or curve_path.stat().st_size <= 0:
        raise FileNotFoundError("source_loro_curve.jsonl is missing or empty")
    records: list[dict[str, Any]] = []
    try:
        lines = curve_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("source_loro_curve.jsonl cannot be read") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"source_loro_curve.jsonl line {line_number} is invalid JSON") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"source_loro_curve.jsonl line {line_number} is not an object")
        record = dict(record)
        _source_loro_row_identity_matches(
            record,
            row,
            label=f"source_loro_curve.jsonl line {line_number}",
        )
        if record.get("planned_updates") != planned_updates:
            raise ValueError("source-LORO curve planned_updates does not match row")
        if record.get("source_only") is not True:
            raise ValueError("source-LORO curve must be source-only")
        for name in _SOURCE_LORO_ACCESS_FLAGS:
            if record.get(name) is not False:
                raise ValueError(f"source-LORO curve {name} must be false")
        update = record.get("update")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update < PAIRBICAD_FINAL_MIN_UPDATES
            or update > planned_updates
            or update % PAIRBICAD_UPDATE_INTERVAL != 0
        ):
            raise ValueError("source-LORO curve contains an invalid update")
        if records and update <= records[-1]["update"]:
            raise ValueError("source-LORO curve updates must increase strictly")
        records.append(record)
    if not records:
        raise ValueError("source_loro_curve.jsonl has no records")
    if records[-1]["update"] != stop_update:
        raise ValueError("source-LORO curve does not end at selection stop_update")

    best_snapshot = root / "source_loro" / f"checkpoint_u{best_update}.pth"
    declared_snapshot = selection.get("best_snapshot")
    if isinstance(declared_snapshot, str) and declared_snapshot.strip():
        declared_path = (root / declared_snapshot).resolve()
        if root.resolve() not in declared_path.parents:
            raise ValueError("source-LORO best snapshot escapes the row root")
        if declared_path != best_snapshot.resolve() and declared_path.name != "source_loro_best.pth":
            raise ValueError("source-LORO best snapshot does not match best_update")
        best_snapshot = declared_path
    if not best_snapshot.is_file() or best_snapshot.stat().st_size <= 0:
        raise FileNotFoundError(
            "source-LORO best-update snapshot is missing or empty: "
            f"{best_snapshot.name}"
        )

    expectation = _row_expectation(row, optimizer_updates=stop_update)
    expectation["planned_optimizer_updates"] = planned_updates
    expectation["source_loro_stop_update"] = stop_update
    expectation["source_loro_best_update"] = best_update
    expectation["source_loro_best_snapshot"] = str(
        best_snapshot.resolve().relative_to(root.resolve())
    )
    return expectation


def launch_row_process(row: PlanRow, roots: LauncherRoots, *, run_id: str) -> str:
    """Launch one row in its already-reserved directory and retain its log."""

    row_root = _row_root(row, roots)
    if not row_root.is_dir():
        raise FileNotFoundError(f"reserved row root is missing: {row_root}")
    command = build_train_command(row, roots, run_id=run_id)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(row.gpu_id)
    log_path = row_root / "train.log"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite row log: {log_path}")
    with log_path.open("x", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        _record_technical_failure(
            row_root,
            reason="TRAINING_SUBPROCESS_FAILED",
            details={"returncode": int(completed.returncode)},
        )
        return "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"

    if row.stage == "pairbicad_convergence":
        try:
            runtime_expectation = _convergence_runtime_expectation(row_root, row)
        except (FileNotFoundError, OSError, ValueError) as exc:
            _record_technical_failure(
                row_root,
                reason="SOURCE_LORO_CLOSURE_FAILED",
                details={"exception_type": type(exc).__name__, "message": str(exc)},
            )
            return "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    else:
        runtime_expectation = _row_expectation(row)

    try:
        checkpoint = _locate_final_checkpoint(row_root)
        context = _FormalEvaluationContext(row, roots, command)
        evaluation = evaluate_final_checkpoint(
            checkpoint,
            expected_runtime=runtime_expectation,
            output_dir=row_root,
            model_builder=context.build_model,
            trainer_runtime_restorer=context.restore_trainer_runtime,
            evaluator=context.evaluate,
        )
        closure = validate_artifact_closure(row_root)
        if row.stage == "pairbicad_convergence":
            closure = {
                **dict(closure),
                "source_loro": {
                    "planned_optimizer_updates": row.optimizer_updates,
                    "stop_update": runtime_expectation["optimizer_updates"],
                    "best_update": runtime_expectation["source_loro_best_update"],
                    "best_snapshot": runtime_expectation["source_loro_best_snapshot"],
                },
            }
        if not bool(evaluation.get("complete")) or not bool(closure.get("complete")):
            _record_technical_failure(
                row_root,
                reason="FINAL_ARTIFACT_CLOSURE_INCOMPLETE",
                details={"evaluation": evaluation, "closure": closure},
            )
            return "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
        _write_json_atomic_once(row_root / "ARTIFACTS_COMPLETE.json", closure)
        return "ARTIFACTS_COMPLETE"
    except Exception as exc:
        _record_technical_failure(
            row_root,
            reason="FINAL_ARTIFACT_EVALUATION_FAILED",
            details={"exception_type": type(exc).__name__, "message": str(exc)},
        )
        return "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"


def _csv_ints(raw: str, *, name: str, minimum: int = 1) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    return _unique_ints(values, name=name, minimum=minimum)


def _csv_strings(raw: str, *, name: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a non-empty unique comma-separated list")
    return values


def _max_jobs_arg(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max_jobs_per_gpu must be an integer") from exc
    try:
        return _validated_max_jobs(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan a source-only BiCAD-XR Phase1 matrix.")
    parser.add_argument(
        "--stage",
        choices=(
            "quick",
            "confirm",
            "full",
            "pairbicad",
            "pairbicad_convergence",
            "pairbicad_final",
        ),
        default="quick",
    )
    parser.add_argument("--candidates", default="")
    parser.add_argument("--folds", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--gpu-ids", dest="gpu_ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--max-jobs-per-gpu",
        dest="max_jobs_per_gpu",
        type=_max_jobs_arg,
        default=None,
    )
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--output-root", dest="output_root", default="")
    parser.add_argument("--code-root", dest="code_root", default="")
    parser.add_argument("--python", dest="python_path", default=sys.executable)
    parser.add_argument("--wisig-pkl", dest="wisig_pkl", default="")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-row", action="store_true")
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--source-rxs", default="")
    parser.add_argument("--train-days", default="")
    parser.add_argument("--optimizer-updates", type=int, default=None)
    parser.add_argument("--row-root", default="")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    stage_key = str(args.stage).strip().lower()
    if args.max_jobs_per_gpu is None:
        args.max_jobs_per_gpu = (
            PAIRBICAD_MAX_JOBS_PER_GPU
            if stage_key in {"pairbicad", *PAIRBICAD_CONVERGENCE_STAGES}
            else MAX_JOBS_PER_GPU
        )
    if stage_key in PAIRBICAD_CONVERGENCE_STAGES and args.max_jobs_per_gpu != PAIRBICAD_MAX_JOBS_PER_GPU:
        raise ValueError(
            f"{stage_key} stage requires max_jobs_per_gpu={PAIRBICAD_MAX_JOBS_PER_GPU}"
        )
    if args.optimizer_updates is None:
        args.optimizer_updates = (
            PAIRBICAD_OPTIMIZER_UPDATES
            if stage_key == "pairbicad"
            else PAIRBICAD_CONVERGENCE_OPTIMIZER_UPDATES
            if stage_key == "pairbicad_convergence"
            else None
            if stage_key == "pairbicad_final"
            else OPTIMIZER_UPDATES
        )
    return args


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    code_root = Path(args.code_root).resolve() if args.code_root else Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root) if args.output_root else code_root / "runs"
    wisig_pkl = Path(args.wisig_pkl) if args.wisig_pkl else code_root / "Dataset_WigSig" / "ManySig.pkl"
    return code_root, output_root, Path(args.python_path), wisig_pkl


def _worker_row(args: argparse.Namespace) -> int:
    if not args.candidate_id or not args.fold or not args.seed or not args.row_root:
        raise ValueError("worker-row requires candidate-id, fold, seed and row-root")
    source_rxs = _csv_ints(args.source_rxs, name="source-rxs")
    train_days = _csv_ints(args.train_days, name="train-days")
    row = PlanRow(
        candidate_id=_canonical_candidate(args.candidate_id),
        fold=args.fold,
        seed=args.seed,
        optimizer_updates=args.optimizer_updates,
        gpu_id=args.gpu_id,
        source_receivers=source_rxs,
        train_days=train_days,
        stage=str(args.stage).strip().lower(),
    )
    code_root, _, python_path, wisig_pkl = _resolve_paths(args)
    roots = LauncherRoots(code_root, python_path, Path(args.row_root).resolve().parent, wisig_pkl)
    return 0 if launch_row_process(row, roots, run_id=args.run_id) == "ARTIFACTS_COMPLETE" else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_row:
        return _worker_row(args)

    candidates = _csv_strings(args.candidates, name="candidates") if args.candidates else None
    folds = _csv_ints(args.folds, name="folds") if args.folds else None
    seeds = _csv_ints(args.seeds, name="seeds") if args.seeds else None
    gpu_ids = _csv_ints(args.gpu_ids, name="gpu-ids", minimum=0)
    rows = build_plan(
        stage=args.stage,
        candidates=candidates,
        folds=folds,
        seeds=seeds,
        gpu_ids=gpu_ids,
        optimizer_updates=args.optimizer_updates,
    )
    stage_key = str(args.stage).strip().lower()
    if stage_key == "pairbicad":
        if args.max_jobs_per_gpu != PAIRBICAD_MAX_JOBS_PER_GPU:
            raise ValueError(
                f"pairbicad stage requires max_jobs_per_gpu={PAIRBICAD_MAX_JOBS_PER_GPU}"
            )
        rows = queue_rows(rows, gpu_ids=gpu_ids)
    elif stage_key in PAIRBICAD_CONVERGENCE_STAGES:
        if args.max_jobs_per_gpu != PAIRBICAD_MAX_JOBS_PER_GPU:
            raise ValueError(
                f"{stage_key} stage requires max_jobs_per_gpu={PAIRBICAD_MAX_JOBS_PER_GPU}"
            )
        rows = pack_rows(rows, gpu_ids=gpu_ids, max_jobs_per_gpu=args.max_jobs_per_gpu)
    else:
        rows = pack_rows(rows, gpu_ids=gpu_ids, max_jobs_per_gpu=args.max_jobs_per_gpu)

    if args.dry_run:
        for row in rows:
            print(json.dumps(_row_payload(row), ensure_ascii=False, sort_keys=True))
        return 0

    if not args.run_id:
        raise ValueError("run-id is required for a real launch")
    code_root, output_root, python_path, wisig_pkl = _resolve_paths(args)
    run_root = reserve_run_layout(output_root, args.run_id, rows)
    write_plan_json(run_root, rows, run_id=args.run_id)
    roots = LauncherRoots(code_root, python_path, run_root, wisig_pkl)
    statuses = run_plan(
        rows,
        lambda row: launch_row_process(row, roots, run_id=args.run_id),
        max_active_per_gpu=args.max_jobs_per_gpu,
    )
    write_final_status(run_root, rows, statuses)
    return 0 if all(value == "ARTIFACTS_COMPLETE" for value in statuses.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
