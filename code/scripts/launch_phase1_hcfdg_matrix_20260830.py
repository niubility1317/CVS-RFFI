#!/usr/bin/env python3
"""Launch the source-only Phase1 HCF-DG screening matrix.

The public planning helpers are import-safe and side-effect free. Runtime
training is entered only through ``--worker-row``; matrix dispatch never
opens a target, Phase2, query, or truth input.
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
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvsrffi.phase1_hcfdg.config import (
    MatrixRow,
    SELECTION_SEEDS,
    StageBudget,
    V2_OPTIMIZER_UPDATES,
    deep_screen_rows,
    quick_screen_rows,
    residual_rows,
)


RUN_ID_DEFAULT = "phase1_hcfdg_v2_source_loro_20260830_r1"
HCFDG_SMOKE_STAGE_BUDGET = StageBudget(
    stage0=1,
    stage1=1,
    stage2=1,
    stage3=1,
    stage4=0,
    freeze_progress=0.50,
    environment_update_interval=4,
    environment_updates_per_interval=1,
)
SOURCE_RECEIVERS = (1, 3, 4, 6, 8)
TRAIN_DAYS = (1, 2, 3)
FINAL_SCENARIOS = (
    "clean",
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
TERMINAL_STATES = {
    "ARTIFACTS_COMPLETE",
    "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
}


class PlanRow(NamedTuple):
    candidate_id: str
    heldout_receiver: int
    seed: int
    optimizer_updates: int
    gpu: int
    source_receivers: tuple[int, ...]
    train_days: tuple[int, ...]
    v2_parent_candidate_id: str | None = None

    @property
    def row_id(self) -> str:
        return f"{self.candidate_id}-F{self.heldout_receiver}-S{self.seed}"


class LauncherRoots(NamedTuple):
    code_root: Path
    python: Path
    run_root: Path
    wisig_pkl: Path


def _rows_for_stage(
    stage: str,
    folds: Sequence[int],
    *,
    v2_passed: bool,
    v2_parent_candidate_id: str,
) -> tuple[MatrixRow, ...]:
    key = str(stage).strip().lower()
    if key == "quick":
        return quick_screen_rows(folds)
    if key == "deep":
        return deep_screen_rows(folds)
    if key == "residual":
        return residual_rows(
            folds,
            v2_passed=v2_passed,
            v2_parent_candidate_id=v2_parent_candidate_id,
        )
    raise ValueError(f"unknown stage: {stage}")


def _validated_seeds(seeds: Sequence[int] | None) -> tuple[int, ...]:
    if seeds is None:
        return tuple(SELECTION_SEEDS)
    if isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must contain integer IDs")
    try:
        resolved = tuple(seeds)
    except TypeError as exc:
        raise ValueError("seeds must contain integer IDs") from exc
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError("seeds must be a non-empty unique list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in resolved):
        raise ValueError("seeds must contain integer IDs")
    if any(seed not in SELECTION_SEEDS for seed in resolved):
        raise ValueError(f"seeds must be selected from {SELECTION_SEEDS}")
    return tuple(resolved)


def build_plan(
    *,
    stage: str = "quick",
    folds: Sequence[int] = (1, 8),
    gpus: Sequence[int] = tuple(range(8)),
    seeds: Sequence[int] | None = None,
    v2_passed: bool = False,
    v2_parent_candidate_id: str = "A9",
) -> list[PlanRow]:
    """Build the immutable source-LORO matrix in report order."""

    selected_gpus = tuple(int(value) for value in gpus)
    if not selected_gpus or len(selected_gpus) != len(set(selected_gpus)):
        raise ValueError("gpus must be a non-empty unique sequence")
    if any(value < 0 for value in selected_gpus):
        raise ValueError("gpu IDs must be non-negative")
    selected_seeds = _validated_seeds(seeds)
    base_rows = _rows_for_stage(
        stage,
        folds,
        v2_passed=v2_passed,
        v2_parent_candidate_id=v2_parent_candidate_id,
    )
    base_rows = tuple(row for row in base_rows if int(row.seed) in selected_seeds)
    if not base_rows:
        raise ValueError("seed selection produced no matrix rows")
    rows: list[PlanRow] = []
    for index, row in enumerate(base_rows):
        heldout = int(row.heldout_receiver)
        if heldout not in SOURCE_RECEIVERS:
            raise ValueError(f"heldout receiver {heldout} is not a registered source receiver")
        source_receivers = tuple(value for value in SOURCE_RECEIVERS if value != heldout)
        rows.append(
            PlanRow(
                candidate_id=str(row.candidate_id),
                heldout_receiver=heldout,
                seed=int(row.seed),
                optimizer_updates=int(row.optimizer_updates),
                gpu=selected_gpus[index % len(selected_gpus)],
                source_receivers=source_receivers,
                train_days=TRAIN_DAYS,
                v2_parent_candidate_id=row.v2_parent_candidate_id,
            )
        )
    return rows


def validate_output_root(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists():
        raise FileExistsError(f"output root already exists: {resolved}")
    return resolved


def _row_root(row: PlanRow, roots: LauncherRoots) -> Path:
    return Path(roots.run_root) / row.row_id


def build_train_command(row: PlanRow, roots: LauncherRoots) -> list[str]:
    script = Path(roots.code_root) / "code" / "scripts" / Path(__file__).name
    command = [
        str(roots.python),
        "-u",
        str(script),
        "--worker-row",
        "--candidate-id",
        row.candidate_id,
        "--heldout-rx",
        str(row.heldout_receiver),
        "--source-rxs",
        ",".join(map(str, row.source_receivers)),
        "--train-days",
        ",".join(map(str, row.train_days)),
        "--optimizer-updates",
        str(row.optimizer_updates),
        "--seed",
        str(row.seed),
        "--gpu",
        str(row.gpu),
        "--wisig-pkl",
        str(roots.wisig_pkl),
        "--row-root",
        str(_row_root(row, roots)),
    ]
    option_names = {token.lower() for token in command if token.startswith("--")}
    forbidden_options = {
        "--phase2",
        "--phase2-path",
        "--target",
        "--target-rxs",
        "--query",
        "--query-path",
        "--truth",
        "--truth-path",
    }
    if option_names & forbidden_options:
        raise ValueError("Phase1 training command contains a forbidden data-role token")
    return command


def _row_payload(row: PlanRow) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "candidate_id": row.candidate_id,
        "heldout_receiver": row.heldout_receiver,
        "source_receivers": list(row.source_receivers),
        "train_days": list(row.train_days),
        "seed": row.seed,
        "optimizer_updates": row.optimizer_updates,
        "gpu": row.gpu,
        "v2_parent_candidate_id": row.v2_parent_candidate_id,
    }


def _write_new_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_plan_json(run_root: str | Path, rows: Sequence[PlanRow], *, run_id: str) -> Path:
    root = Path(run_root)
    selected_seeds = list(dict.fromkeys(int(row.seed) for row in rows))
    return _write_new_json(
        root / "plan.json",
        {
            "run_id": str(run_id),
            "protocol": "phase1_source_only_hcfdg",
            "row_count": len(rows),
            "seeds": selected_seeds,
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
        if statuses.get(row.row_id) not in TERMINAL_STATES
    }
    if missing or nonterminal:
        raise ValueError(f"all rows must be terminal; missing={missing}, nonterminal={nonterminal}")
    return _write_new_json(
        Path(run_root) / "final_status.json",
        {
            "row_count": len(rows),
            "statuses": {row.row_id: statuses[row.row_id] for row in rows},
        },
    )


def validate_artifact_closure(row_root: str | Path) -> dict[str, Any]:
    root = Path(row_root)
    checkpoint = root / "final_hcfdg.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError("final checkpoint is missing or empty")
    metrics: dict[str, Any] = {}
    for scenario in FINAL_SCENARIOS:
        metric_path = root / f"eval_{scenario}.json"
        log_path = root / f"eval_{scenario}.log"
        if not metric_path.is_file() or not log_path.is_file():
            raise ValueError(f"missing final evaluation artifact for {scenario}")
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        if payload.get("scenario") != scenario:
            raise ValueError(f"scenario identity mismatch for {scenario}")
        if payload.get("checkpoint_load_strict") is not True:
            raise ValueError(f"strict checkpoint reconstruction missing for {scenario}")
        for key in ("missing_keys", "unexpected_keys", "shape_mismatches"):
            if payload.get(key) not in ([], None):
                raise ValueError(f"strict reconstruction failure for {scenario}: {key}")
        metrics[scenario] = payload
    return {
        "status": "ARTIFACTS_COMPLETE",
        "checkpoint": str(checkpoint),
        "scenarios": metrics,
    }


def evaluate_final_checkpoint(
    row: PlanRow,
    row_root: str | Path,
    *,
    reconstruct_fn: Callable[[Path], tuple[Any, Mapping[str, Any]]],
    evaluate_fn: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(row_root)
    checkpoint = root / "final_hcfdg.pt"
    model, audit = reconstruct_fn(checkpoint)
    if any(audit.get(key) for key in ("missing_keys", "unexpected_keys", "shape_mismatches")):
        raise ValueError(f"strict checkpoint reconstruction failed: {dict(audit)}")
    written: dict[str, Any] = {}
    for scenario in FINAL_SCENARIOS:
        metrics = dict(evaluate_fn(model, scenario, row))
        payload = {
            **metrics,
            "scenario": scenario,
            "checkpoint": str(checkpoint),
            "checkpoint_load_strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
        }
        (root / f"eval_{scenario}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / f"eval_{scenario}.log").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written[scenario] = payload
    return written


def run_plan(
    rows: Sequence[PlanRow],
    row_runner: Callable[[PlanRow], str],
    *,
    max_active_per_gpu: int = 2,
) -> dict[str, str]:
    if max_active_per_gpu != 2:
        raise ValueError("formal HCF-DG packing requires exactly two active rows per GPU")
    if not rows:
        return {}
    semaphores = {row.gpu: threading.BoundedSemaphore(2) for row in rows}

    def guarded(row: PlanRow) -> tuple[str, str]:
        with semaphores[row.gpu]:
            return row.row_id, str(row_runner(row))

    statuses: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(semaphores) * 2) as pool:
        futures = [pool.submit(guarded, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            row_id, status = future.result()
            statuses[row_id] = status
    return statuses


def run_row(row: PlanRow, roots: LauncherRoots) -> str:
    row_root = _row_root(row, roots)
    if row_root.exists():
        raise FileExistsError(f"row output already exists: {row_root}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(row.gpu)
    completed = subprocess.run(build_train_command(row, roots), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"worker returned exit status {completed.returncode}")
    return validate_artifact_closure(row_root)["status"]


def _load_ssdg_module(code_root: Path):
    path = code_root / "code" / "SSDG" / "train_ssdg.py"
    spec = importlib.util.spec_from_file_location("hcfdg_ssdg_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source data runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_int_csv(value: str, *, name: str) -> tuple[int, ...]:
    try:
        resolved = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError(f"{name} must be a non-empty unique list")
    return resolved


def _bool_tensor(values: Iterable[Any]):
    import torch

    return torch.tensor([bool(value) for value in values], dtype=torch.bool)


def _content_keys_from_iq(iq: Any):
    """Build a small finite content key from the current IQ only.

    The key removes per-sample absolute gain before extracting normalized
    amplitude statistics, short-lag temporal autocorrelation, coarse spectral
    mass, low-order differences, and local-energy changes.  Its dimension is
    fixed and independent of the IQ capture length.
    """

    import torch
    from torch.nn import functional as functional

    values = torch.as_tensor(iq, dtype=torch.float32)
    if values.ndim < 2:
        raise ValueError("IQ content keys require a batch dimension and samples")
    sample_count = int(values.shape[0])
    if sample_count <= 0:
        raise ValueError("IQ content keys require at least one sample")
    if values.ndim == 2:
        values = values.unsqueeze(1)
    elif values.ndim == 3:
        if values.shape[1] <= 4 and values.shape[2] > values.shape[1]:
            pass
        elif values.shape[2] <= 4 and values.shape[1] > values.shape[2]:
            values = values.transpose(1, 2)
        else:
            values = values.reshape(sample_count, 1, -1)
    else:
        values = values.reshape(sample_count, 1, -1)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[-1] <= 0:
        raise ValueError("IQ content keys require non-empty samples")

    if values.shape[1] >= 2:
        amplitude = torch.sqrt(values[:, 0].square() + values[:, 1].square())
    else:
        amplitude = values[:, 0].abs()
    gain = torch.sqrt(amplitude.square().mean(dim=1, keepdim=True)).clamp_min(1e-6)
    normalized = amplitude / gain
    mean = normalized.mean(dim=1)
    std = normalized.std(dim=1, unbiased=False)
    amplitude_stats = torch.stack(
        (mean, std, normalized.amin(dim=1), normalized.amax(dim=1)), dim=1
    )

    centered = normalized - mean.unsqueeze(1)
    variance = centered.square().mean(dim=1).clamp_min(1e-6)
    autocorrelations = []
    for lag in (1, 2, 4, 8):
        if normalized.shape[1] > lag:
            value = (centered[:, :-lag] * centered[:, lag:]).mean(dim=1) / variance
        else:
            value = torch.zeros_like(mean)
        autocorrelations.append(value)
    autocorrelation_stats = torch.stack(autocorrelations, dim=1)

    spectrum = torch.fft.rfft(normalized, dim=1).abs().square()
    spectrum = functional.adaptive_avg_pool1d(spectrum.unsqueeze(1), 8).squeeze(1)
    spectrum = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def difference_stats(order: int):
        difference = normalized
        for _ in range(order):
            difference = difference[:, 1:] - difference[:, :-1]
        if difference.shape[1] == 0:
            return torch.zeros(sample_count, 2, device=normalized.device, dtype=normalized.dtype)
        return torch.stack(
            (difference.abs().mean(dim=1), difference.std(dim=1, unbiased=False)),
            dim=1,
        )

    difference_stats_value = torch.cat((difference_stats(1), difference_stats(2)), dim=1)

    local_energy = functional.adaptive_avg_pool1d(
        normalized.square().unsqueeze(1), 4
    ).squeeze(1)
    local_delta = local_energy[:, 1:] - local_energy[:, :-1]
    local_energy_stats = torch.cat(
        (
            local_energy,
            local_delta.abs().mean(dim=1, keepdim=True),
            local_delta.std(dim=1, unbiased=False).unsqueeze(1),
        ),
        dim=1,
    )
    result = torch.cat(
        (
            amplitude_stats,
            autocorrelation_stats,
            spectrum,
            difference_stats_value,
            local_energy_stats,
        ),
        dim=1,
    )
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_worker_config(candidate_id: str, *, smoke: bool):
    from dataclasses import replace

    from cvsrffi.phase1_hcfdg.config import candidate_config

    key = str(candidate_id).strip().upper()
    config = candidate_config(key)
    if not smoke:
        return config
    if key not in {f"A{i}" for i in range(6, 13)}:
        raise ValueError("HCF-DG V2 smoke requires an A6-A12 candidate")
    return replace(
        config,
        optimizer_updates=HCFDG_SMOKE_STAGE_BUDGET.total_updates,
    )


def _source_runtime_args(ssdg: Any, args: argparse.Namespace, representation_mode: str):
    smoke = bool(getattr(args, "smoke", False))
    cli = [
        "--wisig_pkl", str(args.wisig_pkl),
        "--wisig_equalized", "1",
        "--wisig_train_days", str(args.train_days),
        "--wisig_test_days", "",
        "--wisig_train_rxs", str(args.source_rxs),
        "--wisig_test_rxs", "",
        "--wisig_allow_shared_days_if_receivers_disjoint", "false",
        "--phase1_source_only_eval", "true",
        "--split_mode", "tx_rx_day_1_7_2",
        "--labeled_ratio", "0.07",
        "--unlabeled_ratio", "0.63",
        "--source_val_ratio", "0.30",
        "--source_cal_ratio", "0.15",
        "--source_select_ratio", "0.15",
        "--phase1_source_role_protocol", "l_s_u_s_v_cal_v_select",
        "--output_dir", str(args.row_root),
        "--batch_size", "64" if smoke else "96",
        "--eval_batch_size", "512",
        "--num_workers", "0" if smoke else "4",
        "--prefetch_factor", "2",
        "--wisig_max_day123_per_combo", "20" if smoke else "0",
        "--model_variant", "lite_d",
        "--representation_mode", representation_mode,
        "--branch_ablation", "no_dac",
        "--domain_branch_ablation", "no_stats",
        "--domain_enhancer", "rcn_stats",
        "--use_mixstyle", "false",
        "--no_use_sat_consistency",
        "--lambda_domain", "0",
        "--lambda_adv", "0",
        "--lambda_group_ce", "0",
        "--lambda_fishr", "0",
        "--seed", str(args.seed),
        "--device", "cuda:0",
    ]
    return ssdg.build_arg_parser().parse_args(cli)


def _batch_mapping(raw: Any, *, receiver_map: Mapping[int, int], day_map: Mapping[int, int]):
    import torch

    if not isinstance(raw, (tuple, list)) or len(raw) < 4:
        raise TypeError("WiSig source batches must contain IQ, TX, domain, and metadata")
    x, y, domain, metadata = raw[:4]
    if not isinstance(metadata, Mapping):
        raise TypeError("WiSig source metadata must be a mapping")
    receiver_raw = torch.as_tensor(metadata["rx_i"]).reshape(-1).long()
    day_raw = torch.as_tensor(metadata["day_i"]).reshape(-1).long()
    receiver = torch.tensor(
        [receiver_map[int(value)] for value in receiver_raw.tolist()], dtype=torch.long
    )
    day = torch.tensor([day_map[int(value)] for value in day_raw.tolist()], dtype=torch.long)
    count = int(torch.as_tensor(y).reshape(-1).numel())
    return {
        "iq": x,
        "tx": torch.as_tensor(y).reshape(-1).long(),
        "receiver": receiver,
        "day": day,
        "channel": torch.zeros(count, dtype=torch.long),
        "domain": torch.as_tensor(domain).reshape(-1).long(),
        "q_phys": torch.zeros(count, 5, dtype=torch.float32),
    }


class _MappedLoader:
    def __init__(self, loader: Iterable[Any], receiver_map: Mapping[int, int], day_map: Mapping[int, int]):
        self.loader = loader
        self.receiver_map = dict(receiver_map)
        self.day_map = dict(day_map)
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self):
        for raw in self.loader:
            yield _batch_mapping(raw, receiver_map=self.receiver_map, day_map=self.day_map)

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]


class _RectangularLoader:
    def __init__(self, dataset: Any, *, seed: int, receiver_map: Mapping[int, int], day_map: Mapping[int, int]):
        import numpy as np
        from torch.utils.data import default_collate

        from cvsrffi.phase1_hcfdg.sampler import HCFDGEpisodeBatchSampler

        self.dataset = dataset
        self.default_collate = default_collate
        self.receiver_map = dict(receiver_map)
        self.day_map = dict(day_map)
        index = list(getattr(dataset, "index", []))
        if not index:
            raise ValueError("rectangular HCF-DG training requires an indexed WiSig source dataset")
        metadata = {
            "tx_ids": np.asarray([int(item.tx_i) for item in index], dtype=np.int64),
            "receiver_ids": np.asarray(
                [self.receiver_map[int(item.rx_i)] for item in index], dtype=np.int64
            ),
            "day_ids": np.asarray([self.day_map[int(item.day_i)] for item in index], dtype=np.int64),
            "channel_ids": (
                np.random.default_rng(int(seed)).random(len(index)) < 0.30
            ).astype(np.int64),
            "q_phys": np.zeros((len(index), 5), dtype=np.float32),
        }
        self.sampler = HCFDGEpisodeBatchSampler(
            metadata,
            seed=int(seed),
            episodes_per_epoch=256,
        )
        self._epoch = 0

    def __iter__(self):
        import torch

        self.sampler.set_epoch(self._epoch)
        self._epoch += 1
        for episode in self.sampler:
            raw = self.default_collate([self.dataset[index] for index in episode.indices])
            batch = _batch_mapping(
                raw,
                receiver_map=self.receiver_map,
                day_map=self.day_map,
            )
            batch.update(
                {
                    "receiver": torch.as_tensor(episode.receiver_ids).long(),
                    "day": torch.as_tensor(episode.day_ids).long(),
                    "channel": torch.as_tensor(episode.channel_ids).long(),
                    "domain": torch.as_tensor(episode.domain_ids).long(),
                    "query_domain": int(episode.query_domain),
                    "support_mask": _bool_tensor(episode.support_mask),
                    "query_mask": _bool_tensor(episode.query_mask),
                    "valid_tx_mask": _bool_tensor(episode.valid_tx_mask),
                    "episode_type": episode.episode_type,
                }
            )
            batch["content_keys"] = _content_keys_from_iq(batch["iq"])
            yield batch

    def __len__(self) -> int:
        return 256


def _control_loss(output: Any, *, labels: Any, **_kwargs: Any):
    import torch
    import torch.nn.functional as functional

    logits = output
    if isinstance(output, Mapping):
        logits = output.get("tx_logits", output.get("logits"))
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("control candidate must return rank-2 TX logits")
    target = torch.as_tensor(labels, device=logits.device).reshape(-1).long()
    return functional.cross_entropy(logits, target)


def _serializable_model_args(model_args: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(model_args).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            output[str(key)] = value
        elif isinstance(value, (tuple, list)) and all(
            item is None or isinstance(item, (str, int, float, bool)) for item in value
        ):
            output[str(key)] = list(value)
    return output


def _build_runtime_model(
    ssdg: Any,
    model_args: Any,
    *,
    candidate_id: str,
    device: Any,
    num_receivers: int,
    num_days: int,
):
    from cvsrffi.phase1_hcfdg.model import HCFDGModel

    baseline = ssdg.build_baseline_model(model_args, device)
    if candidate_id in {"A0", "A1"}:
        return baseline, "control"
    identity_backbone = getattr(baseline, "id_backbone", None)
    if identity_backbone is None:
        raise ValueError("lite_d baseline exposes no identity backbone")
    model = HCFDGModel(
        identity_backbone,
        num_classes=int(model_args.num_classes),
        num_receivers=int(num_receivers),
        num_days=int(num_days),
        num_channels=2,
        q_phys_dim=5,
    ).to(device)
    return model, "hcfdg"


def _strict_reconstruct(checkpoint_path: Path, *, ssdg: Any, device: Any):
    import torch
    from types import SimpleNamespace

    payload = torch.load(checkpoint_path, map_location=device)
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("checkpoint is missing HCF-DG runtime reconstruction metadata")
    model_args = SimpleNamespace(**dict(runtime["model_args"]))
    model, family = _build_runtime_model(
        ssdg,
        model_args,
        candidate_id=str(runtime["candidate_id"]),
        device=device,
        num_receivers=int(runtime["num_receivers"]),
        num_days=int(runtime["num_days"]),
    )
    try:
        incompatible = model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"strict checkpoint reconstruction failed: {exc}") from exc
    audit = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "shape_mismatches": [],
    }
    model.eval()
    return model, family, audit


def _evaluate_loader(
    model: Any,
    family: str,
    loader: Iterable[Any],
    *,
    scenario: str,
    ssdg: Any,
    ssdg_args: Any,
    device: Any,
    seed: int,
) -> dict[str, Any]:
    import torch

    correct = total = 0
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        for raw in loader:
            x, y = raw[0].to(device), raw[1].to(device).reshape(-1).long()
            if scenario != "clean":
                x, _ = ssdg.apply_sat_channel_for_scenario(
                    x,
                    scenario,
                    ssdg_args,
                    gen=generator,
                    return_meta=False,
                )
            logits = model.inference_logits(x) if family == "hcfdg" else model(x)
            prediction = logits.argmax(dim=1)
            matches = prediction.eq(y)
            correct += int(matches.sum().item())
            total += int(y.numel())
            for class_id in y.unique().tolist():
                mask = y == int(class_id)
                class_total[int(class_id)] = class_total.get(int(class_id), 0) + int(mask.sum().item())
                class_correct[int(class_id)] = class_correct.get(int(class_id), 0) + int(matches[mask].sum().item())
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
        "class_floor": min(per_class.values()) if per_class else None,
    }


def _build_heldout_loader(ssdg: Any, ssdg_args: Any, *, heldout_receiver: int, train_days: tuple[int, ...], device: Any):
    dataset_payload = ssdg.load_wisig_compact_pkl(ssdg_args.wisig_pkl)
    day_list = list(dataset_payload.get("capture_date_list", []))
    receiver_list = list(dataset_payload.get("rx_list", []))
    days = ssdg._resolve_days(day_list, list(train_days), list(train_days))
    receivers = ssdg._resolve_rxs(receiver_list, [heldout_receiver], [heldout_receiver])
    dataset = ssdg.WiSigCompactDataset(
        dataset_payload,
        out_len=int(ssdg_args.wisig_out_len),
        crop_mode="center",
        normalize=True,
        equalized=1,
        day_keep=days,
        rx_keep=receivers,
        domain="rx_day",
        seed=int(ssdg_args.seed),
        build_index=True,
    )
    return ssdg.make_loader(
        dataset,
        int(ssdg_args.eval_batch_size),
        False,
        int(ssdg_args.num_workers),
        device,
        False,
        int(ssdg_args.prefetch_factor),
    )


def _worker_row(args: argparse.Namespace) -> int:
    import torch

    from cvsrffi.phase1_hcfdg.satellite import build_single_view_batch
    from cvsrffi.phase1_hcfdg.trainer import HCFDGTrainer

    candidate_id = str(args.candidate_id).strip().upper()
    config = _resolve_worker_config(candidate_id, smoke=bool(args.smoke))
    if bool(args.smoke):
        if int(args.optimizer_updates) not in (0, int(config.optimizer_updates)):
            raise ValueError(
                "HCF-DG smoke requires the dedicated stage0-stage3 budget"
            )
    elif int(args.optimizer_updates) != int(config.optimizer_updates):
        raise ValueError("optimizer update count does not match the frozen candidate")
    source_receivers = _parse_int_csv(args.source_rxs, name="source_rxs")
    train_days = _parse_int_csv(args.train_days, name="train_days")
    if tuple(train_days) != TRAIN_DAYS:
        raise ValueError("formal HCF-DG training days must be 1,2,3")
    if int(args.heldout_rx) not in SOURCE_RECEIVERS:
        raise ValueError("heldout receiver is not a registered source receiver")
    if set(source_receivers) != set(SOURCE_RECEIVERS) - {int(args.heldout_rx)}:
        raise ValueError("source receivers must be the four registered receivers outside the LORO fold")
    row_root = Path(args.row_root).resolve()
    if row_root.exists():
        raise FileExistsError(f"row output already exists: {row_root}")
    row_root.mkdir(parents=True, exist_ok=False)

    code_root = Path(__file__).resolve().parents[2]
    ssdg = _load_ssdg_module(code_root)
    representation_mode = "single_parameter_matched" if candidate_id == "A1" else "dual"
    ssdg_args = _source_runtime_args(ssdg, args, representation_mode)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ssdg.set_seed(int(args.seed))
    data_context = ssdg._build_ssdg_wisig_data(ssdg_args, device)
    if data_context["split_info"].get("target_access") is not False:
        raise ValueError("source data builder reported target access")

    model_args = ssdg.merge_checkpoint_args(
        {"model": None, "args": {}, "stats": {}, "split_info": None},
        ssdg_args,
        input_len=int(data_context["input_len"]),
        num_domains=int(data_context["num_domains"]),
    )
    model_args = ssdg._apply_model_cli_args(model_args, ssdg_args)
    receiver_map = {receiver: position for position, receiver in enumerate(source_receivers)}
    day_map = {day: position for position, day in enumerate(train_days)}
    model, family = _build_runtime_model(
        ssdg,
        model_args,
        candidate_id=candidate_id,
        device=device,
        num_receivers=len(receiver_map),
        num_days=len(day_map),
    )
    base_train_loader = data_context["train_loader"]
    if bool(config.use_rectangular_batch):
        labeled_loader: Iterable[Any] = _RectangularLoader(
            base_train_loader.dataset,
            seed=int(args.seed),
            receiver_map=receiver_map,
            day_map=day_map,
        )
    else:
        labeled_loader = _MappedLoader(base_train_loader, receiver_map, day_map)
    unlabeled_loader = _MappedLoader(data_context["unlabeled_loader"], receiver_map, day_map)

    def satellite_augmentor(x: Any, *, scenario: str, generator: Any):
        channel_seed = int(
            torch.randint(
                0,
                torch.iinfo(torch.int64).max,
                (1,),
                generator=generator,
                device="cpu",
            ).item()
        )
        channel_generator = torch.Generator(device=x.device).manual_seed(channel_seed)
        return ssdg.apply_sat_channel_for_scenario(
            x,
            scenario,
            ssdg_args,
            gen=channel_generator,
            return_meta=True,
        )

    trainer = HCFDGTrainer(
        model=model,
        config=config,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        validation_loader=None,
        build_single_view_batch=build_single_view_batch,
        satellite_augmentor=satellite_augmentor,
        loss_fn=_control_loss if family == "control" else None,
        device=device,
        output_dir=row_root,
        checkpoint_path=row_root / "final_hcfdg.pt",
        source_split={
            "role": "source_loro",
            "source_receivers": list(source_receivers),
            "heldout_receiver": int(args.heldout_rx),
            "train_days": list(train_days),
            "external_domain_access": False,
        },
        fold=int(args.heldout_rx),
        seed=int(args.seed),
        stage_budget=HCFDG_SMOKE_STAGE_BUDGET if bool(args.smoke) else None,
    )
    state = trainer.train()
    if state.optimizer_updates != int(config.optimizer_updates):
        raise RuntimeError("training ended without the frozen optimizer update count")
    if bool(args.smoke) and int(state.stage_updates.get("stage3", 0)) < 1:
        raise RuntimeError("HCF-DG smoke did not reach stage3")

    checkpoint_path = row_root / "final_hcfdg.pt"
    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["runtime"] = {
        "candidate_id": candidate_id,
        "family": family,
        "model_args": _serializable_model_args(model_args),
        "num_receivers": len(receiver_map),
        "num_days": len(day_map),
        "source_receivers": list(source_receivers),
        "heldout_receiver": int(args.heldout_rx),
        "train_days": list(train_days),
        "target_access": False,
        "smoke": bool(args.smoke),
        "stage_updates": dict(state.stage_updates),
    }
    payload["inference"] = (
        {
            "head": "tx_logits",
            "common_head_only": False,
            "environment_inputs_required": False,
        }
        if family == "control"
        else {
            "head": "common_identity_head",
            "common_head_only": True,
            "environment_inputs_required": False,
        }
    )
    torch.save(payload, checkpoint_path)

    heldout_loader = _build_heldout_loader(
        ssdg,
        ssdg_args,
        heldout_receiver=int(args.heldout_rx),
        train_days=train_days,
        device=device,
    )
    row = PlanRow(
        candidate_id=candidate_id,
        heldout_receiver=int(args.heldout_rx),
        seed=int(args.seed),
        optimizer_updates=int(config.optimizer_updates),
        gpu=int(args.gpu),
        source_receivers=source_receivers,
        train_days=train_days,
    )

    def reconstruct(path: Path):
        rebuilt, rebuilt_family, audit = _strict_reconstruct(path, ssdg=ssdg, device=device)
        setattr(rebuilt, "_hcfdg_runtime_family", rebuilt_family)
        return rebuilt, audit

    def evaluate(rebuilt: Any, scenario: str, _row: PlanRow):
        rebuilt_family = str(getattr(rebuilt, "_hcfdg_runtime_family"))
        return _evaluate_loader(
            rebuilt,
            rebuilt_family,
            heldout_loader,
            scenario=scenario,
            ssdg=ssdg,
            ssdg_args=ssdg_args,
            device=device,
            seed=int(args.seed),
        )

    evaluate_final_checkpoint(
        row,
        row_root,
        reconstruct_fn=reconstruct,
        evaluate_fn=evaluate,
    )
    closure = validate_artifact_closure(row_root)
    _write_new_json(row_root / "ARTIFACTS_COMPLETE.json", closure)
    return 0


def _dispatch_formal(args: argparse.Namespace) -> int:
    if str(args.stage).strip().lower() == "deep":
        if not str(args.run_id or "").strip():
            raise ValueError("formal deep requires an explicit --run-id")
        if int(args.optimizer_updates) not in (0, V2_OPTIMIZER_UPDATES):
            raise ValueError("formal deep optimizer budget is fixed at 6300 updates")
    run_root = validate_output_root(args.run_root)
    wisig_pkl = Path(args.wisig_pkl).resolve()
    if not wisig_pkl.is_file():
        raise FileNotFoundError(f"WiSig source package does not exist: {wisig_pkl}")
    code_root = Path(args.code_root).resolve() if args.code_root else Path(__file__).resolve().parents[2]
    python = Path(args.python).resolve() if args.python else Path(sys.executable).resolve()
    gpus = _parse_int_csv(args.gpus, name="gpus")
    folds = _parse_int_csv(args.folds, name="folds")
    seeds = _parse_int_csv(args.seeds, name="seeds") if str(args.seeds).strip() else None
    rows = build_plan(
        stage=args.stage,
        folds=folds,
        gpus=gpus,
        seeds=seeds,
        v2_passed=bool(args.v2_passed),
        v2_parent_candidate_id=str(args.v2_parent_candidate_id),
    )
    if str(args.stage).strip().lower() == "deep":
        if len(rows) <= 0 or any(row.optimizer_updates != V2_OPTIMIZER_UPDATES for row in rows):
            raise ValueError("formal deep plan must use 6300 updates for every row")
        if seeds is not None and any(row.seed not in seeds for row in rows):
            raise ValueError("formal deep plan contains a seed outside --seeds")
    run_root.mkdir(parents=True, exist_ok=False)
    roots = LauncherRoots(
        code_root=code_root,
        python=python,
        run_root=run_root,
        wisig_pkl=wisig_pkl,
    )
    write_plan_json(
        run_root,
        rows,
        run_id=str(args.run_id or RUN_ID_DEFAULT),
    )
    failure_lock = threading.Lock()
    failure_fingerprints: Counter[str] = Counter()
    systemic_stop = threading.Event()

    def execute(row: PlanRow) -> str:
        if systemic_stop.is_set():
            failure_root = _row_root(row, roots)
            failure_root.mkdir(parents=True, exist_ok=True)
            _write_new_json(
                failure_root / "TECHNICAL_FAILURE.json",
                {
                    "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
                    "exception_type": "NotDispatchedAfterSystemicFailure",
                    "message": "dispatcher stopped new rows after two matching technical failures",
                },
            )
            return "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
        try:
            return run_row(row, roots)
        except Exception as exc:  # preserve the row and keep the dispatcher state explicit
            failure_root = _row_root(row, roots)
            failure_root.mkdir(parents=True, exist_ok=True)
            failure_path = failure_root / "TECHNICAL_FAILURE.json"
            if not failure_path.exists():
                _write_new_json(
                    failure_path,
                    {
                        "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            fingerprint = f"{type(exc).__name__}:{str(exc)}"
            with failure_lock:
                failure_fingerprints[fingerprint] += 1
                if failure_fingerprints[fingerprint] >= 2:
                    systemic_stop.set()
            return "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"

    statuses = run_plan(rows, execute, max_active_per_gpu=2)
    write_final_status(run_root, rows, statuses)
    return 0 if set(statuses.values()) == {"ARTIFACTS_COMPLETE"} else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-row", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage", choices=("quick", "deep", "residual"), default="quick")
    parser.add_argument("--folds", default="1,8")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--code-root", default="")
    parser.add_argument("--python", default="")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--v2-passed", action="store_true")
    parser.add_argument("--v2-parent-candidate-id", default="A9")
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--heldout-rx", type=int, default=-1)
    parser.add_argument("--source-rxs", default="")
    parser.add_argument("--train-days", default="")
    parser.add_argument("--optimizer-updates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wisig-pkl", default="")
    parser.add_argument("--row-root", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.formal and args.worker_row:
        raise ValueError("--formal and --worker-row are mutually exclusive")
    if args.formal and args.smoke:
        raise ValueError("--smoke is a worker-only entry point")
    if args.worker_row and args.seeds:
        raise ValueError("--seeds is a formal matrix selector; worker rows use --seed")
    if args.smoke and not args.worker_row:
        raise ValueError("--smoke requires --worker-row")
    if args.worker_row:
        return _worker_row(args)
    if args.formal:
        if not args.run_root:
            raise ValueError("--formal requires --run-root")
        return _dispatch_formal(args)
    raise RuntimeError("select --formal or --worker-row")


if __name__ == "__main__":
    raise SystemExit(main())
