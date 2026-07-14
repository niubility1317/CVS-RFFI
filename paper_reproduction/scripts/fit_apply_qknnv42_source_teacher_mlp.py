"""Fit a tiny source-only qKNN residual MLP and apply it to frozen target caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paper_reproduction.scripts.fit_apply_qknnv42_source_teacher_adapter import (
    POLICIES,
    POLICY_VIEW_COUNTS,
    _aligned_source_rows,
    _cache_path,
    _group_holdout_mask,
    _manifest,
    _mark_post_adapter_manifest,
    _sha256,
    _validate_frozen_source,
    _validate_teacher,
)


class ResidualQKNNAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        base = F.normalize(rows.float(), dim=1)
        delta = self.up(F.gelu(self.down(self.norm(base))))
        return F.normalize(base + self.alpha * delta, dim=1)


def _device(value: str) -> torch.device:
    requested = str(value)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {requested}")
    return torch.device(requested)


def _train(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    *,
    rank: int,
    alpha: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[ResidualQKNNAdapter, dict[str, float | None]]:
    torch.manual_seed(int(seed))
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = F.normalize(torch.as_tensor(y, dtype=torch.float32, device=device), dim=1)
    train_idx = torch.as_tensor(np.flatnonzero(train_mask), dtype=torch.long, device=device)
    holdout_idx = torch.as_tensor(np.flatnonzero(~train_mask), dtype=torch.long, device=device)
    model = ResidualQKNNAdapter(int(x.shape[1]), int(rank), float(alpha)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device=device).manual_seed(int(seed) + 17)
    batch_size = min(256, int(train_idx.numel()))
    for _epoch in range(int(epochs)):
        model.train()
        order = train_idx[torch.randperm(int(train_idx.numel()), generator=generator, device=device)]
        for start in range(0, int(order.numel()), batch_size):
            index = order[start : start + batch_size]
            predicted = model(x_t.index_select(0, index))
            target = y_t.index_select(0, index)
            cosine = 1.0 - F.cosine_similarity(predicted, target, dim=1).mean()
            mse = F.mse_loss(predicted, target)
            residual = F.mse_loss(predicted, F.normalize(x_t.index_select(0, index), dim=1))
            loss = cosine + 0.2 * mse + 0.01 * residual
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        if int(holdout_idx.numel()) > 0:
            predicted = model(x_t.index_select(0, holdout_idx))
            target = y_t.index_select(0, holdout_idx)
            cosine = float(F.cosine_similarity(predicted, target, dim=1).mean().item())
            mse = float(F.mse_loss(predicted, target).item())
        else:
            cosine = None
            mse = None
    return model, {"holdout_cosine": cosine, "holdout_mse": mse}


@torch.no_grad()
def _apply(model: nn.Module, rows: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(rows, dtype=torch.float32, device=device)
    output = []
    for start in range(0, len(tensor), 2048):
        output.append(model(tensor[start : start + 2048]).cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32, copy=False)


def _save_state(path: Path, model: ResidualQKNNAdapter) -> None:
    # Torch 2.1 + NumPy 2.x can expose zero-copy arrays whose array-function
    # dispatch is rejected by np.savez.  Re-materialize through Python values
    # so the persisted payload is owned by the active NumPy runtime.
    state = {
        key: np.asarray(value.detach().cpu().tolist(), dtype=np.float32)
        for key, value in model.state_dict().items()
    }
    np.savez(
        path,
        **state,
        alpha=np.asarray(model.alpha, dtype=np.float32),
    )


def fit_apply(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = _device(str(args.device))
    teacher = np.load(args.teacher_cache, allow_pickle=False)
    teacher_manifest = _manifest(teacher)
    teacher_evidence = _validate_teacher(
        args.teacher_cache,
        teacher_manifest,
        expected_sha256=str(args.expected_teacher_sha256),
    )
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("policies must not contain duplicates")
    if len(set(args.receivers)) != len(args.receivers):
        raise ValueError("receivers must not contain duplicates")
    if not args.rank_grid or any(int(value) <= 0 for value in args.rank_grid):
        raise ValueError("rank_grid values must be positive")
    if len(set(int(value) for value in args.rank_grid)) != len(args.rank_grid):
        raise ValueError("rank_grid must not contain duplicates")
    if not args.alpha_grid or any(
        not math.isfinite(float(value)) or not 0.0 < float(value) <= 2.0
        for value in args.alpha_grid
    ):
        raise ValueError("alpha_grid values must be finite and in (0,2]")
    if len(set(float(value) for value in args.alpha_grid)) != len(args.alpha_grid):
        raise ValueError("alpha_grid must not contain duplicates")
    if int(args.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if not math.isfinite(float(args.lr)) or float(args.lr) <= 0.0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(float(args.weight_decay)) or float(args.weight_decay) < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    summaries: dict[str, Any] = {}
    for policy in args.policies:
        source_path = _cache_path(
            args.frozen_source_root,
            str(args.source_receiver),
            str(args.frozen_subdir_base),
            str(policy),
            str(args.feature_name),
        )
        frozen_source = np.load(source_path, allow_pickle=False)
        _validate_frozen_source(
            _manifest(frozen_source),
            expected_checkpoint_sha256=str(args.expected_checkpoint_sha256),
            expected_policy=str(policy),
            expected_tta_view_count=POLICY_VIEW_COUNTS[str(policy)],
        )
        x, y, physical_keys = _aligned_source_rows(frozen_source, teacher)
        if any(int(value) > int(x.shape[1]) for value in args.rank_grid):
            raise ValueError(f"rank_grid values must not exceed feature dim {x.shape[1]}")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("source/teacher features must be finite")
        holdout = _group_holdout_mask(physical_keys)
        selection: list[dict[str, float | int]] = []
        best: tuple[float, int, float, ResidualQKNNAdapter] | None = None
        for rank in args.rank_grid:
            for alpha in args.alpha_grid:
                model, metrics = _train(
                    x,
                    y,
                    ~holdout,
                    rank=int(rank),
                    alpha=float(alpha),
                    epochs=int(args.epochs),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    seed=int(args.seed),
                    device=device,
                )
                row = {"rank": int(rank), "alpha": float(alpha), **metrics}
                selection.append(row)
                key = (float(metrics["holdout_cosine"]), -int(rank), -float(alpha))
                if best is None or key > (best[0], -best[1], -best[2]):
                    best = (key[0], int(rank), float(alpha), model)
        if best is None:
            raise RuntimeError("MLP selection produced no candidate")
        _score, rank, alpha, _selected_model = best
        final_model, full_fit_metrics = _train(
            x,
            y,
            np.ones(len(x), dtype=bool),
            rank=rank,
            alpha=alpha,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed) + 1,
            device=device,
        )
        adapter_dir = args.out_root / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = adapter_dir / f"source_teacher_residual_mlp_{policy}.npz"
        _save_state(adapter_path, final_model)
        parameter_count = int(sum(parameter.numel() for parameter in final_model.parameters()))
        estimated_macs = int(2 * x.shape[1] * rank)
        outputs: list[str] = []
        for receiver in args.receivers:
            input_path = _cache_path(
                args.frozen_target_root,
                str(receiver),
                str(args.frozen_subdir_base),
                str(policy),
                str(args.feature_name),
            )
            with np.load(input_path, allow_pickle=False) as source:
                payload = {key: np.asarray(source[key]) for key in source.files}
                manifest = _manifest(source)
            _validate_frozen_source(
                manifest,
                expected_checkpoint_sha256=str(args.expected_checkpoint_sha256),
                expected_policy=str(policy),
                expected_tta_view_count=POLICY_VIEW_COUNTS[str(policy)],
            )
            if set(np.unique(payload["dataset_role"].astype(str)).tolist()) != {
                "target_old",
                "target_new",
            }:
                raise ValueError(
                    f"target cache contains roles unused by Stage2-C qKNN: {input_path}"
                )
            if not np.all(np.isfinite(payload["features"])):
                raise ValueError(f"target features must be finite: {input_path}")
            payload["features"] = _apply(final_model, payload["features"], device)
            if not np.all(np.isfinite(payload["features"])):
                raise ValueError(f"adapted target features are non-finite: {input_path}")
            adapter_info = {
                "mode": "source_teacher_residual_mlp",
                "policy": str(policy),
                "rank": rank,
                "alpha": alpha,
                "epochs": int(args.epochs),
                "training_role": "source",
                "training_row_count": int(len(x)),
                "uses_target_rows_for_fit": False,
                "uses_target_labels_for_fit": False,
                "uses_target_query_for_fit": False,
                "updates_adv3b02": False,
                "gradient_updates_adv3b02": 0,
                "parameter_count": parameter_count,
                "estimated_macs_per_sample": estimated_macs,
                "parameter_bytes_fp32": int(parameter_count * 4 + 4),
                "adapter_path": str(adapter_path),
            }
            manifest = _mark_post_adapter_manifest(manifest, adapter_info)
            payload["manifest_json"] = np.asarray(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
            output_path = _cache_path(
                args.out_root,
                str(receiver),
                str(args.output_subdir_base),
                str(policy),
                str(args.feature_name),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **payload)
            outputs.append(str(output_path))
        summaries[str(policy)] = {
            "source_cache": str(source_path),
            "aligned_source_rows": int(len(x)),
            "selection": selection,
            "selected_rank": rank,
            "selected_alpha": alpha,
            "final_fit_holdout_metrics_are_not_applicable": full_fit_metrics,
            "adapter_path": str(adapter_path),
            "adapter_sha256": _sha256(adapter_path),
            "parameter_count": parameter_count,
            "estimated_macs_per_sample": estimated_macs,
            "outputs": outputs,
        }
    summary = {
        "schema": "qknnv42_source_teacher_residual_mlp_adapter_v1",
        "training_scope": "source_only",
        "device": str(device),
        "adv3b02_gradient_updates": 0,
        "uses_target_rows_for_fit": False,
        "teacher_manifest_payload_source": teacher_manifest.get("payload_source", ""),
        "teacher_evidence": teacher_evidence,
        "policies": summaries,
    }
    (args.out_root / "adapter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--frozen-target-root", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-receiver", default="20-1")
    parser.add_argument("--receivers", nargs="+", default=["20-1", "3-19", "7-14", "7-7", "8-8"])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["none"])
    parser.add_argument("--frozen-subdir-base", default="ADV3B02_FROZEN_QKNN_FFT96")
    parser.add_argument("--output-subdir-base", default="ADV3B02_FROZEN_QKNN_MLP_FFT96")
    parser.add_argument("--feature-name", default="features_frozen_adv3b02_fft96.npz")
    parser.add_argument("--rank-grid", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=713101)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    print(
        json.dumps(
            fit_apply(parse_args()), ensure_ascii=False, indent=2, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
