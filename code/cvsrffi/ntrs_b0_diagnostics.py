"""No-training paired-shift diagnostics for ADVB02 NTRS-V4 source calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def torch_from_numpy_compatible(value: np.ndarray) -> torch.Tensor:
    """Convert through the buffer protocol for Torch2.1/NumPy2 ABI compatibility."""

    array = np.ascontiguousarray(value)
    dtype_map = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.bool_): torch.bool,
    }
    if array.dtype not in dtype_map:
        raise TypeError(f"unsupported B0 NumPy dtype: {array.dtype}")
    return torch.frombuffer(memoryview(array), dtype=dtype_map[array.dtype]).reshape(array.shape).clone()


def _logits(embedding: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.normalize(embedding.float(), dim=1) @ F.normalize(weight.float(), dim=1).transpose(0, 1)


def _accuracy_and_flips(raw_logits: torch.Tensor, corrected_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    raw_ok = raw_logits.argmax(dim=1) == labels
    corrected_ok = corrected_logits.argmax(dim=1) == labels
    rescued = int((~raw_ok & corrected_ok).sum().item())
    harmed = int((raw_ok & ~corrected_ok).sum().item())
    return {
        "accuracy": float(corrected_ok.float().mean().item()),
        "rescued": rescued,
        "harmed": harmed,
        "net_rescue": rescued - harmed,
        "flip_precision": float(rescued / max(1, rescued + harmed)),
    }


def analyze_paired_shift(
    clean_embedding: torch.Tensor,
    satellite_embedding: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor,
    *,
    ranks: Iterable[int] = (4, 8, 16, 32),
    learned_correction: torch.Tensor | None = None,
    tx_ids: torch.Tensor | None = None,
    scenario_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    if clean_embedding.shape != satellite_embedding.shape or clean_embedding.dim() != 2:
        raise ValueError("paired embeddings must be aligned [N, D] tensors")
    labels = labels.view(-1).long()
    if int(labels.numel()) != int(clean_embedding.size(0)):
        raise ValueError("labels must align with paired embeddings")
    shift = (satellite_embedding - clean_embedding).float()
    centered = shift - shift.mean(dim=0, keepdim=True)
    _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    variance = singular.square()
    total = variance.sum().clamp_min(1e-12)
    raw_logits = _logits(satellite_embedding, classifier_weight)
    result: dict[str, object] = {
        "sample_count": int(shift.size(0)),
        "embedding_dim": int(shift.size(1)),
        "raw_accuracy": float((raw_logits.argmax(dim=1) == labels).float().mean().item()),
        "ranks": {},
        "pca_transport": {
            "mean_shift": shift.mean(dim=0).detach().cpu().tolist(),
            "basis_by_rank": {},
        },
    }
    for requested in ranks:
        rank = max(1, min(int(requested), int(vh.size(0))))
        basis = vh[:rank].transpose(0, 1)
        projected = centered @ basis @ basis.transpose(0, 1) + shift.mean(dim=0, keepdim=True)
        corrected = satellite_embedding - projected.to(dtype=satellite_embedding.dtype)
        metrics = _accuracy_and_flips(raw_logits, _logits(corrected, classifier_weight), labels)
        metrics.update(
            {
                "explained_variance": float(variance[:rank].sum().div(total).item()),
                "transport_rmse": float((shift - projected).square().mean().sqrt().item()),
            }
        )
        result["ranks"][str(int(requested))] = metrics
        result["pca_transport"]["basis_by_rank"][str(int(requested))] = (
            basis.detach().cpu().tolist()
        )
    full_corrected = satellite_embedding - shift.to(dtype=satellite_embedding.dtype)
    result["full_shift_oracle"] = _accuracy_and_flips(
        raw_logits, _logits(full_corrected, classifier_weight), labels
    )

    if learned_correction is not None:
        if learned_correction.shape != shift.shape:
            raise ValueError("learned correction must align with paired embeddings")
        grid = torch.linspace(0.0, 1.0, 21, device=shift.device)
        global_rows = []
        ce_rows = []
        for gate in grid:
            logits = _logits(satellite_embedding - gate * learned_correction, classifier_weight)
            global_rows.append(
                {"gate": float(gate.item()), **_accuracy_and_flips(raw_logits, logits, labels)}
            )
            ce_rows.append(F.cross_entropy(logits, labels, reduction="none"))
        ce_matrix = torch.stack(ce_rows, dim=1)
        best_index = ce_matrix.argmin(dim=1)
        selected_logits = torch.stack(
            [
                _logits(satellite_embedding - gate * learned_correction, classifier_weight)
                for gate in grid
            ],
            dim=1,
        )[torch.arange(labels.numel(), device=labels.device), best_index]
        result["continuous_gate_oracle"] = {
            "global": global_rows,
            "sample_oracle": _accuracy_and_flips(raw_logits, selected_logits, labels),
        }

    total_variance = float(centered.square().sum().item())
    variance_result: dict[str, float] = {"total": total_variance}
    if scenario_ids is not None:
        scenario_ids = scenario_ids.view(-1).to(device=shift.device)
        scenario_mean = torch.zeros_like(shift)
        for value in torch.unique(scenario_ids):
            mask = scenario_ids == value
            scenario_mean[mask] = shift[mask].mean(dim=0)
        variance_result["scenario_common_ratio"] = float(
            (scenario_mean - shift.mean(dim=0, keepdim=True)).square().sum().div(max(total_variance, 1e-12)).item()
        )
    if tx_ids is not None:
        tx_ids = tx_ids.view(-1).to(device=shift.device)
        tx_mean = torch.zeros_like(shift)
        for value in torch.unique(tx_ids):
            mask = tx_ids == value
            tx_mean[mask] = shift[mask].mean(dim=0)
        variance_result["tx_main_ratio"] = float(
            (tx_mean - shift.mean(dim=0, keepdim=True)).square().sum().div(max(total_variance, 1e-12)).item()
        )
        if scenario_ids is not None:
            grand_mean = shift.mean(dim=0, keepdim=True)
            interaction = torch.zeros_like(shift)
            for tx_value in torch.unique(tx_ids):
                tx_mask = tx_ids == tx_value
                tx_group_mean = shift[tx_mask].mean(dim=0, keepdim=True)
                for scenario_value in torch.unique(scenario_ids):
                    scenario_mask = scenario_ids == scenario_value
                    cell_mask = tx_mask & scenario_mask
                    if not bool(cell_mask.any()):
                        continue
                    scenario_group_mean = shift[scenario_mask].mean(dim=0, keepdim=True)
                    cell_mean = shift[cell_mask].mean(dim=0, keepdim=True)
                    interaction[cell_mask] = (
                        cell_mean - tx_group_mean - scenario_group_mean + grand_mean
                    )
            variance_result["tx_scenario_interaction_ratio"] = float(
                interaction.square().sum().div(max(total_variance, 1e-12)).item()
            )
    result["variance_decomposition"] = variance_result
    return result


def _tensor(payload: Mapping[str, np.ndarray], key: str, required: bool = True) -> torch.Tensor | None:
    if key not in payload:
        if required:
            raise KeyError(f"missing required array: {key}")
        return None
    return torch_from_numpy_compatible(np.asarray(payload[key]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    with np.load(args.input_npz, allow_pickle=False) as payload:
        result = analyze_paired_shift(
            _tensor(payload, "clean_embedding"),
            _tensor(payload, "satellite_embedding"),
            _tensor(payload, "labels"),
            _tensor(payload, "classifier_weight"),
            learned_correction=_tensor(payload, "learned_correction", required=False),
            tx_ids=_tensor(payload, "tx_ids", required=False),
            scenario_ids=_tensor(payload, "scenario_ids", required=False),
        )
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B0 output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
