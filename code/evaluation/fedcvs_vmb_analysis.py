from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F


def _tensor(value: Any) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    tensor = tensor.detach().float().cpu()
    if tensor.ndim > 2:
        tensor = tensor.flatten(1)
    return tensor


def _cosine_stats(values: Iterable[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "count": int(t.numel()),
        "mean": float(t.mean().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def prototype_drift(previous: Any, current: Any) -> Dict[str, float]:
    """Report cosine drift between two prototype snapshots."""

    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        keys = sorted(set(previous).intersection(current))
        drifts = []
        for key in keys:
            prev = _tensor(previous[key]).view(-1)
            cur = _tensor(current[key]).view(-1)
            if prev.numel() == cur.numel() and prev.numel() > 0:
                cos = F.cosine_similarity(prev.unsqueeze(0), cur.unsqueeze(0), dim=1).item()
                drifts.append(1.0 - float(cos))
        stats = _cosine_stats(drifts)
        stats["matched_prototypes"] = int(len(keys))
        return stats

    prev = _tensor(previous)
    cur = _tensor(current)
    if prev.shape != cur.shape:
        raise ValueError(f"prototype shapes differ: previous={tuple(prev.shape)} current={tuple(cur.shape)}")
    if prev.ndim == 1:
        prev = prev.unsqueeze(0)
        cur = cur.unsqueeze(0)
    cos = F.cosine_similarity(prev, cur, dim=1)
    stats = _cosine_stats((1.0 - cos).tolist())
    stats["matched_prototypes"] = int(prev.size(0))
    return stats


def prototype_separation(prototypes: Any) -> Dict[str, float]:
    proto = _tensor(prototypes)
    if proto.ndim == 1:
        proto = proto.unsqueeze(0)
    if proto.size(0) < 2:
        return {"count": 0, "mean_cosine": float("nan"), "max_cosine": float("nan")}
    norm = F.normalize(proto, dim=1)
    sim = norm @ norm.t()
    mask = ~torch.eye(sim.size(0), dtype=torch.bool)
    vals = sim[mask]
    return {
        "count": int(vals.numel()),
        "mean_cosine": float(vals.mean().item()),
        "max_cosine": float(vals.max().item()),
    }


def _flatten_gradient(gradient: Mapping[str, Any]) -> torch.Tensor:
    pieces = []
    for key in sorted(gradient):
        value = gradient[key]
        if torch.is_tensor(value):
            pieces.append(value.detach().float().cpu().reshape(-1))
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


def gradient_cosine_report(client_gradients: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    vectors = {cid: _flatten_gradient(grad) for cid, grad in client_gradients.items()}
    client_ids = [cid for cid, vec in vectors.items() if vec.numel() > 0]
    cosines = []
    for i, cid_a in enumerate(client_ids):
        for cid_b in client_ids[i + 1 :]:
            a = vectors[cid_a]
            b = vectors[cid_b]
            if a.numel() == b.numel():
                cosines.append(float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()))
    stats = _cosine_stats(cosines)
    stats["clients"] = int(len(client_ids))
    return stats


def summarize_vmb_logs(logs_jsonl: str | Path) -> Dict[str, Any]:
    path = Path(logs_jsonl)
    grad_norms = []
    grad_cos_means = []
    tx_proto_counts = []
    rx_proto_counts = []
    stages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        update = row.get("vmb_server_update") or {}
        if "grad_norm" in update:
            grad_norms.append(float(update["grad_norm"]))
        cos = row.get("vmb_gradient_cosine") or {}
        if "mean" in cos:
            grad_cos_means.append(float(cos["mean"]))
        proto = row.get("global_vmb_proto_summary") or {}
        if "tx_count_nonzero" in proto:
            tx_proto_counts.append(float(proto["tx_count_nonzero"]))
        if "rx_count_nonzero" in proto:
            rx_proto_counts.append(float(proto["rx_count_nonzero"]))
        if row.get("vmb_stage"):
            stages.append(str(row["vmb_stage"]))
    return {
        "rounds_with_vmb_update": len(grad_norms),
        "grad_norm": _cosine_stats(grad_norms),
        "grad_cosine_mean": _cosine_stats(grad_cos_means),
        "tx_proto_classes": _cosine_stats(tx_proto_counts),
        "rx_proto_clients": _cosine_stats(rx_proto_counts),
        "stages": sorted(set(stages)),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize FedCVS-RFFI-VMB training diagnostics.")
    parser.add_argument("--logs_jsonl", required=True, help="Federated logs.jsonl produced by train.py.")
    parser.add_argument("--output", required=True, help="Path to write diagnostics JSON.")
    args = parser.parse_args(argv)
    summary = summarize_vmb_logs(args.logs_jsonl)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
