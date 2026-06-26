from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_feature_tensor(value: Any) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    tensor = tensor.detach().float().cpu()
    if tensor.ndim > 2:
        tensor = tensor.flatten(1)
    if tensor.ndim != 2:
        raise ValueError(f"features must be a 2D tensor after flattening, got shape={tuple(tensor.shape)}")
    return tensor


def _as_label_tensor(value: Any) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    tensor = tensor.detach().long().cpu().view(-1)
    if tensor.numel() == 0:
        raise ValueError("labels are empty")
    return tensor


def _infer_num_classes(labels: torch.Tensor, num_classes: Optional[int]) -> int:
    if num_classes is not None:
        return max(1, int(num_classes))
    return int(labels.max().item()) + 1


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 4 or val_fraction <= 0.0:
        idx = torch.arange(n)
        return idx, idx
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator)
    val_n = max(1, min(n - 1, int(round(n * float(val_fraction)))))
    return perm[val_n:], perm[:val_n]


def train_linear_probe(
    features: Any,
    labels: Any,
    *,
    num_classes: Optional[int] = None,
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Train a deterministic linear probe and report held-out accuracy."""

    x = _as_feature_tensor(features)
    y = _as_label_tensor(labels)
    if x.size(0) != y.numel():
        raise ValueError(f"feature rows ({x.size(0)}) and labels ({y.numel()}) differ")

    n_classes = _infer_num_classes(y, num_classes)
    train_idx, val_idx = _split_indices(int(y.numel()), val_fraction, seed)
    dev = torch.device(device)
    torch.manual_seed(int(seed))
    probe = nn.Linear(int(x.size(1)), n_classes).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    x_dev = x.to(dev)
    y_dev = y.to(dev)

    for _ in range(max(1, int(epochs))):
        probe.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x_dev[train_idx.to(dev)]), y_dev[train_idx.to(dev)])
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(x_dev[val_idx.to(dev)])
        pred = logits.argmax(dim=1)
        target = y_dev[val_idx.to(dev)]
        correct = int((pred == target).sum().item())
        total = int(target.numel())
        acc = 100.0 * correct / max(1, total)
    return {
        "acc": acc,
        "correct": correct,
        "total": total,
        "num_classes": n_classes,
        "feature_dim": int(x.size(1)),
        "epochs": int(epochs),
    }


def run_four_probes(
    *,
    z_t: Any,
    z_r: Any,
    tx_labels: Any,
    rx_labels: Any,
    epochs: int = 200,
    seed: int = 0,
    val_fraction: float = 0.2,
    device: str | torch.device = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """Run the four disentanglement probes required by the VMB design report."""

    return {
        "acc_y_given_zt": train_linear_probe(
            z_t, tx_labels, epochs=epochs, seed=seed, val_fraction=val_fraction, device=device
        ),
        "acc_d_given_zt": train_linear_probe(
            z_t, rx_labels, epochs=epochs, seed=seed + 1, val_fraction=val_fraction, device=device
        ),
        "acc_d_given_zr": train_linear_probe(
            z_r, rx_labels, epochs=epochs, seed=seed + 2, val_fraction=val_fraction, device=device
        ),
        "acc_y_given_zr": train_linear_probe(
            z_r, tx_labels, epochs=epochs, seed=seed + 3, val_fraction=val_fraction, device=device
        ),
    }


def _load_payload(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("probe input must be a mapping with z_t/z_r and tx/rx labels")
    return payload


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f"none of the expected keys exist: {', '.join(keys)}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run FedCVS-RFFI-VMB four-probe diagnostics.")
    parser.add_argument("--features", required=True, help="JSON/PT file with z_t/z_id, z_r/z_dom, tx/y, rx/d tensors.")
    parser.add_argument("--output", required=True, help="Path to write probe metrics JSON.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = _load_payload(Path(args.features))
    metrics = run_four_probes(
        z_t=_first(payload, "z_t", "z_id"),
        z_r=_first(payload, "z_r", "z_dom"),
        tx_labels=_first(payload, "tx", "tx_labels", "y"),
        rx_labels=_first(payload, "rx", "rx_labels", "d"),
        epochs=int(args.epochs),
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
