from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .datasets import SyntheticIQDataset, collate_iq_dict
from .metrics import accuracy_from_logits


def synthetic_iq_loader(
    *,
    num_tx: int,
    num_rx: int,
    samples_per_pair: int = 4,
    length: int = 256,
    batch_size: int = 16,
    seed: int = 0,
    transform=None,
    shuffle: bool = True,
) -> DataLoader:
    ds = SyntheticIQDataset(
        num_tx=num_tx,
        num_rx=num_rx,
        samples_per_pair=samples_per_pair,
        length=length,
        seed=seed,
        transform=transform,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_iq_dict)


@torch.no_grad()
def evaluate_tx_logits(model, loader: DataLoader, device: torch.device, output_key: str = "tx_logits") -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    for batch in loader:
        x = batch["spec"].to(device) if "spec" in batch else batch["iq"].to(device)
        y = batch["label"].to(device)
        out = model(x)
        logits = out[output_key] if isinstance(out, dict) else out[0]
        pred = logits.argmax(dim=1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
    return {"accuracy": correct / max(1, total)}
