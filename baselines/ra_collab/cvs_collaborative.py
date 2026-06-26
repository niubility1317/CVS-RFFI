from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Tuple

import torch

from baselines.common.cvs_trainer import logits_from_output
from baselines.ra_collab.collaborative_inference import adaptive_soft_fusion, soft_fusion


def _batch_value(batch: Dict[str, Any], key: str, index: int) -> int:
    value = batch.get(key)
    if torch.is_tensor(value):
        return int(value[index].item())
    if isinstance(value, (list, tuple)):
        return int(value[index])
    if key == "label":
        return int(batch["label"][index].item())
    raise KeyError(f"Batch does not contain grouping key {key!r}.")


def _group_key(batch: Dict[str, Any], index: int, group_keys: Iterable[str]) -> Tuple[int, ...]:
    return tuple(_batch_value(batch, key, index) for key in group_keys)


@torch.no_grad()
def evaluate_collaborative_tx(
    model,
    loader,
    device,
    *,
    forward_fn: Callable[[Any, Dict[str, Any], torch.device], Any] | None = None,
    fusion: str = "soft",
    group_keys: Iterable[str] = ("label", "day", "sig_i"),
) -> Dict[str, float]:
    """Evaluate receiver-collaborative inference on CVS/WiSig aligned groups.

    Samples sharing ``group_keys`` are treated as simultaneous observations of one
    transmitter packet from multiple receivers. The final prediction is computed
    by soft or adaptive soft fusion over receiver predictions.
    """

    model.eval()
    grouped_logits: Dict[Tuple[int, ...], list[torch.Tensor]] = defaultdict(list)
    grouped_labels: Dict[Tuple[int, ...], int] = {}
    grouped_snr: Dict[Tuple[int, ...], list[float]] = defaultdict(list)
    group_keys = tuple(group_keys)
    for batch in loader:
        labels = batch["label"].to(device)
        if forward_fn is None:
            output = model(batch["iq"].to(device), grl_lambda=0.0, return_rx=False)
        else:
            output = forward_fn(model, batch, device)
        logits = logits_from_output(output).detach().cpu()
        for i in range(int(labels.numel())):
            key = _group_key(batch, i, group_keys)
            grouped_logits[key].append(logits[i])
            grouped_labels[key] = int(labels[i].item())
            meta = batch.get("meta", [{}])[i] if isinstance(batch.get("meta"), list) else {}
            if isinstance(meta, dict) and "snr" in meta:
                grouped_snr[key].append(float(meta["snr"]))

    correct = 0
    total = 0
    receiver_observations = 0
    for key, preds in grouped_logits.items():
        stacked = torch.stack(preds, dim=0)
        receiver_observations += int(stacked.size(0))
        if fusion == "adaptive" and len(grouped_snr.get(key, [])) == stacked.size(0):
            fused = adaptive_soft_fusion(stacked, torch.tensor(grouped_snr[key]), snr_scale="linear")
        elif fusion == "soft":
            fused = soft_fusion(stacked)
        else:
            fused = soft_fusion(stacked)
        pred = int(fused.argmax(dim=0).item())
        correct += int(pred == grouped_labels[key])
        total += 1

    return {
        "tx_acc": 100.0 * correct / max(1, total),
        "tx_correct": correct,
        "tx_total": total,
        "num_groups": total,
        "receiver_observations": receiver_observations,
    }
