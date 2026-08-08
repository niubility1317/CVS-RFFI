#!/usr/bin/env python
"""Fit and score the frozen Phase1 GI-EpiOR source-only rejector."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_gi_epior import (  # noqa: E402
    GI_EPIOR_THRESHOLD,
    GIEpiORError,
    GIEpiORHead,
    GIEpiORRuntime,
    bundle_payload,
    canonical_physical_ids,
    deterministic_reference_query_split,
    fit_gi_epior,
    runtime_state_bytes,
)
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


def _as_strings(value: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape == ():
        return np.asarray([canonical_tx_id(arr.item())] * int(n), dtype=object)
    rows = [canonical_tx_id(item) for item in arr.reshape(-1).tolist()]
    if len(rows) != int(n):
        raise GIEpiORError("metadata length does not match features")
    return np.asarray(rows, dtype=object)


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "features" not in data.files or "tx_logits" not in data.files:
            raise GIEpiORError("feature NPZ requires features and tx_logits")
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32)
        if features.ndim != 2 or logits.ndim != 2 or features.shape[0] != logits.shape[0]:
            raise GIEpiORError("feature/logit shape mismatch")
        if not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise GIEpiORError("feature/logit arrays must be finite")
        n = int(features.shape[0])

        def pick(name: str) -> np.ndarray:
            if name not in data.files:
                raise GIEpiORError(f"feature NPZ missing metadata key: {name}")
            return _as_strings(np.asarray(data[name]), n)

        payload = {
            "features": features,
            "tx_logits": logits,
            "tx_ids": pick("tx_ids"),
            "rx_ids": pick("rx_ids"),
            "day_ids": pick("day_ids"),
            "eq_ids": pick("eq_ids"),
            "sig_ids": pick("sig_ids"),
            "dataset_role": pick("dataset_role"),
            "channel_views": pick("channel_views"),
            "sat_scenarios": pick("sat_scenarios"),
        }
    payload["physical_ids"] = canonical_physical_ids(
        payload["tx_ids"], payload["rx_ids"], payload["day_ids"], payload["eq_ids"], payload["sig_ids"]
    )
    return payload


def _save_bundle(path: str | Path, payload: Mapping[str, Any]) -> None:
    state = payload["head_state"]
    arrays: dict[str, Any] = {
        "prototypes": np.asarray(payload["prototypes"], dtype=np.float32),
        "scales": np.asarray(payload["scales"], dtype=np.float32),
        "head_0_weight": np.asarray(state["net.0.weight"], dtype=np.float32),
        "head_0_bias": np.asarray(state["net.0.bias"], dtype=np.float32),
        "head_2_weight": np.asarray(state["net.2.weight"], dtype=np.float32),
        "head_2_bias": np.asarray(state["net.2.bias"], dtype=np.float32),
        "manifest_json": np.asarray(
            json.dumps(
                {
                    "schema": payload["schema"],
                    "class_ids": payload["class_ids"],
                    "threshold": payload["threshold"],
                    "fit_receipt": payload["fit_receipt"],
                    "runtime_state_bytes": payload["runtime_state_bytes"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)


def _load_bundle(path: str | Path) -> tuple[GIEpiORRuntime, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        head = GIEpiORHead()
        state = {
            "net.0.weight": torch.as_tensor(np.asarray(data["head_0_weight"]), dtype=torch.float32),
            "net.0.bias": torch.as_tensor(np.asarray(data["head_0_bias"]), dtype=torch.float32),
            "net.2.weight": torch.as_tensor(np.asarray(data["head_2_weight"]), dtype=torch.float32),
            "net.2.bias": torch.as_tensor(np.asarray(data["head_2_bias"]), dtype=torch.float32),
        }
        head.load_state_dict(state, strict=True)
        runtime = GIEpiORRuntime(
            torch.as_tensor(np.asarray(data["prototypes"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(data["scales"]), dtype=torch.float32),
            head.eval(),
        ).eval()
    if float(manifest.get("threshold", -1.0)) != GI_EPIOR_THRESHOLD:
        raise GIEpiORError("bundle threshold is not frozen at 0.5")
    return runtime, manifest


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _auc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float | None:
    known = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    unknown = np.asarray(unknown_scores, dtype=np.float64).reshape(-1)
    if known.size == 0 or unknown.size == 0:
        return None
    wins = (unknown[:, None] > known[None, :]).sum(dtype=np.float64)
    ties = (unknown[:, None] == known[None, :]).sum(dtype=np.float64)
    return float((wins + 0.5 * ties) / float(known.size * unknown.size))


def _rate(mask: np.ndarray, correct: np.ndarray) -> float | None:
    count = int(mask.sum())
    return None if count == 0 else float(np.asarray(correct, dtype=bool)[mask].mean())


def _min_group_rate(groups: Sequence[Any], mask: np.ndarray, correct: np.ndarray) -> float | None:
    group_values = np.asarray(groups, dtype=object)
    rates = [_rate(mask & (group_values == group), correct) for group in sorted(set(group_values[mask].tolist()))]
    finite = [value for value in rates if value is not None]
    return None if not finite else float(min(finite))


def _fit(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    source_tx_ids = tuple(parse_tx_id_list(args.source_tx_ids))
    source_role = str(args.source_role)
    source_mask = np.isin(payload["tx_ids"], np.asarray(source_tx_ids, dtype=object)) & (
        payload["dataset_role"] == source_role
    )
    if int(source_mask.sum()) == 0:
        raise GIEpiORError("no source rows selected for fit")
    result = fit_gi_epior(
        torch.as_tensor(payload["features"][source_mask], dtype=torch.float32),
        payload["tx_ids"][source_mask],
        payload["physical_ids"][source_mask],
        source_tx_ids,
    )
    bundle = bundle_payload(result, source_tx_ids)
    _save_bundle(args.output_bundle, bundle)
    scripted = torch.jit.script(result.runtime.eval())
    script_path = Path(args.output_torchscript)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(script_path))
    sample = torch.as_tensor(payload["features"][source_mask][: min(512, int(source_mask.sum()))], dtype=torch.float32)
    with torch.no_grad():
        eager = result.runtime(sample)
        traced = scripted(sample)
    parity = max(float(torch.max(torch.abs(left - right)).item()) for left, right in zip(eager, traced))
    if parity > 1.0e-5:
        raise GIEpiORError(f"eager/TorchScript parity exceeded: {parity}")
    iterations = 20
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            result.runtime(sample)
    latency_ms = 1000.0 * (time.perf_counter() - started) / float(iterations)
    receipt = {
        **result.receipt,
        "feature_npz": str(args.feature_npz),
        "source_role": source_role,
        "fit_rows": int(source_mask.sum()),
        "non_source_rows_excluded_from_fit": int((~source_mask).sum()),
        "outer_zero_fit": True,
        "bundle": str(args.output_bundle),
        "torchscript": str(args.output_torchscript),
        "eager_torchscript_max_abs": parity,
        "latency_batch_rows": int(sample.size(0)),
        "latency_ms_per_batch_cpu": float(latency_ms),
        "runtime_state_bytes": runtime_state_bytes(result.runtime),
        "device": "cpu",
    }
    _atomic_json(args.output_receipt, receipt)
    return receipt


def _score(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    runtime, manifest = _load_bundle(args.bundle)
    source_tx_ids = tuple(parse_tx_id_list(args.source_tx_ids))
    if list(source_tx_ids) != list(manifest.get("class_ids", [])):
        raise GIEpiORError("bundle/source class order mismatch")
    source_mask = np.isin(payload["tx_ids"], np.asarray(source_tx_ids, dtype=object)) & (
        payload["dataset_role"] == str(args.source_role)
    )
    _, source_query_local, split_receipt = deterministic_reference_query_split(
        payload["tx_ids"][source_mask], payload["physical_ids"][source_mask], source_tx_ids
    )
    source_indices = np.flatnonzero(source_mask)
    known_query = np.zeros(payload["features"].shape[0], dtype=bool)
    known_query[source_indices[source_query_local]] = True
    held_ids = set(parse_tx_id_list(args.held_tx_ids))
    held = (payload["dataset_role"] == str(args.held_role)) & np.asarray(
        [not held_ids or value in held_ids for value in payload["tx_ids"]], dtype=bool
    )
    proxy = payload["dataset_role"] == str(args.proxy_role)
    with torch.no_grad():
        e_epi_t, d_class_t, ratio_t = runtime(torch.as_tensor(payload["features"], dtype=torch.float32))
    e_epi = e_epi_t.cpu().numpy()
    d_class = d_class_t.cpu().numpy()
    ratio = ratio_t.cpu().numpy()
    accepted = e_epi < GI_EPIOR_THRESHOLD
    pred_class = np.asarray(payload["tx_logits"]).argmax(axis=1)
    pred_tx = np.asarray(
        [source_tx_ids[index] if 0 <= int(index) < len(source_tx_ids) else str(index) for index in pred_class],
        dtype=object,
    )
    closed_correct = pred_tx == payload["tx_ids"]
    full_correct = closed_correct & accepted
    known_closed = _rate(known_query, closed_correct)
    known_full = _rate(known_query, full_correct)
    held_far = _rate(held, accepted)
    proxy_far = _rate(proxy, accepted)
    metrics = {
        "schema": "cvs.phase1.gi_epior_score.v1",
        "method": "GI-EpiOR",
        "threshold": GI_EPIOR_THRESHOLD,
        "threshold_policy": "fixed_sigmoid_0p5_no_quantile_no_outer_calibration",
        "feature_npz": str(args.feature_npz),
        "bundle": str(args.bundle),
        "view_name": str(args.view_name),
        "source_tx_ids": list(source_tx_ids),
        "held_tx_ids": sorted(held_ids),
        "fit_receipt": manifest.get("fit_receipt", {}),
        "split_receipt": split_receipt,
        "known_query_count": int(known_query.sum()),
        "known_closed_accuracy_no_reject": known_closed,
        "known_full_accuracy_after_reject": known_full,
        "known_drop_pp": None if known_closed is None or known_full is None else 100.0 * (known_closed - known_full),
        "known_coverage": _rate(known_query, accepted),
        "known_min_class_full_accuracy": _min_group_rate(payload["tx_ids"], known_query, full_correct),
        "known_min_rx_full_accuracy": _min_group_rate(payload["rx_ids"], known_query, full_correct),
        "known_min_day_full_accuracy": _min_group_rate(payload["day_ids"], known_query, full_correct),
        "held_count": int(held.sum()),
        "held_far": held_far,
        "held_safe_rejection": None if held_far is None else 1.0 - held_far,
        "held_auc": _auc(e_epi[known_query], e_epi[held]),
        "proxy_count": int(proxy.sum()),
        "proxy_far": proxy_far,
        "proxy_safe_rejection": None if proxy_far is None else 1.0 - proxy_far,
        "proxy_auc": _auc(e_epi[known_query], e_epi[proxy]),
        "nct_ratio_continuous_only": {
            "thresholded": False,
            "held_auc": _auc(ratio[known_query], ratio[held]),
            "proxy_auc": _auc(ratio[known_query], ratio[proxy]),
        },
        "d_class_shape": list(d_class.shape),
        "p_local_source": "frozen_checkpoint_tx_logits_argmax",
        "outer_used_for_fit_or_calibration": False,
    }
    _atomic_json(args.output_json, metrics)
    rows: list[dict[str, Any]] = []
    for index in range(payload["features"].shape[0]):
        rows.append(
            {
                "row": index,
                "physical_id": payload["physical_ids"][index],
                "role": payload["dataset_role"][index],
                "tx_id": payload["tx_ids"][index],
                "rx_id": payload["rx_ids"][index],
                "day_id": payload["day_ids"][index],
                "channel_view": payload["channel_views"][index],
                "sat_scenario": payload["sat_scenarios"][index],
                "known_query": int(known_query[index]),
                "held_query": int(held[index]),
                "proxy_query": int(proxy[index]),
                "p_local": pred_tx[index],
                "e_epi": f"{float(e_epi[index]):.9g}",
                "nct_ratio_continuous": f"{float(ratio[index]):.9g}",
                "accepted": int(accepted[index]),
                "closed_correct_known": int(bool(known_query[index] and closed_correct[index])),
                "full_correct_known": int(bool(known_query[index] and full_correct[index])),
            }
        )
    target = Path(args.output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit", help="Fit one frozen GI-EpiOR head from source rows only")
    fit_parser.add_argument("--feature-npz", required=True)
    fit_parser.add_argument("--source-tx-ids", required=True)
    fit_parser.add_argument("--source-role", default="source")
    fit_parser.add_argument("--output-bundle", required=True)
    fit_parser.add_argument("--output-torchscript", required=True)
    fit_parser.add_argument("--output-receipt", required=True)
    score_parser = sub.add_parser("score", help="Score one immutable feature NPZ with a frozen GI-EpiOR bundle")
    score_parser.add_argument("--feature-npz", required=True)
    score_parser.add_argument("--bundle", required=True)
    score_parser.add_argument("--source-tx-ids", required=True)
    score_parser.add_argument("--held-tx-ids", required=True)
    score_parser.add_argument("--source-role", default="source")
    score_parser.add_argument("--held-role", default="target_old")
    score_parser.add_argument("--proxy-role", default="proxy_unknown")
    score_parser.add_argument("--view-name", required=True)
    score_parser.add_argument("--output-json", required=True)
    score_parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = _fit(args) if args.command == "fit" else _score(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
