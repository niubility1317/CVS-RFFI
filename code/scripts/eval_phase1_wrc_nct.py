#!/usr/bin/env python
"""Build and score one frozen WRC-NCT readout from a GI-EpiOR v3 bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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
    canonical_physical_ids,
)
from cvsrffi.phase1_wrc_nct import (  # noqa: E402
    WRC_NCT_ALPHA,
    WRCNCTError,
    WRCNCTRuntime,
    fit_wrc_nct,
    runtime_state_bytes,
)
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


def _as_strings(value: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        return np.asarray([text] * int(n), dtype=object)
    rows = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in arr.reshape(-1).tolist()]
    if len(rows) != int(n):
        raise WRCNCTError("metadata length does not match features")
    return np.asarray(rows, dtype=object)


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "features" not in data.files or "tx_logits" not in data.files:
            raise WRCNCTError("feature NPZ requires features and tx_logits")
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32)
        if features.ndim != 2 or logits.ndim != 2 or features.shape[0] != logits.shape[0] or features.shape[0] == 0:
            raise WRCNCTError("feature/logit shape mismatch or empty rows")
        if not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise WRCNCTError("feature/logit arrays must be finite")
        n = int(features.shape[0])

        def pick(name: str) -> np.ndarray:
            if name not in data.files:
                raise WRCNCTError(f"feature NPZ missing metadata key: {name}")
            return _as_strings(np.asarray(data[name]), n)

        payload = {
            "features": features,
            "tx_logits": logits,
            "tx_ids": np.asarray([canonical_tx_id(value) for value in pick("tx_ids")], dtype=object),
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


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_gi_bundle(path: str | Path) -> tuple[GIEpiORRuntime, dict[str, Any], str]:
    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as data:
        required = {
            "prototypes",
            "scales",
            "head_0_weight",
            "head_0_bias",
            "head_2_weight",
            "head_2_bias",
            "manifest_json",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise WRCNCTError(f"GI bundle missing fields: {','.join(missing)}")
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        head = GIEpiORHead()
        head.load_state_dict(
            {
                "net.0.weight": torch.as_tensor(np.asarray(data["head_0_weight"]), dtype=torch.float32),
                "net.0.bias": torch.as_tensor(np.asarray(data["head_0_bias"]), dtype=torch.float32),
                "net.2.weight": torch.as_tensor(np.asarray(data["head_2_weight"]), dtype=torch.float32),
                "net.2.bias": torch.as_tensor(np.asarray(data["head_2_bias"]), dtype=torch.float32),
            },
            strict=True,
        )
        runtime = GIEpiORRuntime(
            torch.as_tensor(np.asarray(data["prototypes"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(data["scales"]), dtype=torch.float32),
            head.eval(),
        ).eval()
    if str(manifest.get("schema")) != "cvs.phase1.gi_epior_bundle.v1":
        raise WRCNCTError("upstream bundle is not a GI-EpiOR v3-compatible bundle")
    if float(manifest.get("threshold", -1.0)) != GI_EPIOR_THRESHOLD:
        raise WRCNCTError("upstream GI-EpiOR bundle threshold is not frozen at 0.5")
    return runtime, manifest, _sha256(bundle_path)


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
    values = np.asarray(groups, dtype=object)
    rates = [_rate(mask & (values == group), correct) for group in sorted(set(values[mask].tolist()))]
    finite = [value for value in rates if value is not None]
    return None if not finite else float(min(finite))


def _drop_pp(closed: float | None, full: float | None) -> float | None:
    return None if closed is None or full is None else 100.0 * (closed - full)


def _parity_receipt(
    gi_runtime: GIEpiORRuntime,
    runtime: WRCNCTRuntime,
    sample: torch.Tensor,
    script_path: str | Path,
) -> dict[str, Any]:
    scripted = torch.jit.script(runtime.eval())
    target = Path(script_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(target))
    with torch.no_grad():
        _, gi_d_class, gi_ratio = gi_runtime.eval()(sample)
        eager = runtime.eval()(sample)
        scripted_out = scripted(sample)
    gi_sorted = torch.sort(gi_d_class, dim=1).values
    ratio_delta = float(torch.max(torch.abs(eager[2] - gi_ratio)).item())
    d1_delta = float(torch.max(torch.abs(eager[0] - gi_sorted[:, 0])).item())
    d2_delta = float(torch.max(torch.abs(eager[1] - gi_sorted[:, 1])).item())
    script_numeric = max(float(torch.max(torch.abs(left - right)).item()) for left, right in zip(eager[:3], scripted_out[:3]))
    script_accept = bool(torch.equal(eager[3], scripted_out[3]))
    maximum = max(ratio_delta, d1_delta, d2_delta, script_numeric)
    if maximum > 1.0e-5 or not script_accept:
        raise WRCNCTError(f"GI/WRC or eager/TorchScript parity exceeded: {maximum}")
    return {
        "sample_rows": int(sample.size(0)),
        "gi_ratio_max_abs": ratio_delta,
        "gi_d1_max_abs": d1_delta,
        "gi_d2_max_abs": d2_delta,
        "eager_torchscript_numeric_max_abs": script_numeric,
        "eager_torchscript_accept_equal": script_accept,
    }


def _write_scores(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise WRCNCTError("cannot write empty score CSV")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    gi_runtime, gi_manifest, gi_sha256 = _load_gi_bundle(args.gi_bundle)
    source_tx_ids = tuple(parse_tx_id_list(args.source_tx_ids))
    if list(source_tx_ids) != list(gi_manifest.get("class_ids", [])):
        raise WRCNCTError("source TX order must exactly match the upstream GI bundle")
    source_mask = np.isin(payload["tx_ids"], np.asarray(source_tx_ids, dtype=object)) & (
        payload["dataset_role"] == str(args.source_role)
    )
    if int(source_mask.sum()) == 0:
        raise WRCNCTError("no source-known rows selected")
    source_indices = np.flatnonzero(source_mask)
    result = fit_wrc_nct(
        torch.as_tensor(payload["features"][source_mask], dtype=torch.float32),
        payload["tx_ids"][source_mask],
        payload["rx_ids"][source_mask],
        payload["physical_ids"][source_mask],
        source_tx_ids,
        gi_runtime,
    )
    reference = np.zeros(payload["features"].shape[0], dtype=bool)
    calibration = np.zeros_like(reference)
    known_eval = np.zeros_like(reference)
    reference[source_indices[result.reference_mask]] = True
    calibration[source_indices[result.calibration_mask]] = True
    known_eval[source_indices[result.evaluation_mask]] = True
    held_ids = set(parse_tx_id_list(args.held_tx_ids))
    held = (payload["dataset_role"] == str(args.held_role)) & np.asarray(
        [not held_ids or value in held_ids for value in payload["tx_ids"]], dtype=bool
    )
    proxy = payload["dataset_role"] == str(args.proxy_role)
    sample_rows = min(512, int(payload["features"].shape[0]))
    sample = torch.as_tensor(payload["features"][:sample_rows], dtype=torch.float32)
    parity = _parity_receipt(gi_runtime, result.runtime, sample, args.output_torchscript)
    all_features = torch.as_tensor(payload["features"], dtype=torch.float32)
    with torch.no_grad():
        d1_t, d2_t, ratio_t, accepted_t = result.runtime(all_features)
    d1 = np.asarray(d1_t.detach().cpu().tolist(), dtype=np.float32).reshape(-1)
    d2 = np.asarray(d2_t.detach().cpu().tolist(), dtype=np.float32).reshape(-1)
    ratio = np.asarray(ratio_t.detach().cpu().tolist(), dtype=np.float32).reshape(-1)
    accepted = np.asarray(accepted_t.detach().cpu().tolist(), dtype=bool).reshape(-1)
    pred_class = np.asarray(payload["tx_logits"]).argmax(axis=1)
    pred_tx = np.asarray(
        [source_tx_ids[index] if 0 <= int(index) < len(source_tx_ids) else str(index) for index in pred_class],
        dtype=object,
    )
    closed_correct = np.asarray(pred_tx == payload["tx_ids"], dtype=bool)
    full_correct = closed_correct & accepted
    known_closed = _rate(known_eval, closed_correct)
    known_full = _rate(known_eval, full_correct)
    min_class_closed = _min_group_rate(payload["tx_ids"], known_eval, closed_correct)
    min_class_full = _min_group_rate(payload["tx_ids"], known_eval, full_correct)
    min_rx_closed = _min_group_rate(payload["rx_ids"], known_eval, closed_correct)
    min_rx_full = _min_group_rate(payload["rx_ids"], known_eval, full_correct)
    min_day_closed = _min_group_rate(payload["day_ids"], known_eval, closed_correct)
    min_day_full = _min_group_rate(payload["day_ids"], known_eval, full_correct)
    held_far = _rate(held, accepted)
    proxy_far = _rate(proxy, accepted)
    readout = {
        **result.receipt,
        "immutable": True,
        "feature_npz": str(args.feature_npz),
        "source_role": str(args.source_role),
        "source_tx_ids": list(source_tx_ids),
        "upstream_gi_bundle": str(args.gi_bundle),
        "upstream_gi_bundle_sha256": gi_sha256,
        "upstream_gi_bundle_schema": gi_manifest.get("schema"),
        "threshold_torchscript": str(args.output_torchscript),
        "runtime_state_bytes": runtime_state_bytes(result.runtime),
        "parity": parity,
        "outer_used_for_fit_or_calibration": False,
    }
    _atomic_json(args.output_readout_json, readout)
    metrics = {
        "schema": "cvs.phase1.wrc_nct_score.v1",
        "method": "WRC-NCT",
        "evidence_boundary": "PHASE1_SOURCE_ONLY_DEVELOPMENT_NON_CONFIRMATORY",
        "alpha": WRC_NCT_ALPHA,
        "threshold": result.receipt["tau"],
        "threshold_policy": result.receipt["threshold_policy"],
        "feature_npz": str(args.feature_npz),
        "readout": str(args.output_readout_json),
        "upstream_gi_bundle": str(args.gi_bundle),
        "upstream_gi_bundle_sha256": gi_sha256,
        "view_name": str(args.view_name),
        "source_tx_ids": list(source_tx_ids),
        "held_tx_ids": sorted(held_ids),
        "split_receipt": result.receipt["split"],
        "known_evaluation_count": int(known_eval.sum()),
        "known_closed_accuracy_no_reject": known_closed,
        "known_full_accuracy_after_reject": known_full,
        "known_drop_pp": _drop_pp(known_closed, known_full),
        "known_coverage": _rate(known_eval, accepted),
        "known_min_class_closed_accuracy_no_reject": min_class_closed,
        "known_min_class_full_accuracy": min_class_full,
        "known_min_class_drop_pp": _drop_pp(min_class_closed, min_class_full),
        "known_min_rx_closed_accuracy_no_reject": min_rx_closed,
        "known_min_rx_full_accuracy": min_rx_full,
        "known_min_rx_drop_pp": _drop_pp(min_rx_closed, min_rx_full),
        "known_min_day_closed_accuracy_no_reject": min_day_closed,
        "known_min_day_full_accuracy": min_day_full,
        "known_min_day_drop_pp": _drop_pp(min_day_closed, min_day_full),
        "held_count": int(held.sum()),
        "held_far": held_far,
        "held_safe_rejection": None if held_far is None else 1.0 - held_far,
        "held_auc": _auc(ratio[known_eval], ratio[held]),
        "proxy_count": int(proxy.sum()),
        "proxy_far": proxy_far,
        "proxy_safe_rejection": None if proxy_far is None else 1.0 - proxy_far,
        "proxy_auc": _auc(ratio[known_eval], ratio[proxy]),
        "p_local_source": "frozen_checkpoint_tx_logits_argmax",
        "outer_used_for_fit_or_calibration": False,
        "runtime_state_bytes": runtime_state_bytes(result.runtime),
        "parity": parity,
    }
    _atomic_json(args.output_metrics_json, metrics)
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
                "source_reference": int(reference[index]),
                "source_calibration": int(calibration[index]),
                "known_evaluation": int(known_eval[index]),
                "held_query": int(held[index]),
                "proxy_query": int(proxy[index]),
                "p_local": pred_tx[index],
                "d1": f"{float(d1[index]):.9g}",
                "d2": f"{float(d2[index]):.9g}",
                "nct_ratio": f"{float(ratio[index]):.9g}",
                "accepted": int(accepted[index]),
                "closed_correct_known": int(bool(known_eval[index] and closed_correct[index])),
                "full_correct_known": int(bool(known_eval[index] and full_correct[index])),
            }
        )
    _write_scores(args.output_scores_csv, rows)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-npz", required=True)
    parser.add_argument("--gi-bundle", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--held-tx-ids", required=True)
    parser.add_argument("--source-role", default="source")
    parser.add_argument("--held-role", default="target_old")
    parser.add_argument("--proxy-role", default="proxy_unknown")
    parser.add_argument("--view-name", required=True)
    parser.add_argument("--output-readout-json", required=True)
    parser.add_argument("--output-torchscript", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    parser.add_argument("--output-scores-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = _run(args)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
