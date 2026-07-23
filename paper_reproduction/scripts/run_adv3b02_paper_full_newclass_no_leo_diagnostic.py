#!/usr/bin/env python3
"""Run a matched diagnostic with raw new-class IQ and unchanged LEO old-class IQ.

This is deliberately not a Stage2 or formal CVS scenario runner. It reuses the
v7 physical cache rows and split rule, replaces only ``target_new`` IQ with the
same ManyTx physical records before LEO overlay, finalizes predictions, and only
then computes metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
SCRIPT_ROOT = CODE_ROOT / "scripts"
for value in (str(REPO_ROOT), str(CODE_ROOT), str(SCRIPT_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT), str(SCRIPT_ROOT)):
    sys.path.insert(0, value)

from build_cvs_stage2_predictor_bundle import _select_support_query  # noqa: E402
from cvsrffi.checkpoint_loading import (  # noqa: E402
    build_exact_ssdg_model_from_checkpoint,
)
from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl  # noqa: E402
from model_dual_cvsincnet import backbone_forward_compat  # noqa: E402
from paper_reproduction.cvs_aligned.adv3b02_paper_full_ci import (  # noqa: E402
    METHODS as LEGACY_METHODS,
    fit_paper_full,
    predict_after as predict_after_legacy,
    predict_before as predict_before_legacy,
)
from paper_reproduction.cvs_aligned.adv3b02_official_repo_ci import (  # noqa: E402
    METHODS as OFFICIAL_METHODS,
    fit_official_repo,
    predict_after as predict_after_official,
    predict_before as predict_before_official,
)
from paper_reproduction.scripts.run_adv3b02_paper_full_ci_truth_free_predictor import (  # noqa: E402
    _tensor,
)


STATUS = "DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL"
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _resolve(base: Path, raw: str) -> Path:
    value = Path(str(raw))
    return value if value.is_absolute() else (base / value).resolve()


def _load_cache_set(path: Path) -> dict[str, dict[str, np.ndarray]]:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    scenario_map = dict(manifest["cache_npz_by_scenario"])
    if tuple(scenario_map) != SCENARIOS:
        raise ValueError("diagnostic source cache scenario tuple drift")
    result: dict[str, dict[str, np.ndarray]] = {}
    reference_ids: np.ndarray | None = None
    for scenario in SCENARIOS:
        cache_path = _resolve(path.parent, scenario_map[scenario]).resolve(strict=True)
        expected = str(manifest["cache_sha256_by_scenario"][scenario])
        if _sha256_file(cache_path) != expected:
            raise ValueError(f"source cache hash drift: {scenario}")
        with np.load(cache_path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        ids = np.asarray(arrays["sample_ids"]).astype(str)
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(ids, reference_ids):
            raise ValueError("source cache physical rows drift across scenarios")
        result[scenario] = arrays
    return result


def _raw_key(
    tx: Any, rx: Any, day: Any, equalized: Any, sig: Any
) -> tuple[str, str, str, str, str]:
    return str(tx), str(rx), str(day), str(equalized), str(sig)


def build_raw_new_lookup(
    manytx: Mapping[str, Any],
    *,
    receiver: str,
    cache_arrays: Mapping[str, np.ndarray],
) -> tuple[dict[tuple[str, str, str, str, str], np.ndarray], dict[str, Any]]:
    roles = np.asarray(cache_arrays["dataset_role"]).astype(str)
    new_mask = roles == "target_new"
    if not bool(np.any(new_mask)):
        raise ValueError("diagnostic cache contains no target_new rows")
    tx_values = np.asarray(cache_arrays["tx_ids"]).astype(str)
    day_values = np.asarray(cache_arrays["day_ids"]).astype(str)
    new_labels = sorted(set(tx_values[new_mask].tolist()))
    new_days = sorted(set(day_values[new_mask].tolist()))
    tx_list = list(manytx.get("tx_list", []))
    rx_list = list(manytx.get("rx_list", []))
    day_list = list(manytx.get("capture_date_list", []))
    tx_keep = [tx_list.index(value) for value in new_labels]
    rx_keep = [rx_list.index(str(receiver))]
    day_keep = [day_list.index(value) for value in new_days]
    dataset = WiSigCompactDataset(
        dict(manytx),
        out_len=256,
        equalized=1,
        tx_keep=tx_keep,
        rx_keep=rx_keep,
        day_keep=day_keep,
        domain="rx",
        max_samples_per_combo=None,
        sample_strategy="front",
        seed=0,
    )
    needed = {
        _raw_key(
            cache_arrays["tx_ids"][index],
            cache_arrays["rx_ids"][index],
            cache_arrays["day_ids"][index],
            cache_arrays["eq_ids"][index],
            cache_arrays["sig_ids"][index],
        )
        for index in np.flatnonzero(new_mask).tolist()
    }
    lookup: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    for index in range(len(dataset)):
        row, _label, _domain, meta = dataset[index]
        key = _raw_key(
            meta["tx"],
            meta["rx"],
            meta["day"],
            meta["equalized"],
            meta["sig_i"],
        )
        if key in needed:
            lookup[key] = np.asarray(row.detach().cpu().tolist(), dtype=np.float32)
            if len(lookup) == len(needed):
                break
    missing = sorted(needed - set(lookup))
    if missing:
        raise ValueError(f"cannot recover matched raw target_new rows: {missing[:3]}")
    return lookup, {
        "receiver": str(receiver),
        "new_class_count_in_cache": len(new_labels),
        "matched_raw_new_rows": len(lookup),
        "raw_new_key_root_sha256": hashlib.sha256(
            json.dumps(sorted(lookup), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def replace_new_class_iq(
    arrays: Mapping[str, np.ndarray],
    raw_lookup: Mapping[tuple[str, str, str, str, str], np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32).copy()
    roles = np.asarray(arrays["dataset_role"]).astype(str)
    new_indices = np.flatnonzero(roles == "target_new")
    old_indices = np.flatnonzero(roles == "target_old")
    old_before = _sha256_array(iq[old_indices])
    for index in new_indices.tolist():
        key = _raw_key(
            arrays["tx_ids"][index],
            arrays["rx_ids"][index],
            arrays["day_ids"][index],
            arrays["eq_ids"][index],
            arrays["sig_ids"][index],
        )
        iq[index] = raw_lookup[key]
    if _sha256_array(iq[old_indices]) != old_before:
        raise RuntimeError("old-class LEO IQ changed during raw-new replacement")
    return iq, {
        "old_class_rows_unchanged": int(len(old_indices)),
        "new_class_rows_replaced_with_raw": int(len(new_indices)),
        "old_class_iq_sha256": old_before,
        "new_class_raw_iq_sha256": _sha256_array(iq[new_indices]),
    }


def _load_model(
    checkpoint_path: Path, base_state_path: Path, *, device: torch.device
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    exact, load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=256, device=device
    )
    feature_key = str(getattr(exact, "id_feature_key", "feat_joint"))

    def feature_fn(backbone: torch.nn.Module, rows: torch.Tensor):
        auxiliary = backbone_forward_compat(
            backbone,
            rows,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        feature = auxiliary.get(feature_key)
        if not torch.is_tensor(feature):
            feature = auxiliary.get("feat_joint")
        logits = auxiliary.get("logits")
        if not torch.is_tensor(feature) or not torch.is_tensor(logits):
            raise ValueError("ADV3B02 identity output drift")
        return feature.float(), logits.float()

    state = torch.load(base_state_path, map_location="cpu", weights_only=False)
    schema = state.get("schema")
    if schema == "cvs.adv3b02.official_repo_base_state.v2":
        if int(state.get("base_sample_count", 0)) != 8400:
            raise ValueError("official-repo base state requires exactly 8400 rows")
        base_state = {"csil": state["csil"], "mopc_hr": state["mopc_hr"]}
    elif schema == "cvs.adv3b02.paper_full_base_state.v1":
        base_state = {
            "old_fingerprints": state["old_fingerprints"].to(device),
            "old_prototypes": state["old_prototypes"].to(device),
            "fisher": {
                name: value.to(device) for name, value in state["fisher"].items()
            },
        }
    else:
        raise ValueError("paper-full base-state schema drift")
    receipt = {
        "schema": schema,
        "base_sample_count": int(state.get("base_sample_count", 0)),
        "source_receiver_labels": list(state.get("source_receiver_labels", [])),
        "checkpoint_load_audit": load_audit,
    }
    return exact.id_backbone.to(device).eval(), feature_fn, base_state, receipt


def _accuracy(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return (
        float(np.mean(prediction[mask] == truth[mask]))
        if bool(np.any(mask))
        else 0.0
    )


def score_predictions(
    *,
    before: np.ndarray,
    after: np.ndarray,
    truth: np.ndarray,
    old_count: int,
) -> dict[str, Any]:
    old_mask = truth < int(old_count)
    new_mask = ~old_mask
    old_before = _accuracy(before, truth, old_mask)
    old_after = _accuracy(after, truth, old_mask)
    seen_new = _accuracy(after, truth, new_mask)
    harmonic = (
        0.0
        if old_after + seen_new <= 0.0
        else 2.0 * old_after * seen_new / (old_after + seen_new)
    )
    before_by_class: list[float] = []
    after_by_class: list[float] = []
    for class_index in range(int(old_count)):
        class_mask = truth == class_index
        before_by_class.append(_accuracy(before, truth, class_mask))
        after_by_class.append(_accuracy(after, truth, class_mask))
    forgetting = float(
        np.mean(
            [
                max(0.0, old_value - new_value)
                for old_value, new_value in zip(before_by_class, after_by_class)
            ]
        )
    )
    return {
        "old_acc_before_increment": old_before,
        "old_acc_after_increment": old_after,
        "seen_new_acc": seen_new,
        "H_old_new": harmonic,
        "candidate_average_forgetting": forgetting,
        "min_old_class_acc": min(after_by_class),
        "old_class_acc_before": before_by_class,
        "old_class_acc_after": after_by_class,
        "query_count": int(len(truth)),
        "target_old_query_count": int(np.count_nonzero(old_mask)),
        "target_new_query_count": int(np.count_nonzero(new_mask)),
    }


def _loss_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {"steps": 0}
    numeric_keys = sorted(
        {
            key
            for row in trace
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"epoch", "iteration", "stage"}
        }
    )
    return {
        "steps": len(trace),
        "first": trace[0],
        "last": trace[-1],
        "mean": {
            key: float(np.mean([float(row[key]) for row in trace if key in row]))
            for key in numeric_keys
        },
    }


def run_cell(
    cell: Mapping[str, Any],
    package: Mapping[str, Any],
    *,
    cache_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    mixed_iq_by_scenario: Mapping[str, np.ndarray],
    backbone: torch.nn.Module,
    feature_fn,
    base_state: Mapping[str, Any],
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    cell_root = output_root / "cells" / str(cell["cell_id"])
    if cell_root.exists():
        receipt = json.loads(
            (cell_root / "cell_receipt.json").read_text(encoding="utf-8-sig")
        )
        if receipt.get("status") != STATUS:
            raise ValueError("existing diagnostic cell status drift")
        return receipt
    cell_root.mkdir(parents=True, exist_ok=False)
    old_labels = list(package["old_class_labels"])
    new_labels = list(package["new_class_labels"])
    support_labels = [
        *(("target_old", value) for value in old_labels),
        *(("target_new", value) for value in new_labels),
    ]
    scenario_predictions: dict[str, dict[str, np.ndarray]] = {}
    scenario_resources: dict[str, Any] = {}
    query_records_reference: list[dict[str, Any]] | None = None
    for scenario in SCENARIOS:
        arrays = cache_by_scenario[scenario]
        support_idx, query_idx, support_y, support_rank, query_records = (
            _select_support_query(
                dict(arrays),
                receiver=str(package["receiver"]),
                seed=int(package["seed"]),
                support_labels=support_labels,
                reference_query_labels=[],
                support_pool_max_k=20,
                query_per_tx=20,
            )
        )
        if query_records_reference is None:
            query_records_reference = query_records
        elif query_records != query_records_reference:
            raise ValueError("matched query structure drift across scenarios")
        selected = support_rank < int(cell["k_shot"])
        selected_support_idx = support_idx[selected]
        selected_support_y = support_y[selected]
        support_x = _tensor(
            np.asarray(mixed_iq_by_scenario[scenario], dtype=np.float32)[
                selected_support_idx
            ],
            dtype=torch.float32,
            device=device,
        )
        support_y_tensor = _tensor(
            selected_support_y,
            dtype=torch.long,
            device=device,
        )
        method = str(cell["method"])
        if method in OFFICIAL_METHODS:
            fitted = fit_official_repo(
                method,
                copy.deepcopy(backbone),
                support_x,
                support_y_tensor,
                feature_fn=feature_fn,
                old_count=len(old_labels),
                seed=int(cell["seed"]),
                base_state=base_state,
            )
        elif method in LEGACY_METHODS:
            fitted = fit_paper_full(
                method,
                copy.deepcopy(backbone),
                support_x,
                support_y_tensor,
                feature_fn=feature_fn,
                old_count=len(old_labels),
                seed=int(cell["seed"]),
                base_state=base_state,
            )
        else:
            raise ValueError("unsupported comparison method")
        query_x = _tensor(
            np.asarray(mixed_iq_by_scenario[scenario], dtype=np.float32)[query_idx],
            dtype=torch.float32,
            device=device,
        )
        if method in OFFICIAL_METHODS:
            before_tensor = predict_before_official(fitted, query_x)
            after_tensor = predict_after_official(fitted, query_x)
        else:
            before_tensor = predict_before_legacy(fitted, query_x)
            after_tensor = predict_after_legacy(fitted, query_x)
        before = np.asarray(
            before_tensor.detach().cpu().tolist(), dtype=np.int64
        )
        after = np.asarray(
            after_tensor.detach().cpu().tolist(), dtype=np.int64
        )
        truth = np.asarray(
            [int(row["registered_class_index"]) for row in query_records],
            dtype=np.int64,
        )
        scenario_predictions[scenario] = {
            "before": before,
            "after": after,
            "truth": truth,
        }
        scenario_resources[scenario] = {
            "resource": fitted.resource,
            "loss": _loss_summary(fitted.loss_trace),
        }
    prediction_path = cell_root / "predictions.npz"
    with prediction_path.open("xb") as handle:
        np.savez(
            handle,
            **{
                f"{scenario}_{name}": value
                for scenario, payload in scenario_predictions.items()
                for name, value in payload.items()
            },
        )
    prediction_sha = _sha256_file(prediction_path)
    rows = []
    for scenario, payload in scenario_predictions.items():
        rows.append(
            {
                "status": STATUS,
                "row_id": str(cell["cell_id"]),
                "method": str(cell["method"]),
                "receiver_label": str(cell["receiver"]),
                "seed": int(cell["seed"]),
                "new_class_count": int(cell["new_class_count"]),
                "k_shot": int(cell["k_shot"]),
                "scenario": scenario,
                "old_class_channel_policy": "unchanged_v7_leo_weak",
                "new_class_channel_policy": "matched_raw_no_leo",
                "prediction_artifact_sha256": prediction_sha,
                **score_predictions(
                    before=payload["before"],
                    after=payload["after"],
                    truth=payload["truth"],
                    old_count=len(old_labels),
                ),
            }
        )
    _write_json_new(cell_root / "formal_rows.json", rows)
    receipt = {
        "schema": "cvs.adv3b02.paper_full_newclass_no_leo_diagnostic_cell.v1",
        "status": STATUS,
        "cell_id": str(cell["cell_id"]),
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": prediction_sha,
        "rows": rows,
        "scenario_resources": scenario_resources,
        "query_rows_used_for_training": 0,
        "predictions_finalized_before_scoring": True,
    }
    _write_json_new(cell_root / "cell_receipt.json", receipt)
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    cells = list(plan["cells"])
    if args.smoke_only:
        wanted = set(plan["smoke_cell_ids"])
        cells = [cell for cell in cells if cell["cell_id"] in wanted]
    cells = [
        cell
        for index, cell in enumerate(cells)
        if index % int(args.shard_count) == int(args.shard_index)
    ]
    packages = {value["package_id"]: value for value in plan["packages"]}
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        str(args.device) if torch.cuda.is_available() else "cpu"
    )
    backbone, feature_fn, base_state, base_receipt = _load_model(
        Path(args.checkpoint).resolve(strict=True),
        Path(args.base_state).resolve(strict=True),
        device=device,
    )
    manytx = load_wisig_compact_pkl(str(Path(args.manytx_pkl).resolve(strict=True)))
    cache_memo: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    mixed_memo: dict[str, dict[str, np.ndarray]] = {}
    raw_lookup_memo: dict[str, dict[tuple[str, str, str, str, str], np.ndarray]] = {}
    replacement_audits: dict[str, Any] = {}
    completed = []
    for cell in cells:
        package = packages[str(cell["package_id"])]
        cache_key = str(package["target_cache_set"])
        if cache_key not in cache_memo:
            cache_path = Path(cache_key).resolve(strict=True)
            caches = _load_cache_set(cache_path)
            receiver = str(package["receiver"])
            if receiver not in raw_lookup_memo:
                lookup, lookup_audit = build_raw_new_lookup(
                    manytx,
                    receiver=receiver,
                    cache_arrays=caches[SCENARIOS[0]],
                )
                raw_lookup_memo[receiver] = lookup
                replacement_audits[f"lookup:{receiver}"] = lookup_audit
            mixed = {}
            for scenario in SCENARIOS:
                mixed[scenario], audit = replace_new_class_iq(
                    caches[scenario], raw_lookup_memo[receiver]
                )
                replacement_audits[f"{cache_key}:{scenario}"] = audit
            cache_memo[cache_key] = caches
            mixed_memo[cache_key] = mixed
        receipt = run_cell(
            cell,
            package,
            cache_by_scenario=cache_memo[cache_key],
            mixed_iq_by_scenario=mixed_memo[cache_key],
            backbone=backbone,
            feature_fn=feature_fn,
            base_state=base_state,
            device=device,
            output_root=output_root,
        )
        completed.append(receipt["cell_id"])
    result = {
        "schema": "cvs.adv3b02.paper_full_newclass_no_leo_diagnostic_shard.v1",
        "status": STATUS,
        "plan": str(plan_path),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "completed_cell_count": len(completed),
        "completed_cell_ids": completed,
        "base_state_receipt": base_receipt,
        "channel_replacement_audits": replacement_audits,
        "base_state_authority": (
            "official_repo_full_source_8400"
            if base_receipt.get("schema")
            == "cvs.adv3b02.official_repo_base_state.v2"
            else "legacy_v7_base_state"
        ),
        "formal_cvs_claim_allowed": False,
    }
    receipt_path = output_root / f"shard_{int(args.shard_index)}_receipt.json"
    if not receipt_path.exists():
        _write_json_new(receipt_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manytx-pkl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index/count are invalid")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
