#!/usr/bin/env python3
"""Build the formal D103-R1 0.07/0.63/0.30 source feature split.

The command consumes a newly built ``source_train`` LEO-weak cache covering
the complete source pool.  One preregistered weak observation is selected per
physical sample, the frozen dual runtime supplies z_dom, and the exact frozen
checkpoint supplies pre_relu.  The complete feature pool exists only in
memory; publication is delegated to the structurally separated source archive
builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for _value in (str(REPO_ROOT), str(CODE_ROOT)):
    while _value in sys.path:
        sys.path.remove(_value)
for _value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, _value)

from cvsrffi.checkpoint_loading import (  # noqa: E402
    build_exact_ssdg_model_from_checkpoint,
)
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMAS,
    LEO_WEAK_CACHE_SET_SCHEMAS,
    load_verified_leo_weak_cache_set,
)
from cvsrffi.rxid_metabias4_source_archive import (  # noqa: E402
    POOL_MEMBERS,
    publish_source_split_archives,
)
from scripts.export_adv3b02_dual_feature_torchscript import (  # noqa: E402
    _seal_graph_executor_optimize_false,
)
from scripts.export_phase1_jp4_tap_archive import (  # noqa: E402
    _forward_taps,
    _load_exact_sha_bound_checkpoint,
)
from scripts.export_phase1_singleobs_dual_feature_archive import (  # noqa: E402
    _class_registry,
    _forward_once_per_selected_iq_batch,
    _load_runtime_closure,
    _load_selection_salt,
    _resolve_device,
    _select_verified_observations,
    _sha256,
    _sha256_file,
)


def _resolve_cache_member(
    manifest_path: Path,
    raw_path: str,
    expected_sha256: str,
    scenario: str,
) -> str:
    expected = _sha256(expected_sha256, name=f"cache member {scenario}")
    raw = Path(str(raw_path))
    path = (raw if raw.is_absolute() else manifest_path.parent / raw).resolve()
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
        raise ValueError(f"cache member path/SHA drift: {scenario}")
    return expected


def export_d103_r1_source_splits(
    *,
    cache_set_path: str | Path,
    cache_set_sha256: str,
    selection_salt_receipt_path: str | Path,
    selection_salt_receipt_sha256: str,
    runtime_path: str | Path,
    runtime_sha256: str,
    runtime_role: str,
    export_receipt_path: str | Path,
    export_receipt_sha256: str,
    parity_receipt_path: str | Path,
    parity_receipt_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    class_ids: tuple[str, ...] | list[str],
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> dict[str, Any]:
    runtime_device = _resolve_device(device)
    execution_contract = _seal_graph_executor_optimize_false(runtime_device)
    runtime = _load_runtime_closure(
        runtime_path=runtime_path,
        runtime_sha256=runtime_sha256,
        runtime_role=runtime_role,
        export_receipt_path=export_receipt_path,
        export_receipt_sha256=export_receipt_sha256,
        parity_receipt_path=parity_receipt_path,
        parity_receipt_sha256=parity_receipt_sha256,
        device=runtime_device,
        execution_contract=execution_contract,
    )
    checkpoint_sha = _sha256(checkpoint_sha256, name="checkpoint")
    if runtime["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("dual runtime/checkpoint lineage drift")
    checkpoint_file = Path(checkpoint_path).resolve()
    if (
        not checkpoint_file.is_file()
        or checkpoint_file.is_symlink()
        or _sha256_file(checkpoint_file) != checkpoint_sha
    ):
        raise ValueError("checkpoint path/SHA drift")
    salt = _load_selection_salt(
        selection_salt_receipt_path,
        selection_salt_receipt_sha256,
        checkpoint_sha=checkpoint_sha,
    )

    cache_path = Path(cache_set_path).resolve()
    expected_cache = _sha256(cache_set_sha256, name="cache set")
    if (
        not cache_path.is_file()
        or cache_path.is_symlink()
        or _sha256_file(cache_path) != expected_cache
    ):
        raise ValueError("source_train cache-set path/SHA drift")
    arrays_by_scenario, cache_payload, cache_audit = load_verified_leo_weak_cache_set(
        cache_path,
        expected_scope="source_train",
        allowed_roles={"source"},
        accepted_outer_schemas=LEO_WEAK_CACHE_SET_SCHEMAS,
        accepted_inner_schemas=LEO_WEAK_CACHE_SCHEMAS,
    )
    if (
        cache_payload.get("cache_scope") != "source_train"
        or tuple(arrays_by_scenario) != FORMAL_LEO_WEAK_SCENARIOS
        or _sha256_file(cache_path) != expected_cache
    ):
        raise ValueError("source_train cache scope/scenario closure drift")
    scenario_paths = cache_payload.get("cache_npz_by_scenario")
    scenario_hashes = cache_payload.get("cache_sha256_by_scenario")
    if (
        not isinstance(scenario_paths, Mapping)
        or not isinstance(scenario_hashes, Mapping)
        or tuple(scenario_paths) != FORMAL_LEO_WEAK_SCENARIOS
        or tuple(scenario_hashes) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise ValueError("source_train cache member mapping drift")
    cache_member_hashes = {
        scenario: _resolve_cache_member(
            cache_path,
            str(scenario_paths[scenario]),
            str(scenario_hashes[scenario]),
            scenario,
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    metadata, selected_iq = _select_verified_observations(
        arrays_by_scenario, salt["selection_salt_sha256"]
    )
    if selected_iq.shape[1:] != (2, int(runtime["input_len"])):
        raise ValueError("selected received-IQ shape drift")
    registry = _class_registry(class_ids, logit_width=runtime["tx_classes"])
    if set(metadata["labels"].astype(str).tolist()) != set(registry):
        raise ValueError("source pool/class registry mismatch")

    dual_zid, z_dom, _tx_logits, dual_calls = (
        _forward_once_per_selected_iq_batch(
            runtime,
            selected_iq,
            device=runtime_device,
            batch_size=batch_size,
        )
    )
    checkpoint, checkpoint_load_audit = _load_exact_sha_bound_checkpoint(
        checkpoint_file, checkpoint_sha
    )
    model, rebuild_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(runtime["input_len"]), device=runtime_device
    )
    model.to(runtime_device).eval()
    eager_zid, _hidden, pre_relu, eager_calls = _forward_taps(
        model,
        selected_iq,
        device=runtime_device,
        batch_size=batch_size,
    )
    maximum = float(
        np.max(
            np.abs(
                dual_zid.astype(np.float64, copy=False)
                - eager_zid.astype(np.float64, copy=False)
            )
        )
    )
    if not np.isfinite(maximum) or maximum > 1.0e-5:
        raise ValueError(f"dual/eager z_id parity failed: max_abs={maximum}")
    pool = {
        "z_id": eager_zid,
        "z_dom": z_dom,
        "pre_relu": pre_relu,
        "labels": metadata["labels"],
        "receiver_ids": metadata["receiver_ids"],
        "day_ids": metadata["day_ids"],
        "physical_ids": metadata["physical_ids"],
        "scenario_names": metadata["scenario_names"],
        "observation_ids": metadata["observation_ids"],
        "class_ids": np.asarray(registry, dtype=np.str_),
    }
    if tuple(pool) != POOL_MEMBERS:
        raise ValueError("source pool construction order drift")
    del selected_iq, model, checkpoint, _hidden, _tx_logits
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()

    upstream_audit = {
        "dual_calls": dual_calls,
        "eager_calls": eager_calls,
        "dual_eager_z_id_max_abs": maximum,
        "cache_member_sha256": cache_member_hashes,
        "cache_loader_audit": cache_audit,
        "checkpoint_load_audit": checkpoint_load_audit,
        "checkpoint_rebuild_audit": rebuild_audit,
        "complete_feature_pool_persisted": False,
        "received_iq_persisted": False,
        "target_access": False,
        "formal_query_access": False,
    }
    result = publish_source_split_archives(
        pool,
        output_dir=output_dir,
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=runtime["sha256"],
        cache_set_sha256=expected_cache,
        selection_salt_receipt_sha256=salt["sha256"],
        upstream_audit=upstream_audit,
    )
    result["runtime_audit"] = upstream_audit
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "cache-set",
        "cache-set-sha256",
        "selection-salt-receipt",
        "selection-salt-receipt-sha256",
        "runtime",
        "runtime-sha256",
        "runtime-role",
        "export-receipt",
        "export-receipt-sha256",
        "parity-receipt",
        "parity-receipt-sha256",
        "checkpoint",
        "checkpoint-sha256",
        "class-ids",
        "output-dir",
    ):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = vars(args)
    renames = {
        "cache_set": "cache_set_path",
        "selection_salt_receipt": "selection_salt_receipt_path",
        "runtime": "runtime_path",
        "export_receipt": "export_receipt_path",
        "parity_receipt": "parity_receipt_path",
        "checkpoint": "checkpoint_path",
    }
    for old, new in renames.items():
        values[new] = values.pop(old)
    values["class_ids"] = tuple(
        item.strip() for item in values["class_ids"].split(",") if item.strip()
    )
    result = export_d103_r1_source_splits(**values)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
