#!/usr/bin/env python3
"""Build the immutable D104 historical-query exclusion manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.phase1_rb_metabias4_bundle import (  # noqa: E402
    merge_verified_phase1_tap_and_dual_archives,
)
from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    build_receiver_package_indices,
    canonical_sha256,
)
from cvsrffi.rxid_metabias4_source_archive import (  # noqa: E402
    SCHEMA as D103_SOURCE_SPLIT_SCHEMA,
    partition_source_pool,
)
CANDIDATE_ID = "D104-R1-ANGQ-RXID-MB4"
SPLIT_ID = "d104_source_seed104713_v2"
HISTORICAL_QUERY_COUNT = 2478
HISTORICAL_QUERY_CANONICAL_ROOT_SHA256 = (
    "7870604d8ddba8268ba127065d4eaf1142931660d95411c9633c2ffa59d6b558"
)
WITHDRAWN_UNREPRODUCIBLE_LEGACY_ROOT = (
    "036456779eea6594f2330f2e9a96cceda580088b0d451982198e3056f762854d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tap_path = args.tap_archive.resolve(strict=True)
    dual_path = args.dual_archive.resolve(strict=True)
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable exclusion manifest exists: {output}")

    tap = _load(tap_path)
    dual = _load(dual_path)
    merged = merge_verified_phase1_tap_and_dual_archives(tap, dual)
    source_pool = {
        "z_id": np.asarray(tap["z_id"], dtype=np.float32),
        "z_dom": np.asarray(merged["z_dom"], dtype=np.float32),
        "pre_relu": np.asarray(merged["pre_relu"], dtype=np.float32),
        "labels": merged["labels"].astype(str),
        "receiver_ids": merged["receiver_ids"].astype(str),
        "day_ids": merged["day_ids"].astype(str),
        "physical_ids": merged["physical_ids"].astype(str),
        "scenario_names": tap["scenario_names"].astype(str),
        "observation_ids": tap["observation_ids"].astype(str),
        "class_ids": merged["class_ids"].astype(str),
    }
    _labeled, _unlabeled, scorer, legacy_receipt = partition_source_pool(source_pool)
    receivers = tuple(sorted(set(scorer["receiver_ids"].astype(str).tolist())))
    classes = tuple(sorted(set(scorer["labels"].astype(str).tolist())))
    query_ids: set[str] = set()
    support_ids: set[str] = set()
    package_rows = []
    for receiver in receivers:
        support, query = build_receiver_package_indices(
            scorer["receiver_ids"],
            scorer["labels"],
            scorer["physical_ids"],
            held_receiver=receiver,
            registered_classes=classes,
            k_shot=1,
        )
        local_support = sorted(
            scorer["physical_ids"][support].astype(str).tolist()
        )
        local_query = sorted(scorer["physical_ids"][query].astype(str).tolist())
        support_ids.update(local_support)
        query_ids.update(local_query)
        package_rows.append(
            {
                "held_receiver": receiver,
                "K": 1,
                "support_count": len(local_support),
                "query_count": len(local_query),
                "support_physical_id_root_sha256": canonical_sha256(local_support),
                "query_physical_id_root_sha256": canonical_sha256(local_query),
            }
        )
    ordered_query = sorted(query_ids)
    ordered_support = sorted(support_ids)
    query_root = canonical_sha256(ordered_query)
    if (
        len(ordered_query) != HISTORICAL_QUERY_COUNT
        or len(ordered_support) != 42
        or set(ordered_query).intersection(ordered_support)
        or query_root != HISTORICAL_QUERY_CANONICAL_ROOT_SHA256
    ):
        raise ValueError("historical exclusion manifest reconstruction drift")

    code_paths = {
        "builder_script": Path(__file__).resolve(),
        "source_partition": ROOT / "cvsrffi" / "rxid_metabias4_source_archive.py",
        "held_packager": ROOT / "cvsrffi" / "rxid_metabias4_held_execution.py",
    }
    manifest = {
        "schema": "cvs.d104_r1.historical_query_exclusion_manifest.v2",
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "status": "ACTIVE_REPRODUCIBLE_EXCLUSION_CONTROL",
        "canonical_encoding": (
            "json.dumps(sorted_list,ensure_ascii=False,sort_keys=True,"
            "separators=(',',':'),allow_nan=False)+UTF8; no trailing newline"
        ),
        "active_query_physical_id_root_sha256": query_root,
        "withdrawn_unreproducible_legacy_root_sha256": (
            WITHDRAWN_UNREPRODUCIBLE_LEGACY_ROOT
        ),
        "withdrawn_root_status": "WITHDRAWN_UNREPRODUCIBLE_LEGACY_ROOT",
        "query_physical_id_count": len(ordered_query),
        "query_physical_ids_sorted": ordered_query,
        "support_physical_id_count": len(ordered_support),
        "support_physical_ids_sorted": ordered_support,
        "support_query_intersection_count": 0,
        "source_val_physical_id_root_sha256": (
            legacy_receipt["physical_id_roots"]["source_val"]
        ),
        "legacy_source_split_schema": D103_SOURCE_SPLIT_SCHEMA,
        "legacy_source_split_counts": legacy_receipt["counts"],
        "derivation": (
            "union of K=1 query physical IDs from all seven D103-R2 "
            "source-val receiver packages; support is the corresponding union"
        ),
        "receiver_ids": list(receivers),
        "registered_classes": list(classes),
        "K": 1,
        "packages": package_rows,
        "input_archives": {
            "tap": {"sha256": _sha256(tap_path), "row_count": len(tap["z_id"])},
            "dual": {"sha256": _sha256(dual_path), "row_count": len(dual["z_dom"])},
        },
        "derivation_code": {
            name: {
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for name, path in code_paths.items()
        },
        "query_truth_labels_read": False,
        "query_predictions_computed": False,
        "performance_computed": False,
        "new_source_held_features_read": False,
        "target_access": False,
        "n607_run": False,
    }
    manifest["manifest_content_root_sha256"] = canonical_sha256(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
