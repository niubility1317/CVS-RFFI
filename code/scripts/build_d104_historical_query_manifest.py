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
EXPECTED_TAP_SHA256 = (
    "c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1"
)
EXPECTED_DUAL_SHA256 = (
    "dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0"
)
EXPECTED_SOURCE_PARTITION_CODE_SHA256 = (
    "dcfe6f0c8d0c49b06d7482185329a389eba3f14f542790d6d3577d8b48f3e764"
)
EXPECTED_HELD_PACKAGER_CODE_SHA256 = (
    "571ddb448cd44131a05ff6187fbb66ad20ae115af57412447e8a92c08c39cc1e"
)
EXPECTED_SOURCE_VAL_ROOT_SHA256 = (
    "a88e0ca27b4b5835822b5e0c5437e01f9ededc8372cc47b52575839ba023f8bb"
)
EXPECTED_SUPPORT_ROOT_SHA256 = (
    "2b4f8cd98b1e33e4f7cc3451f321ffe63c1c5a011bfd49168e2b51f8860c5e7a"
)
EXPECTED_RECEIVERS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1", "2-19")
EXPECTED_CLASSES = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
EXPECTED_PACKAGES = (
    ("1-1", 354, "60c028eb6a7d1f23936b66308020406ed9e8c1e565a2b7ac45e1cb909e1bbf39", "4293c19144aab16df048efa8435a3f59ac7e9bcc4f5940a91e088f8ad4c6d7dd"),
    ("1-19", 360, "5e3ae2cf7b5a56a53ca3564823ab3133ea12258e0dbd6347822f30961b6f6379", "a1f352336e890a3ef8ddc39358187aa28ca78785f4c83a5f745ae56cc13a2ba1"),
    ("14-7", 353, "943dcca004a06bfc15aa38ddc4911bd36dde3f01817523e47de4aef11bd72fdb", "cbd444ef0c943c97acd8831193c82dd5fc87be557195abbaf892315f77b8fe2d"),
    ("18-2", 349, "d6ad0d0dcfdbcd447df51382548cef45b952b83aecb81e3e099154386e21d627", "3896a87bc525f2c9badcf7a408558fd177a41cc73d18070673de526b2f3eaeb5"),
    ("19-2", 358, "ffc52864425abc5404c3194a7a34ee77e4c7ac240f05d94ed7314fc15a28f6eb", "fab024500ebc61b70a10b50df7906649f3a7d0bf1164a7ffcd63662a25047a17"),
    ("2-1", 364, "949f1ba96550d3eacd73c86649418f3413db6a404ce6c1ccbd868dae65145739", "db5f7094d15cbf056aae75c072034852766975c7bfe47efea00eca642e37fc7f"),
    ("2-19", 340, "6394d09bbb3f9180fe976ca3689a8a6b857504ad01f1b9a4c867c70d52f83f5e", "d239c7bcb039ca47e675551ed921b3facba1bf86b8a62a97550a102bae025c9d"),
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


def _write_json_new(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _validate_frozen_files(
    tap_path: Path,
    dual_path: Path,
    code_paths: dict[str, Path],
) -> None:
    if (
        _sha256(tap_path) != EXPECTED_TAP_SHA256
        or _sha256(dual_path) != EXPECTED_DUAL_SHA256
        or _sha256(code_paths["source_partition"])
        != EXPECTED_SOURCE_PARTITION_CODE_SHA256
        or _sha256(code_paths["held_packager"])
        != EXPECTED_HELD_PACKAGER_CODE_SHA256
    ):
        raise ValueError("historical exclusion input/code identity drift")


def _validate_source_registry(
    legacy_receipt: dict[str, object],
    receivers: tuple[str, ...],
    classes: tuple[str, ...],
) -> None:
    roots = legacy_receipt.get("physical_id_roots")
    if (
        receivers != EXPECTED_RECEIVERS
        or classes != EXPECTED_CLASSES
        or legacy_receipt.get("counts")
        != {"L_s": 588, "U_s": 5292, "source_val": 2520}
        or not isinstance(roots, dict)
        or roots.get("source_val") != EXPECTED_SOURCE_VAL_ROOT_SHA256
    ):
        raise ValueError("historical source-val registry/root drift")


def _validate_reconstruction(
    ordered_query: list[str],
    ordered_support: list[str],
    package_rows: list[dict[str, object]],
) -> None:
    package_identity = tuple(
        (
            row["held_receiver"],
            row["query_count"],
            row["query_physical_id_root_sha256"],
            row["support_physical_id_root_sha256"],
        )
        for row in package_rows
    )
    if (
        len(ordered_query) != HISTORICAL_QUERY_COUNT
        or len(set(ordered_query)) != HISTORICAL_QUERY_COUNT
        or len(ordered_support) != 42
        or len(set(ordered_support)) != 42
        or set(ordered_query).intersection(ordered_support)
        or canonical_sha256(ordered_query)
        != HISTORICAL_QUERY_CANONICAL_ROOT_SHA256
        or canonical_sha256(ordered_support) != EXPECTED_SUPPORT_ROOT_SHA256
        or len(package_rows) != 7
        or any(row.get("K") != 1 or row.get("support_count") != 6 for row in package_rows)
        or package_identity != EXPECTED_PACKAGES
    ):
        raise ValueError("historical exclusion manifest reconstruction drift")


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
    code_paths = {
        "builder_script": Path(__file__).resolve(),
        "source_partition": ROOT / "cvsrffi" / "rxid_metabias4_source_archive.py",
        "held_packager": ROOT / "cvsrffi" / "rxid_metabias4_held_execution.py",
    }
    _validate_frozen_files(tap_path, dual_path, code_paths)

    tap = _load(tap_path)
    dual = _load(dual_path)
    if len(tap.get("z_id", ())) != 8400 or len(dual.get("z_dom", ())) != 8400:
        raise ValueError("historical exclusion input row-count drift")
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
    _validate_source_registry(legacy_receipt, receivers, classes)
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
    _validate_reconstruction(ordered_query, ordered_support, package_rows)
    query_root = canonical_sha256(ordered_query)

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
        "source_val_labels_used_for_package_reconstruction": True,
        "query_truth_passed_to_predictor": False,
        "query_truth_used_for_scoring": False,
        "query_predictions_computed": False,
        "performance_computed": False,
        "new_source_held_features_read": False,
        "target_access": False,
        "n607_run": False,
    }
    manifest["manifest_content_root_sha256"] = canonical_sha256(manifest)
    _write_json_new(output, manifest)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
