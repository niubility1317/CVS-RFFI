#!/usr/bin/env python
"""Audit and seal hashes for an already-built reusable LEO_weak cache matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_cache_build_matrix import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    validate_registered_cache_coverage,
)


FORBIDDEN_MEMBER_TOKENS = ("clean", "raw", "source")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 30:
        raise ValueError("exact reusable cache manifest must contain 30 cells")

    output = Path(args.output).resolve()
    cell_output = output / "cells"
    cell_output.mkdir(parents=True, exist_ok=True)
    hashes: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    reference_new_tx: list[str] | None = None

    for cell in cells:
        cell_id = str(cell["cell_id"])
        receiver = str(cell["receiver"])
        cache_root = Path(str(cell["cache_output_root"])).resolve()
        cache_set_path = cache_root / "cache_set.json"
        audit = validate_registered_cache_coverage(
            cache_set_path,
            expected_receiver=receiver,
        )
        if audit.get("coverage_pass") is not True:
            raise ValueError(f"coverage failed for {cell_id}")
        _write_json(cell_output / f"{cell_id}.json", audit)
        audits.append(audit)

        cache_set = json.loads(cache_set_path.read_text(encoding="utf-8"))
        paths = [("cache_set", cache_set_path)]
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            cache_path = cache_root / cache_set["cache_npz_by_scenario"][scenario]
            with zipfile.ZipFile(cache_path, "r") as archive:
                member_names = archive.namelist()
            if any(
                token in member.lower()
                for member in member_names
                for token in FORBIDDEN_MEMBER_TOKENS
            ):
                raise ValueError(f"forbidden clean/raw/source member in {cell_id}")
            paths.append((scenario, cache_path))
        for kind, path in paths:
            hashes.append(
                {
                    "cell_id": cell_id,
                    "kind": kind,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )

        spec_path = manifest_path.parent / str(cell["spec_path"])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        new_roles = [row for row in spec["role_specs"] if row["role"] == "target_new"]
        new_tx = new_roles[0]["tx_ids"].split(",")
        if reference_new_tx is None:
            reference_new_tx = new_tx
        elif new_tx != reference_new_tx:
            raise ValueError("nested target-new registry drifts across cells")

    physical_count = sum(
        int(audit["row_count_per_scenario"]) * int(audit["scenario_count"])
        for audit in audits
    )
    summary = {
        "schema": "cvs.phase2.reusable_leo_weak_data_asset.v1",
        "cell_count": len(audits),
        "receiver_count": len({str(cell["receiver"]) for cell in cells}),
        "seed_count": len({int(cell["seed"]) for cell in cells}),
        "scenario_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "physical_sample_count": physical_count,
        "physical_sample_observation_count": physical_count,
        "single_observation_compliant": all(
            audit["cross_scenario_physical_sample_overlap_count"] == 0
            for audit in audits
        ),
        "coverage_pass": all(audit["coverage_pass"] is True for audit in audits),
        "exact_rows_per_role_tx_receiver": sorted(
            {int(audit["exact_rows_per_role_tx_receiver"]) for audit in audits}
        ),
        "forbidden_clean_raw_source_npz_member_count": 0,
        "artifact_file_count": len(hashes),
        "formal_metric_claim_allowed": False,
    }
    reuse_slices = {
        "schema": "cvs.phase2.reusable_leo_weak_slice_registry.v1",
        "new_tx_registry": reference_new_tx,
        "new_class_prefixes": {str(n): reference_new_tx[:n] for n in (5, 10, 20)},
        "support_rank_prefixes": {str(k): list(range(k)) for k in (1, 5, 10, 20)},
        "query_ranks": list(range(20, 40)),
        "additional_leo_channel_generation": False,
        "post_reception_view_counts_as_additional_sample": False,
    }
    _write_json(output / "phase2_data_asset_summary.json", summary)
    _write_json(output / "phase2_data_asset_hashes.json", hashes)
    _write_json(output / "reuse_slices.json", reuse_slices)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
