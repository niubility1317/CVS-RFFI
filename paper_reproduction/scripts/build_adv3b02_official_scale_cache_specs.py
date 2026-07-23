#!/usr/bin/env python3
"""Generate immutable new25 external-comparison LEO cache build specifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713101, 713102, 713103, 713104, 713105)
OLD = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
NEW25 = (
    "1-16", "1-18", "18-10", "14-11", "8-3",
    "18-8", "10-10", "16-19", "20-12", "4-10",
    "13-14", "2-5", "1-8", "19-13", "19-9",
    "3-8", "19-8", "11-19", "2-16", "19-6",
    "13-19", "18-14", "20-4", "20-16", "11-10",
)


def _safe(value: str) -> str:
    return value.replace("-", "_")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    remote_root = PurePosixPath(str(args.remote_cache_root))
    reference_root = PurePosixPath(str(args.reference_cache_root))
    parity_root = PurePosixPath(str(args.remote_parity_root))
    specs = []
    commands = []
    parity_commands = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            leaf = remote_root / f"rx_{_safe(receiver)}" / f"seed_{seed}"
            spec = {
                "schema": "cvs_leo_weak_iq_cache_build_spec_v1",
                "cache_set_id": (
                    f"{args.experiment_id}_{_safe(receiver)}_seed_{seed}_new25"
                ),
                "cache_scope": "external_comparison_registered",
                "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
                "clean_sample_access": False,
                "clean_derived_signal_access": False,
                "star_ground_channel_impl": "simplified_leo_residual",
                "role_specs": [
                    {
                        "role": "target_old",
                        "pkl": str(args.manytx_pkl),
                        "tx_ids": ",".join(OLD),
                        "rxs": receiver,
                        "days": "0",
                        "max_samples_per_tx": 50,
                    },
                    {
                        "role": "target_new",
                        "pkl": str(args.manytx_pkl),
                        "tx_ids": ",".join(NEW25),
                        "rxs": receiver,
                        "days": "0",
                        "max_samples_per_tx": 50,
                    },
                ],
                "dataset_seed": seed,
                "satellite_seed_by_scenario": {
                    scenario: seed * 10 + index
                    for index, scenario in enumerate(SCENARIOS)
                },
                "out_npz_by_scenario": {
                    scenario: str(leaf / f"{scenario}.npz")
                    for scenario in SCENARIOS
                },
                "out_manifest": str(leaf / "cache_set.json"),
                "batch_size": 256,
                "wisig_out_len": 256,
                "wisig_equalized": "1",
                "wisig_domain": "rx_day",
            }
            relative = (
                Path("cache_specs")
                / f"rx_{_safe(receiver)}"
                / f"seed_{seed}.json"
            )
            _write_new(output_dir / relative, spec)
            specs.append(
                {
                    "receiver": receiver,
                    "seed": seed,
                    "relative_path": relative.as_posix(),
                    "content_sha256": _canonical_sha256(spec),
                    "cache_set_manifest": str(leaf / "cache_set.json"),
                }
            )
            commands.append(
                [
                    "python",
                    "code/scripts/build_cvs_leo_weak_iq_cache.py",
                    "--spec",
                    str(PurePosixPath(args.remote_plan_root) / relative.as_posix()),
                    "--device",
                    "cuda:0",
                ]
            )
            parity_commands.append(
                [
                    "python",
                    (
                        "paper_reproduction/scripts/"
                        "verify_adv3b02_official_scale_cache_parity.py"
                    ),
                    "--reference-cache-set",
                    str(
                        reference_root
                        / f"rx_{_safe(receiver)}"
                        / f"seed_{seed}"
                        / "cache_set.json"
                    ),
                    "--expanded-cache-set",
                    str(leaf / "cache_set.json"),
                    "--reference-scope",
                    "stage2_registered",
                    "--expanded-scope",
                    "external_comparison_registered",
                    "--preserved-class-labels",
                    ",".join((*OLD, *NEW25[:20])),
                    "--output",
                    str(
                        parity_root
                        / f"rx_{_safe(receiver)}"
                        / f"seed_{seed}.json"
                    ),
                ]
            )
    manifest = {
        "schema": "cvs.adv3b02.official_scale_cache_specs.v1",
        "experiment_id": str(args.experiment_id),
        "status": "LOCAL_GENERATED_NOT_EXECUTED",
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "old_class_labels": list(OLD),
        "new25_class_labels": list(NEW25),
        "paper_increment_counts": [25, 10, 5, 3],
        "specs": specs,
        "commands": commands,
        "parity_commands": parity_commands,
        "required_parity_gate": (
            "old6+first20 sample_ids and post_channel_iq_sha256 must match "
            "the prior official-repo cache for every receiver/seed/scenario"
        ),
    }
    _write_new(output_dir / "cache_specs_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--manytx-pkl", required=True)
    parser.add_argument("--remote-plan-root", required=True)
    parser.add_argument("--remote-cache-root", required=True)
    parser.add_argument("--reference-cache-root", required=True)
    parser.add_argument("--remote-parity-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
