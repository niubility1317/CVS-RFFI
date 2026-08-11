"""Frozen Hard12 development matrix for the D92-BE four-arm experiment."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from cvsrffi.stage2_d92_be_slim import D92_BE_ARMS


CONTEXT_SHA256 = "067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f"
CANONICAL_SELECTION_SHA256 = (
    "95d94d586f5084d4982d67ec6402c4244f80e818ef3f95a5a03771085a6885a4"
)
LEGACY_UNREPRODUCIBLE_SELECTION_SHA256 = (
    "26ca470a4cc79d13498493863e6958c3fc5c82af1b3dbecd06cf6277d0a650e4"
)
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARM_ORDER = ("FULL", "B0", "E0", "B0E0")
OUTER_PATTERN = re.compile(
    r"^rx_(?P<receiver>[0-9_]+)__seed_(?P<seed>[0-9]+)"
    r"__k_(?P<k>[0-9]+)__new_(?P<new>[0-9]+)$"
)


SELECTION_PAYLOAD: dict[str, Any] = {
    "schema": "cvs.phase2.d92_be_hard12.canonical_manifest.v1",
    "selection_id": "Hard12-v1",
    "protocol_schema": "p2_min_v1",
    "claim_scope": "DEVELOPMENT_ONLY_COVERAGE_CONSTRAINED_STRESS_SCREEN",
    "input_sha256": {
        "d92_retry2_row_metrics_csv": "bc8070cd9235ab41eda5bafd2ec66e9afad48b6466d2066508d0bab46980fa62",
        "next_r5_r11_score_json": "fa2344ae037e4ab5dfec6fea9bb0f534c7d5c9cdeb3596797bdc403b3c9fcc23",
        "target125_context_json": CONTEXT_SHA256,
    },
    "r5_state": "DA0_REG1",
    "scenes": list(SCENES),
    "scoring": {
        "outer_population": 125,
        "tie_rule": "average_ascending_rank",
        "pi_down": "(125-rank_avg)/124",
        "pi_up": "(rank_avg-1)/124",
        "d92_metrics_low_is_hard": [
            "H_old_new",
            "c_old_acc",
            "c_old_floor",
            "seen_new_acc",
        ],
        "d92_metrics_high_is_hard": ["average_forgetting"],
        "r5_metrics_low_is_hard": [
            "H_old_new",
            "old_balanced_accuracy",
            "old_floor",
            "seen_new_acc",
        ],
        "r5_scene_aggregation": "equal_mean_over_three_scenes",
        "weights": {"d92": 0.5, "r5": 0.5},
    },
    "selection": {
        "outer_count": 12,
        "receiver_count_range": [2, 3],
        "seed_count_range": [2, 3],
        "slice_count_rules": {
            "K1_new20": [2, 2],
            "K5_new20": [2, 3],
            "K10_new5": [2, 3],
            "K10_new10": [2, 3],
            "K10_new20": [2, 3],
        },
        "sentinel": "rx_3_19__seed_713104__k_1__new_20",
        "optimization": "maximize_sum_hard",
        "tie_break": "lexicographic_receiver_seed_k_new_plus_1e-9_perturbation",
    },
    "outer_rows": [
        {
            "hard_score": "0.922278225806",
            "outer_key": "rx_20_1__seed_713103__k_1__new_20",
            "role": "liveness",
        },
        {
            "hard_score": "0.574294354839",
            "outer_key": "rx_20_1__seed_713105__k_10__new_20",
            "role": "performance",
        },
        {
            "hard_score": "0.726108870968",
            "outer_key": "rx_20_1__seed_713106__k_5__new_20",
            "role": "performance",
        },
        {
            "hard_score": "0.632358870968",
            "outer_key": "rx_3_19__seed_713103__k_10__new_10",
            "role": "performance",
        },
        {
            "hard_score": "0.962600806452",
            "outer_key": "rx_3_19__seed_713104__k_1__new_20",
            "role": "liveness",
        },
        {
            "hard_score": "0.778830645161",
            "outer_key": "rx_3_19__seed_713106__k_10__new_20",
            "role": "performance",
        },
        {
            "hard_score": "0.509979838710",
            "outer_key": "rx_7_14__seed_713102__k_10__new_20",
            "role": "performance",
        },
        {
            "hard_score": "0.660584677419",
            "outer_key": "rx_7_14__seed_713102__k_5__new_20",
            "role": "performance",
        },
        {
            "hard_score": "0.168447580645",
            "outer_key": "rx_7_7__seed_713104__k_10__new_5",
            "role": "performance",
        },
        {
            "hard_score": "0.190725806452",
            "outer_key": "rx_7_7__seed_713105__k_10__new_5",
            "role": "performance",
        },
        {
            "hard_score": "0.423991935484",
            "outer_key": "rx_8_8__seed_713104__k_10__new_10",
            "role": "performance",
        },
        {
            "hard_score": "0.635786290323",
            "outer_key": "rx_8_8__seed_713105__k_5__new_20",
            "role": "performance",
        },
    ],
    "coverage": {
        "outer_count": 12,
        "scene_count": 36,
        "receiver_counts": {
            "20-1": 3,
            "3-19": 3,
            "7-14": 2,
            "7-7": 2,
            "8-8": 2,
        },
        "seed_counts": {
            "713102": 2,
            "713103": 2,
            "713104": 3,
            "713105": 3,
            "713106": 2,
        },
        "slice_counts": {
            "K1_new20": 2,
            "K5_new20": 3,
            "K10_new5": 2,
            "K10_new10": 2,
            "K10_new20": 3,
        },
    },
}
HARD12_ROWS = tuple(dict(row) for row in SELECTION_PAYLOAD["outer_rows"])


class D92BEHard12Error(ValueError):
    """Raised when the frozen Hard12 matrix cannot be reproduced exactly."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_selection_sha256() -> str:
    return hashlib.sha256(_canonical_bytes(SELECTION_PAYLOAD)).hexdigest()


def _parse_outer(outer_key: str) -> tuple[str, int, int, int]:
    match = OUTER_PATTERN.fullmatch(str(outer_key))
    if match is None:
        raise D92BEHard12Error(f"invalid Hard12 outer key: {outer_key}")
    return (
        match.group("receiver").replace("_", "-"),
        int(match.group("seed")),
        int(match.group("k")),
        int(match.group("new")),
    )


def _package_layout(source_job_root: Path, *, require_files: bool) -> dict[str, Any]:
    paths = {
        "before_enrollment": (
            source_job_root / "offline" / "predictor" / "before" / "enrollment_only",
            source_job_root / "offline" / "seals" / "before_enrollment.seal.json",
        ),
        "before_apply": (
            source_job_root / "offline" / "predictor" / "before" / "apply_only_staging",
            source_job_root / "apply_seals" / "before_apply.seal.json",
        ),
        "after_enrollment": (
            source_job_root / "offline" / "predictor" / "after" / "enrollment_only",
            source_job_root / "offline" / "seals" / "after_enrollment.seal.json",
        ),
        "after_apply": (
            source_job_root / "offline" / "predictor" / "after" / "apply_only_staging",
            source_job_root / "apply_seals" / "after_apply.seal.json",
        ),
    }
    result: dict[str, Any] = {}
    for name, (package_root, seal_path) in paths.items():
        if require_files and (
            not package_root.is_dir()
            or package_root.is_symlink()
            or not seal_path.is_file()
            or seal_path.is_symlink()
        ):
            raise D92BEHard12Error(f"sealed source package is missing: {name}")
        result[name] = {
            "package_root": str(package_root),
            "detached_seal_path": str(seal_path),
            "expected_seal_sha256": (
                _sha256_file(seal_path) if require_files else None
            ),
        }
    return result


def _coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    receiver = Counter(str(row["receiver"]) for row in rows)
    seed = Counter(str(row["seed"]) for row in rows)
    slices = Counter(
        f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}" for row in rows
    )
    return {
        "receiver_counts": dict(sorted(receiver.items())),
        "seed_counts": dict(sorted(seed.items())),
        "slice_counts": dict(sorted(slices.items())),
    }


def build_hard12_manifest(
    *,
    context_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    require_package_files: bool = True,
) -> dict[str, Any]:
    """Join the frozen selection to the sealed D92 source packages."""

    if canonical_selection_sha256() != CANONICAL_SELECTION_SHA256:
        raise D92BEHard12Error("canonical Hard12 selection payload drift")
    context_file = Path(context_path).resolve(strict=True)
    if _sha256_file(context_file) != CONTEXT_SHA256:
        raise D92BEHard12Error("target125 context SHA drift")
    context = json.loads(context_file.read_text(encoding="utf-8-sig"))
    if (
        context.get("schema")
        != "cvs.phase2.d108.cbrrc_smme.target125.input_context.v1"
        or context.get("protocol_schema") != "p2_min_v1"
        or not isinstance(context.get("rows"), list)
        or len(context["rows"]) != 125
    ):
        raise D92BEHard12Error("target125 context schema/count drift")
    context_rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in context["rows"]:
        key = (
            str(row["receiver"]),
            int(row["seed"]),
            int(row["active_k"]),
            int(row["new_count"]),
        )
        if key in context_rows:
            raise D92BEHard12Error("target125 context outer identity duplicated")
        context_rows[key] = dict(row)
    lock_file = Path(method_lock_path).resolve(strict=True)
    lock = json.loads(lock_file.read_text(encoding="utf-8-sig"))
    if (
        lock.get("protocol_schema") != "p2_min_v1"
        or lock.get("selection_sha256") != CANONICAL_SELECTION_SHA256
    ):
        raise D92BEHard12Error("D92-BE method lock drift")
    source_root = Path(str(context["identity"]["d92_output_root"]))
    output = Path(output_root)
    selected_rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for outer_index, frozen in enumerate(HARD12_ROWS):
        receiver, seed, k_shot, new_count = _parse_outer(frozen["outer_key"])
        context_key = (receiver, seed, k_shot, new_count)
        if context_key not in context_rows:
            raise D92BEHard12Error(f"Hard12/context join failed: {frozen['outer_key']}")
        selected_rows.append(
            {
                "outer_key": frozen["outer_key"],
                "outer_role": frozen["role"],
                "hard_score": frozen["hard_score"],
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "new_class_count": new_count,
            }
        )
        source_job_root = source_root / "jobs" / str(frozen["outer_key"])
        packages = _package_layout(
            source_job_root, require_files=bool(require_package_files)
        )
        truth_sidecar = source_job_root / "offline" / "scorer" / "truth_sidecar.json"
        if require_package_files and (
            not truth_sidecar.is_file() or truth_sidecar.is_symlink()
        ):
            raise D92BEHard12Error("source truth sidecar is missing")
        rotation = outer_index % len(ARM_ORDER)
        arm_order = ARM_ORDER[rotation:] + ARM_ORDER[:rotation]
        for arm_position, arm_id in enumerate(arm_order):
            arm = D92_BE_ARMS[arm_id]
            jobs.append(
                {
                    "index": len(jobs),
                    "outer_index": outer_index,
                    "arm_position": arm_position,
                    "planned_shard_index": outer_index % 8,
                    "job_id": f"{frozen['outer_key']}__arm_{arm_id.lower()}",
                    "outer_key": frozen["outer_key"],
                    "outer_role": frozen["role"],
                    "hard_score": frozen["hard_score"],
                    "receiver": receiver,
                    "seed": seed,
                    "k_shot": k_shot,
                    "new_class_count": new_count,
                    "arm_id": arm_id,
                    "candidate": arm.candidate_id,
                    "scenarios": list(SCENES),
                    "source_job_root": str(source_job_root),
                    "packages": packages,
                    "truth_sidecar": str(truth_sidecar),
                    "output_root": str(
                        output / "jobs" / str(frozen["outer_key"]) / arm_id
                    ),
                }
            )
    coverage = _coverage(selected_rows)
    expected_coverage = {
        name: dict(SELECTION_PAYLOAD["coverage"][name])
        for name in ("receiver_counts", "seed_counts", "slice_counts")
    }
    if coverage != expected_coverage or len(jobs) != 48:
        raise D92BEHard12Error("Hard12 coverage/job-count drift")
    identity = context["identity"]
    return {
        "schema": "cvs.phase2.d92_be_hard12.matrix.v1",
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "claim_scope": SELECTION_PAYLOAD["claim_scope"],
        "protocol_schema": "p2_min_v1",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "context_path": str(context_file),
        "context_sha256": CONTEXT_SHA256,
        "method_lock": str(lock_file),
        "method_lock_sha256": _sha256_file(lock_file),
        "ground_component_dir": identity["ground_component"]["directory"],
        "ground_manifest_sha256": identity["ground_component"]["manifest_sha256"],
        "output_root": str(output),
        "shard_count": 8,
        "outer_count": len(selected_rows),
        "performance_outer_count": sum(
            row["outer_role"] == "performance" for row in selected_rows
        ),
        "liveness_outer_count": sum(
            row["outer_role"] == "liveness" for row in selected_rows
        ),
        "job_count": len(jobs),
        "scene_arm_count": len(jobs) * len(SCENES),
        "coverage": coverage,
        "selected_rows": selected_rows,
        "jobs": jobs,
    }


__all__ = [
    "CANONICAL_SELECTION_SHA256",
    "D92BEHard12Error",
    "HARD12_ROWS",
    "LEGACY_UNREPRODUCIBLE_SELECTION_SHA256",
    "SELECTION_PAYLOAD",
    "build_hard12_manifest",
    "canonical_selection_sha256",
]
