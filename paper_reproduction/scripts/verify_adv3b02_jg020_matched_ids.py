"""Fail closed unless all 25 JG K10 rows reuse the historical physical IDs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(CODE_ROOT), str(REPO_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_leo_weak_cache_set,
)
from paper_reproduction.scripts.build_adv3b02_jg020_matched_k10_plan import (  # noqa: E402
    RECEIVERS,
    SEEDS,
)
from scripts.build_cvs_stage2_predictor_bundle import (  # noqa: E402
    _select_support_query,
)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _legacy_manifest(root: Path, receiver: str, seed: int) -> Path:
    relative = (
        Path(f"rx_{_safe(receiver)}")
        / f"seed_{seed}"
        / "k_10"
        / "mrior_sda"
        / "split_manifest.json"
    )
    candidates = (
        root / relative,
        root / "formal" / relative,
        root / "stage2_runs" / relative,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"legacy K10 split manifest missing: {candidates}")


def _manifest_from_spec(spec_path: Path) -> Path:
    payload = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    value = Path(str(payload["out_manifest"]))
    return value if value.is_absolute() else spec_path.parent / value


def verify(plan_path: Path, legacy_root: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != "adv3b02_jg020_matched_stage2b_k10_plan_v1":
        raise ValueError("unexpected JG matched plan schema")
    phase2 = json.loads(Path(str(plan["phase2_config"])).read_text(encoding="utf-8-sig"))
    old_labels = [str(value) for value in phase2["target_old_tx_labels"]]
    spec_by_key: dict[tuple[str, int], Path] = {}
    plan_root = Path(str(plan["runtime_plan_dir"]))
    for entry in plan["cache_specs"]:
        spec_path = plan_root / str(entry["spec"])
        match = re.search(r"rx_([^/\\]+)[/\\]seed_(\d+)\.json$", str(spec_path))
        if match is None:
            raise ValueError(f"cannot parse target cache spec path: {spec_path}")
        receiver = match.group(1).replace("_", "-")
        spec_by_key[(receiver, int(match.group(2)))] = spec_path

    rows: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            spec_path = spec_by_key[(receiver, seed)]
            arrays_by_scenario, cache_manifest, cache_audit = load_verified_leo_weak_cache_set(
                _manifest_from_spec(spec_path),
                expected_scope="stage2_target_old",
                allowed_roles={"target_old"},
            )
            reference = arrays_by_scenario[FORMAL_LEO_WEAK_SCENARIOS[0]]
            support_indices, query_indices, _support_y, support_rank, _query_records = (
                _select_support_query(
                    reference,
                    receiver=receiver,
                    seed=seed,
                    support_labels=[("target_old", label) for label in old_labels],
                    reference_query_labels=[],
                    support_pool_max_k=int(phase2["support_pool_max_k"]),
                    query_per_tx=int(phase2["query_per_tx"]),
                    use_offline_split_partition=True,
                )
            )
            sample_ids = np.asarray(reference["sample_ids"]).astype(str)
            selected_support = [
                value.split("|", 1)[1]
                for value in sample_ids[support_indices[support_rank < 10]].tolist()
            ]
            selected_query = [
                value.split("|", 1)[1] for value in sample_ids[query_indices].tolist()
            ]
            legacy_path = _legacy_manifest(legacy_root, receiver, seed)
            legacy = json.loads(legacy_path.read_text(encoding="utf-8-sig"))
            legacy_support = [str(value) for value in legacy["target_old_support_sample_ids"]]
            legacy_query = [str(value) for value in legacy["target_old_query_sample_ids"]]
            scenario_match = tuple(legacy.get("target_channel_scenarios", ())) == tuple(
                FORMAL_LEO_WEAK_SCENARIOS
            )
            support_set_match = set(selected_support) == set(legacy_support)
            query_set_match = set(selected_query) == set(legacy_query)
            support_order_match = selected_support == legacy_support
            query_order_match = selected_query == legacy_query
            expected_support_seeds = {
                scenario: seed + 1000 + index
                for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
            }
            expected_query_seeds = {
                scenario: seed + 2000 + index
                for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
            }
            support_seed_match = (
                cache_manifest.get("support_satellite_seed_by_scenario")
                == expected_support_seeds
            )
            query_seed_match = (
                cache_manifest.get("query_satellite_seed_by_scenario")
                == expected_query_seeds
            )
            for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
                scenario_arrays = arrays_by_scenario[scenario]
                selected_support_seeds = np.asarray(
                    scenario_arrays["satellite_seeds"]
                )[support_indices[support_rank < 10]]
                selected_query_seeds = np.asarray(
                    scenario_arrays["satellite_seeds"]
                )[query_indices]
                support_seed_match = support_seed_match and bool(
                    np.all(selected_support_seeds == seed + 1000 + scenario_index)
                )
                query_seed_match = query_seed_match and bool(
                    np.all(selected_query_seeds == seed + 2000 + scenario_index)
                )
            runner_binding_match = cache_manifest.get("legacy_runner_sha256") == (
                "1270dbdb40285393519796a65a4f9bce3a0a89debdfce0e9a3ca1521a930a9db"
            ) and cache_manifest.get("legacy_runner_git_commit") == (
                "d7f2f549ceb4903c1ab8b219b44f581379deacf3"
            ) and cache_manifest.get("apply_scenario_source_sha256") == (
                "0441168c391db173db25501165098e0b7236d475003cfdb31b56f5a1f139a22d"
            ) and cache_manifest.get("legacy_support_query_call_ast_sha256") == (
                "1d6f306184fdee90b1c3333714fc187e3c25a0f6836a88c93bf43aa401ecfdf4"
            )
            query_view = int(cache_manifest.get("query_view_count", 0))
            row_identity_match = (
                str(legacy.get("target_receiver_label")) == receiver
                and int(legacy.get("seed", -1)) == seed
                and int(legacy.get("split_seed", -1)) == seed
                and int(legacy.get("k_shot", -1)) == 10
                and int(legacy.get("support_pool_max_k", -1)) == 20
                and int(legacy.get("query_per_tx", -1)) == 20
                and legacy.get("target_sample_strategy") == "seeded_nested"
            )
            status = (
                "PASS"
                if support_set_match
                and query_set_match
                and support_order_match
                and query_order_match
                and scenario_match
                and query_view == 1
                and support_seed_match
                and query_seed_match
                and runner_binding_match
                and row_identity_match
                else "FAIL"
            )
            rows.append(
                {
                    "receiver": receiver,
                    "seed": seed,
                    "k_shot": 10,
                    "status": status,
                    "support_id_set_match": support_set_match,
                    "query_id_set_match": query_set_match,
                    "support_id_order_match": support_order_match,
                    "query_id_order_match": query_order_match,
                    "support_count": len(selected_support),
                    "query_count": len(selected_query),
                    "scenario_tuple_match": scenario_match,
                    "query_view_count": query_view,
                    "support_view_seed_formula_match": support_seed_match,
                    "query_view_seed_formula_match": query_seed_match,
                    "legacy_runner_binding_match": runner_binding_match,
                    "row_identity_contract_match": row_identity_match,
                    "legacy_split_manifest": str(legacy_path),
                    "target_cache_set_id": cache_manifest["cache_set_id"],
                    "cache_audit_status": cache_audit["status"],
                }
            )
    passed = sum(row["status"] == "PASS" for row in rows)
    result = {
        "schema": "adv3b02_jg020_matched_id_audit_v1",
        "status": "PASS" if passed == 25 else "FAIL",
        "row_count": len(rows),
        "passed_row_count": passed,
        "same_target_receiver": True,
        "same_seed": True,
        "same_k_shot": True,
        "same_support_query_physical_id_sets": passed == 25,
        "same_query_view_strategy": passed == 25,
        "rows": rows,
    }
    if result["status"] != "PASS":
        failed = [(row["receiver"], row["seed"]) for row in rows if row["status"] != "PASS"]
        raise ValueError(f"matched ID/View gate failed for rows={failed}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--legacy-run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.plan_manifest, args.legacy_run_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": result["row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
