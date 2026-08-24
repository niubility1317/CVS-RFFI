from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
OPERATING_POINTS = {
    "K10/new5": 10,
    "K10/new10": 10,
    "K10/new20": 10,
    "K5/new20": 5,
    "K1/new20": 1,
}


def _write_raw_inputs(root: Path, scenario: str) -> tuple[Path, Path]:
    support = root / f"support_{scenario}.npz"
    query = root / f"query_{scenario}.npz"
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 10)
    ranks = np.tile(np.arange(10, dtype=np.int64), 2)
    np.savez(
        support,
        support_pool_leo_weak_iq=np.ones((20, 2, 8), dtype=np.float32),
        support_pool_class_indices=labels,
        support_pool_rank_within_class=ranks,
        support_pool_tokens=np.asarray(
            [f"physical-support-{scenario}-{index:03d}" for index in range(20)]
        ),
    )
    np.savez(
        query,
        query_leo_weak_iq=np.ones((4, 2, 8), dtype=np.float32),
        query_tokens=np.asarray(
            [f"opaque-query-{scenario}-{index:03d}" for index in range(4)]
        ),
    )
    return support, query


def _plan(tmp_path: Path) -> dict[str, object]:
    scenario_inputs = {}
    for scenario in SCENARIOS:
        support, query = _write_raw_inputs(tmp_path, scenario)
        scenario_inputs[scenario] = {
            "support_input": str(support),
            "query_input": str(query),
        }
    entries = []
    for operating_point, k_shot in OPERATING_POINTS.items():
        tag = operating_point.replace("/", "_").lower()
        identity_tag = f"k{k_shot}-new{operating_point.rsplit('new', 1)[1]}"
        manifest = tmp_path / f"manifest_{tag}.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "cvs.full_ablation.phase2.feature_cache_manifest.v2",
                    "stage_scope": "stage2b",
                    "receiver": "20-1",
                    "k_shot": k_shot,
                    "phase2_data_status": "VALIDATED_ONCE",
                    "capsule_id": f"capsule-{identity_tag}",
                    "split_id": f"p2_min_v1-rx20-1-{identity_tag}",
                    "scenarios": list(SCENARIOS),
                    "query_truth_present": False,
                    "query_role_present": False,
                    "phase2_source_sample_access": False,
                    "phase2_source_cache_access": False,
                    "phase2_source_label_access": False,
                    "phase2_source_replay": False,
                }
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "receiver": "20-1",
                "operating_point": operating_point,
                "k_shot": k_shot,
                "manifest_path": str(manifest),
                "scenario_inputs": scenario_inputs,
            }
        )
    return {
        "schema": "cvs.stage2.meta_adapter.target_factory_plan.v1",
        "target": "Target5",
        "candidate_id": "CVS_META_ADAPTER_TRI_R4_V1",
        "bundle_id": "ADV3B02_CORE90_SOFT_E200_META_TRI_R4_V1",
        "checkpoint_path": "selected_meta_bundle.pt",
        "prototype_path": "frozen_prototypes.npz",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "seed": 713101,
        "steps": 3,
        "entries": entries,
    }


def test_factory_exports_complete_target5_without_truth(tmp_path: Path) -> None:
    import cvsrffi.stage2_meta_adapter_target_factory as sut

    output = tmp_path / "target5_inputs"
    receipt = sut.build_meta_adapter_target_matrix(_plan(tmp_path), output)

    assert receipt["status"] == "TARGET_INPUTS_COMPLETE"
    assert receipt["target"] == "Target5"
    assert receipt["row_count"] == 15
    assert receipt["query_truth_opened"] is False
    matrix = json.loads((output / "matrix_config.json").read_text(encoding="utf-8"))
    assert matrix["schema"] == "cvs.stage2.meta_adapter.matrix.v1"
    assert matrix["target"] == "Target5"
    assert len(matrix["rows"]) == 15
    assert {
        row["config"]["operating_point"] for row in matrix["rows"]
    } == set(OPERATING_POINTS)
    for row in matrix["rows"]:
        config = row["config"]
        assert config["protocol_schema"] == "p2_min_v1"
        assert config["phase2_data_status"] == "VALIDATED_ONCE"
        assert config["steps"] == 3
        with np.load(config["support_path"], allow_pickle=False) as support:
            assert set(support.files) == {
                "received_iq",
                "support_labels",
                "support_physical_ids",
            }
            assert support["received_iq"].shape[0] == 2 * config["k_shot"]
        with np.load(config["query_path"], allow_pickle=False) as query:
            assert set(query.files) == {"received_iq", "query_ids"}
        audit = json.loads(
            (Path(config["support_path"]).parent / "export_audit.json").read_text(
                encoding="utf-8"
            )
        )
        assert audit["query_truth_opened"] is False
        assert audit["query_role_opened"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda plan: plan["entries"].pop(1), "five operating points"),
        (
            lambda plan: plan["entries"][1].update(
                {"operating_point": "K2/new20", "k_shot": 2}
            ),
            "operating-point",
        ),
        (
            lambda plan: plan["entries"][1].update(
                {"manifest_path": plan["entries"][2]["manifest_path"]}
            ),
            "operating-point identity",
        ),
        (
            lambda plan: json.loads(
                Path(plan["entries"][0]["manifest_path"]).read_text(encoding="utf-8")
            ).update({"query_truth_present": True}),
            "query truth",
        ),
    ],
)
def test_factory_rejects_incomplete_or_forbidden_plan_before_output(
    tmp_path: Path, mutation, match: str
) -> None:
    import cvsrffi.stage2_meta_adapter_target_factory as sut

    plan = _plan(tmp_path)
    if match == "query truth":
        path = Path(plan["entries"][0]["manifest_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["query_truth_present"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        mutation(plan)
    output = tmp_path / "target5_inputs"
    with pytest.raises(ValueError, match=match):
        sut.build_meta_adapter_target_matrix(plan, output)
    assert not output.exists()
