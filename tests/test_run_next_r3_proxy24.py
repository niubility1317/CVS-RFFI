from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_next_r3_proxy24.py"
SPEC = importlib.util.spec_from_file_location("run_next_r3_proxy24_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_missing_real_inputs_fail_closed(tmp_path: Path):
    absent = tmp_path / "absent.npz"
    args = SimpleNamespace(
        received_iq=absent,
        received_iq_sha256="0" * 64,
        phase1_cells=absent,
        phase1_cells_sha256="2" * 64,
    )
    with pytest.raises(runner.MissingRealInputArtifacts, match=r"^MISSING_REAL_INPUT_ARTIFACTS"):
        runner._load_real_rows(args)


def test_new_run_root_refuses_overwrite(tmp_path: Path):
    root = runner._new_root(tmp_path / "run")
    assert (root / "rows").is_dir()
    with pytest.raises(runner.NextR3Proxy24Error, match="new absolute child"):
        runner._new_root(root)


def _predictor_package_fixture(tmp_path: Path):
    receivers = ("1-1", "18-2", "r3", "r4", "r5", "r6", "r7")
    classes = tuple(f"c{index}" for index in range(6))
    physical_ids: list[str] = []
    observation_ids: list[str] = []
    receiver_ids: list[str] = []
    by_receiver_class: dict[tuple[str, str], list[str]] = {}
    observation_by_physical: dict[str, str] = {}
    for receiver in receivers:
        for class_id in classes:
            values: list[str] = []
            for index in range(14):
                physical_id = f"{receiver}-{class_id}-{index:02d}"
                observation_id = f"obs-{physical_id}"
                values.append(physical_id)
                physical_ids.append(physical_id)
                observation_ids.append(observation_id)
                receiver_ids.append(receiver)
                observation_by_physical[physical_id] = observation_id
            by_receiver_class[(receiver, class_id)] = values
    rows = runner.SourceRows(
        received_iq=np.ones((runner.ROW_COUNT, 2, 1), dtype=np.float32),
        receiver_ids=tuple(receiver_ids),
        day_ids=tuple("day" for _ in physical_ids),
        physical_ids=tuple(physical_ids),
        scenario_names=tuple("leo_clear_weak" for _ in physical_ids),
        observation_ids=tuple(observation_ids),
        receiver_registry=receivers,
        received_iq_sha256="a" * 64,
    )
    plan = runner.matrix.build_next_r3_proxy24_plan(classes)
    prepared_by_id: dict[str, dict[str, object]] = {}
    package_rows: list[dict[str, object]] = []
    for planned in plan["rows"]:
        held_receiver = str(planned["held_receiver"])
        held_class = str(planned["held_class"])
        active_k = int(planned["active_k"])
        support_classes = tuple(planned["retained_classes"]) + (held_class,)
        support_ids = [
            physical_id
            for class_id in support_classes
            for physical_id in by_receiver_class[(held_receiver, class_id)][:active_k]
        ]
        query_ids = runner._common_query_order(
            held_receiver,
            held_class,
            [
                physical_id
                for class_id in classes
                for physical_id in by_receiver_class[(held_receiver, class_id)][runner.matrix.MAX_SUPPORT_K :]
            ],
        )
        prepared = {
            "row_id": str(planned["row_id"]),
            "support_physical_ids": support_ids,
            "support_observation_ids": [observation_by_physical[item] for item in support_ids],
            "support_labels": [
                class_id for class_id in support_classes for _ in range(active_k)
            ],
            "query_physical_ids": list(query_ids),
            "query_observation_ids": [observation_by_physical[item] for item in query_ids],
            "prior_key": runner._outer_key(held_receiver, held_class),
        }
        prepared_by_id[str(planned["row_id"])] = prepared
        package_rows.append(prepared)

    checkpoint_sha = "b" * 64
    phase1_seal_sha = "c" * 64
    prior_phase1_root_sha = "d" * 64
    matrix_phase1_root_sha = "e" * 64
    priors: list[dict[str, str]] = []
    seen_prior_keys: set[str] = set()
    for planned in plan["rows"]:
        held_receiver = str(planned["held_receiver"])
        held_class = str(planned["held_class"])
        prior_key = runner._outer_key(held_receiver, held_class)
        if prior_key in seen_prior_keys:
            continue
        seen_prior_keys.add(prior_key)
        binding = runner.tsl.TSL160RuntimeBinding(
            outer_fold_id=(
                f"r3/{held_receiver}/{held_class}|classes="
                f"{','.join(planned['retained_classes'])}"
            ),
            checkpoint_sha256=checkpoint_sha,
            representation_rule_sha256="f" * 64,
            phase1_physical_id_root_sha256=prior_phase1_root_sha,
            phase1_seal_sha256=phase1_seal_sha,
        )
        prior = runner.tsl.TSL160Phase1Prior(
            q_logv0_int8=np.zeros((runner.tsl.Z_DIM,), dtype=np.int8),
            scale_logv0_fp16=np.asarray([0.25], dtype=np.float16),
            offset_logv0_fp16=np.asarray([0.0], dtype=np.float16),
            nu0_fp16=np.asarray([1.0], dtype=np.float16),
            rho_h_mantissa_fp16=np.asarray([0.5], dtype=np.float16),
            rho_h_exp2=np.asarray([0], dtype=np.int16),
            binding=binding,
        )
        priors.append(
            {
                "prior_key": prior_key,
                "prior_wire_json": runner.tsl.serialize_tsl160_prior(prior).decode("ascii"),
                "prior_sha256": prior.prior_sha256,
            }
        )

    pair_bindings: list[dict[str, object]] = []
    for planned in plan["rows"]:
        if int(planned["active_k"]) != 1:
            continue
        held_receiver = str(planned["held_receiver"])
        held_class = str(planned["held_class"])
        row_k1 = prepared_by_id[str(planned["row_id"])]
        planned_k5 = next(
            item
            for item in plan["rows"]
            if item["held_receiver"] == held_receiver
            and item["held_class"] == held_class
            and int(item["active_k"]) == 5
        )
        row_k5 = prepared_by_id[str(planned_k5["row_id"])]

        def support_root(raw: dict[str, object]) -> str:
            return runner._ordered_physical_root(
                tuple(
                    physical_id
                    for class_id in classes
                    for physical_id, label in zip(
                        raw["support_physical_ids"], raw["support_labels"], strict=True
                    )
                    if label == class_id
                )
            )

        pair_bindings.append(
            {
                "prior_key": runner._outer_key(held_receiver, held_class),
                "k1_row_id": row_k1["row_id"],
                "k5_row_id": row_k5["row_id"],
                "phase1_fit_physical_root_sha256": matrix_phase1_root_sha,
                "support_k1_physical_root_sha256": support_root(row_k1),
                "support_k5_physical_root_sha256": support_root(row_k5),
                "query_physical_root_sha256": runner._ordered_physical_root(
                    row_k1["query_physical_ids"]
                ),
                "k1_is_exact_k5_prefix": True,
                "binding_sha256": "0" * 64,
            }
        )

    document: dict[str, object] = {
        "schema": runner.PREDICTOR_PACKAGE_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "received_iq_sha256": rows.received_iq_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "phase1_cells_sha256": phase1_seal_sha,
        "capsule_id": "1" * 64,
        "split_id": "2" * 64,
        "validator_receipt_sha256": "3" * 64,
        "query_order_rule": runner.QUERY_ORDER_RULE,
        "rows": package_rows,
        "priors": priors,
        "pair_bindings": pair_bindings,
    }
    truth = {
        physical_id: class_id
        for class_id in classes
        for receiver in receivers
        for physical_id in by_receiver_class[(receiver, class_id)]
    }

    def write_package(value: dict[str, object]) -> SimpleNamespace:
        path = tmp_path / "predictor_package.json"
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.write_bytes(payload)
        return SimpleNamespace(
            package=path,
            package_sha256=hashlib.sha256(payload).hexdigest(),
        )

    return rows, classes, plan, document, truth, by_receiver_class, observation_by_physical, write_package


def test_predictor_package_separates_truth_and_pairs_common_query(tmp_path: Path):
    rows, classes, plan, document, _truth, _by_cell, _observations, write_package = _predictor_package_fixture(tmp_path)
    package, loaded_plan = runner._load_predictor_package(
        write_package(document), rows, "b" * 64
    )
    assert package.class_registry == classes
    assert loaded_plan["matrix_sha256"] == plan["matrix_sha256"]
    assert "truth" not in document
    for planned in plan["rows"]:
        if int(planned["active_k"]) != 1:
            continue
        row_k1 = package.row(str(planned["row_id"]))
        row_k5 = next(
            package.row(str(item["row_id"]))
            for item in plan["rows"]
            if item["held_receiver"] == planned["held_receiver"]
            and item["held_class"] == planned["held_class"]
            and int(item["active_k"]) == 5
        )
        assert row_k1.query_physical_ids == row_k5.query_physical_ids
        assert row_k1.query_observation_ids == row_k5.query_observation_ids
        for class_id in classes:
            assert row_k1.support_for(class_id) == row_k5.support_for(class_id)[:1]


def test_predictor_refuses_legacy_direct_entry_without_package(tmp_path: Path):
    rows, *_ = _predictor_package_fixture(tmp_path)
    with pytest.raises(runner.MissingRealInputArtifacts, match="predictor package"):
        runner._load_predictor_package(SimpleNamespace(), rows, "b" * 64)


def test_predict_cli_has_no_legacy_split_or_phase1_input():
    parser = runner._parser()
    predict = next(
        action.choices["predict"]
        for action in parser._actions
        if isinstance(getattr(action, "choices", None), dict)
        and "predict" in action.choices
    )
    options = {option for action in predict._actions for option in action.option_strings}
    assert "--package" in options
    assert "--truth-free-split" not in options
    assert "--phase1-cells" not in options


def test_predictor_package_rejects_full_class_labels(tmp_path: Path):
    rows, classes, _plan, document, _truth, _by_cell, _observations, write_package = _predictor_package_fixture(tmp_path)
    document["class_ids"] = list(classes)
    with pytest.raises(runner.NextR3Proxy24Error, match="forbidden"):
        runner._load_predictor_package(write_package(document), rows, "b" * 64)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reg0_query_physical_ids", []),
        ("held_role", "held-class"),
    ),
)
def test_predictor_package_rejects_dual_query_or_held_role_fields(
    tmp_path: Path, field: str, value: object
):
    rows, _classes, _plan, document, _truth, _by_cell, _observations, write_package = _predictor_package_fixture(tmp_path)
    document["rows"][0][field] = value
    with pytest.raises(runner.NextR3Proxy24Error, match="forbidden"):
        runner._load_predictor_package(write_package(document), rows, "b" * 64)


def test_predictor_package_rejects_class_major_query_grouping(tmp_path: Path):
    rows, classes, plan, document, _truth, by_cell, observations, write_package = _predictor_package_fixture(tmp_path)
    raw = document["rows"][0]
    planned = next(item for item in plan["rows"] if item["row_id"] == raw["row_id"])
    class_major = [
        physical_id
        for class_id in classes
        for physical_id in by_cell[(str(planned["held_receiver"]), class_id)][runner.matrix.MAX_SUPPORT_K :]
    ]
    assert class_major != raw["query_physical_ids"]
    raw["query_physical_ids"] = class_major
    raw["query_observation_ids"] = [observations[item] for item in class_major]
    with pytest.raises(runner.NextR3Proxy24Error, match="common-query identity/order"):
        runner._load_predictor_package(write_package(document), rows, "b" * 64)


def test_prepare_artifacts_keep_truth_outside_predictor_package(tmp_path: Path):
    _rows, _classes, plan, document, truth, _by_cell, _observations, _write_package = _predictor_package_fixture(tmp_path)
    result = runner._write_prepare_artifacts(
        tmp_path / "prepare",
        package=document,
        truth=truth,
        receipt={"matrix_sha256": plan["matrix_sha256"]},
    )
    package = json.loads(Path(result["package"]).read_text(encoding="utf-8"))
    receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
    assert "truth" not in package
    assert Path(result["truth"]).is_file()
    assert receipt["truth_in_predictor_package"] is False
    assert receipt["package_has_two_query_lists"] is False
    assert receipt["package_has_full_class_ids"] is False


def test_score_defers_truth_and_writer_refuses_output_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_root = tmp_path / "incomplete_prediction"
    run_root.mkdir()
    output = tmp_path / "score.json"
    output.write_text("existing", encoding="utf-8")

    def unexpected_truth_open(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("truth must not open before prediction closure")

    monkeypatch.setattr(runner, "_require_file", unexpected_truth_open)
    with pytest.raises(runner.NextR3Proxy24Error, match="complete sealed"):
        runner.run_score(
            SimpleNamespace(
                run_root=run_root,
                truth=tmp_path / "truth.json",
                truth_sha256="0" * 64,
                output=output,
            )
        )
    with pytest.raises(runner.NextR3Proxy24Error, match="output overwrite refused"):
        runner._write_json_new(output, {"score": "new"})


def test_bridge_cache_rejects_unbound_received_iq(tmp_path: Path):
    del tmp_path
    rows = runner.SourceRows(
        received_iq=np.ones((2, 2, 1), dtype=np.float32),
        receiver_ids=("r", "r"),
        day_ids=("d", "d"),
        physical_ids=("p0", "p1"),
        scenario_names=("leo_clear_weak", "leo_clear_weak"),
        observation_ids=("o0", "o1"),
        receiver_registry=("r",),
        received_iq_sha256="a" * 64,
    )
    bridge = SimpleNamespace(
        checkpoint_sha256="b" * 64,
        rows=SimpleNamespace(
            received_iq=np.zeros((2, 2, 1), dtype=np.float32),
            physical_ids=("p0", "p1"),
            observation_ids=("o0", "o1"),
        ),
    )
    with pytest.raises(runner.NextR3Proxy24Error, match="bridge input binding drift"):
        runner.BridgeFeatureCache(bridge, rows, "b" * 64)
