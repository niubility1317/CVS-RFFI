import copy
import importlib
import json
import sqlite3
from pathlib import Path

import pytest


PROFILE_PATH = Path(__file__).parents[1] / "configs" / "phase2_canonical_union_profiles_v1.json"
FORMAL_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EXPECTED_PROFILE = {
    "schema": "cvs.phase2.canonical_union_profile.v1",
    "protocol_schema": "p2_min_v1",
    "source_profile_id": "SRC5_MAXP2",
    "source_receivers": ["1-19", "18-2", "19-2", "2-19", "3-19"],
    "receiver_tiers": {
        "dense": ["1-1", "14-7", "2-1", "20-1", "7-14", "7-7", "8-8"],
        "single_day": ["13-13", "2-20", "8-13"],
        "many_tx": ["1-20", "13-7", "18-19", "19-1", "20-19", "8-14", "8-7"],
    },
    "old_tx_ids": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
    "new_tx_candidates": [
        "1-11",
        "10-11",
        "10-7",
        "11-1",
        "11-17",
        "11-4",
        "11-7",
        "13-3",
        "15-1",
        "16-16",
        "2-19",
        "20-12",
        "20-7",
        "3-13",
        "3-18",
        "4-11",
        "5-5",
        "6-1",
        "7-10",
        "7-11",
        "8-18",
        "8-3",
    ],
    "new_class_sizes": [5, 10, 20],
    "k_values": [1, 5, 10, 20],
    "k_max": 20,
    "scenarios": list(FORMAL_SCENARIOS),
    "query_policies": ["MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE"],
}


def _split_module():
    try:
        return importlib.import_module("cvsrffi.phase2_canonical_split")
    except ModuleNotFoundError as error:
        pytest.fail(f"Task 3 production module is missing: {error}")


def _profile_payload(*, simple_ids: bool = False):
    payload = copy.deepcopy(EXPECTED_PROFILE)
    if simple_ids:
        payload["source_receivers"] = ["source-rx"]
        payload["receiver_tiers"] = {
            "dense": ["rx-a", "rx-b"],
            "single_day": [],
            "many_tx": [],
        }
        payload["old_tx_ids"] = [f"old-{index:02d}" for index in range(6)]
        payload["new_tx_candidates"] = [f"tx-{index:02d}" for index in range(22)]
    return payload


def _connection():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE canonical_records (
          physical_sample_id TEXT PRIMARY KEY,
          tx_id TEXT NOT NULL,
          rx_id TEXT NOT NULL,
          day_id TEXT NOT NULL,
          eq_id TEXT NOT NULL,
          sig_id TEXT NOT NULL,
          iq_sha256 TEXT NOT NULL,
          preferred_asset TEXT NOT NULL,
          preferred_source_record_index INTEGER NOT NULL,
          eligible INTEGER NOT NULL CHECK (eligible IN (0,1))
        )
        """
    )
    return connection


def _add_record(
    connection,
    scenario_by_sample,
    *,
    sample_id,
    tx_id,
    rx_id,
    day_id,
    scenario,
    eligible=1,
):
    connection.execute(
        """
        INSERT INTO canonical_records (
          physical_sample_id, tx_id, rx_id, day_id, eq_id, sig_id, iq_sha256,
          preferred_asset, preferred_source_record_index, eligible
        ) VALUES (?, ?, ?, ?, '1', ?, 'digest', 'ManyTx', 0, ?)
        """,
        (sample_id, tx_id, rx_id, day_id, sample_id, eligible),
    )
    if scenario is not None:
        scenario_by_sample[sample_id] = scenario


def test_exact_json_loads_with_immutable_normalization():
    split = _split_module()
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    assert payload == EXPECTED_PROFILE
    profile = split.CanonicalProfile.from_mapping(payload)
    assert profile.source_receivers == tuple(EXPECTED_PROFILE["source_receivers"])
    assert tuple(profile.receiver_tiers) == ("dense", "single_day", "many_tx")
    assert profile.receiver_tiers["dense"] == tuple(EXPECTED_PROFILE["receiver_tiers"]["dense"])
    assert profile.new_tx_candidates == tuple(EXPECTED_PROFILE["new_tx_candidates"])
    assert profile.new_class_sizes == (5, 10, 20)
    with pytest.raises(TypeError):
        profile.receiver_tiers["dense"] = ("changed",)
    with pytest.raises((AttributeError, TypeError)):
        profile.k_max = 5


def test_profile_rejects_source_target_overlap_with_rt_error():
    split = _split_module()
    payload = _profile_payload()
    payload["receiver_tiers"]["dense"].append("1-19")

    with pytest.raises(ValueError, match="R_t"):
        split.CanonicalProfile.from_mapping(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda p: p.__setitem__("schema", "wrong"), "schema"),
        (lambda p: p.__setitem__("protocol_schema", "wrong"), "protocol_schema"),
        (lambda p: p["receiver_tiers"]["dense"].append(p["receiver_tiers"]["dense"][0]), "receiver"),
        (lambda p: p["receiver_tiers"]["many_tx"].append(p["receiver_tiers"]["dense"][0]), "receiver"),
        (lambda p: p["receiver_tiers"].update({"extra": []}), "tier"),
        (lambda p: p["old_tx_ids"].pop(), "six"),
        (lambda p: p["old_tx_ids"].__setitem__(0, p["old_tx_ids"][1]), "unique"),
        (lambda p: p["new_tx_candidates"].pop(), "22"),
        (lambda p: p["new_tx_candidates"].__setitem__(0, p["new_tx_candidates"][1]), "unique"),
        (lambda p: p["new_tx_candidates"].__setitem__(0, p["old_tx_ids"][0]), "disjoint"),
        (lambda p: p.__setitem__("new_class_sizes", [5, 10, 21]), "new_class_sizes"),
        (lambda p: p.__setitem__("k_values", [1, 5, 20]), "k_values"),
        (lambda p: p.__setitem__("k_max", 10), "k_max"),
        (lambda p: p.__setitem__("scenarios", list(reversed(FORMAL_SCENARIOS))), "scenarios"),
        (lambda p: p.__setitem__("query_policies", ["BALANCED_4DAY_CORE", "MAXQ_ALL_UNIQUE"]), "query_policies"),
    ],
)
def test_profile_rejects_invalid_receiver_tx_k_scene_and_policy(mutation, match):
    split = _split_module()
    payload = _profile_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=match):
        split.CanonicalProfile.from_mapping(payload)


def test_receiver_eligibility_requires_every_class_and_scene_at_k_in_input_order():
    split = _split_module()
    connection = _connection()
    scenario_by_sample = {}
    for rx_id in ("rx-good-2", "rx-good-1", "rx-short"):
        for tx_id in ("old-a", "new-a"):
            for scenario in FORMAL_SCENARIOS:
                count = 1 if rx_id == "rx-short" and tx_id == "new-a" and scenario == FORMAL_SCENARIOS[-1] else 2
                for index in range(count):
                    sample_id = f"{rx_id}-{tx_id}-{scenario}-{index}"
                    _add_record(
                        connection,
                        scenario_by_sample,
                        sample_id=sample_id,
                        tx_id=tx_id,
                        rx_id=rx_id,
                        day_id="day-1",
                        scenario=scenario,
                    )

    assert split.eligible_receivers(
        connection,
        registered_tx_ids=("old-a", "new-a"),
        candidate_receivers=("rx-good-2", "rx-short", "rx-good-1"),
        scenario_by_sample=scenario_by_sample,
        k=2,
    ) == ("rx-good-2", "rx-good-1")


def test_receiver_eligibility_rejects_nonpositive_k_and_duplicate_inputs():
    split = _split_module()
    connection = _connection()
    with pytest.raises(ValueError, match="positive"):
        split.eligible_receivers(
            connection,
            registered_tx_ids=("tx",),
            candidate_receivers=("rx",),
            scenario_by_sample={},
            k=0,
        )
    with pytest.raises(ValueError, match="duplicate"):
        split.eligible_receivers(
            connection,
            registered_tx_ids=("tx", "tx"),
            candidate_receivers=("rx",),
            scenario_by_sample={},
            k=1,
        )
    with pytest.raises(ValueError, match="duplicate"):
        split.eligible_receivers(
            connection,
            registered_tx_ids=("tx",),
            candidate_receivers=("rx", "rx"),
            scenario_by_sample={},
            k=1,
        )


def test_new_class_ranking_is_coverage_only_deterministic_and_nested():
    split = _split_module()
    profile = split.CanonicalProfile.from_mapping(_profile_payload(simple_ids=True))
    connection = _connection()
    scenario_by_sample = {}
    for tx_index, tx_id in enumerate(profile.new_tx_candidates):
        for rx_id in ("rx-a", "rx-b"):
            for scenario in FORMAL_SCENARIOS:
                for index in range(profile.k_max):
                    sample_id = f"{tx_id}-{rx_id}-{scenario}-{index}"
                    _add_record(
                        connection,
                        scenario_by_sample,
                        sample_id=sample_id,
                        tx_id=tx_id,
                        rx_id=rx_id,
                        day_id="day-1",
                        scenario=scenario,
                    )
        for extra in range(tx_index):
            sample_id = f"{tx_id}-extra-{extra}"
            _add_record(
                connection,
                scenario_by_sample,
                sample_id=sample_id,
                tx_id=tx_id,
                rx_id="rx-a",
                day_id="day-2",
                scenario=FORMAL_SCENARIOS[0],
            )

    first = split.rank_new_classes(connection, profile, scenario_by_sample)
    second = split.rank_new_classes(connection, profile, scenario_by_sample)

    assert first == second
    assert first[5] == first[10][:5]
    assert first[10] == first[20][:10]
    assert len(first[20]) == len(set(first[20])) == 20
    assert first[20][0] == "tx-21"


def test_new_class_ranking_uses_tx_lexical_tie_break():
    split = _split_module()
    payload = _profile_payload(simple_ids=True)
    payload["new_tx_candidates"][:2] = ["tx-b", "tx-a"]
    profile = split.CanonicalProfile.from_mapping(payload)
    connection = _connection()
    scenario_by_sample = {}
    for tx_id in ("tx-b", "tx-a"):
        for scenario in FORMAL_SCENARIOS:
            sample_id = f"{tx_id}-{scenario}"
            _add_record(
                connection,
                scenario_by_sample,
                sample_id=sample_id,
                tx_id=tx_id,
                rx_id="rx-a",
                day_id="day-1",
                scenario=scenario,
            )

    ranked = split.rank_new_classes(connection, profile, scenario_by_sample)

    assert ranked[20][:2] == ("tx-a", "tx-b")


def test_new_class_ranking_counts_missing_cells_in_fixed_target_universe_as_zero():
    split = _split_module()
    payload = _profile_payload(simple_ids=True)
    payload["new_tx_candidates"][:2] = ["tx-narrow", "tx-broad"]
    profile = split.CanonicalProfile.from_mapping(payload)
    connection = _connection()
    scenario_by_sample = {}
    for scenario in FORMAL_SCENARIOS:
        for index in range(2):
            _add_record(
                connection,
                scenario_by_sample,
                sample_id=f"tx-narrow-rx-a-{scenario}-{index}",
                tx_id="tx-narrow",
                rx_id="rx-a",
                day_id="day-1",
                scenario=scenario,
            )
        for rx_id in ("rx-a", "rx-b"):
            _add_record(
                connection,
                scenario_by_sample,
                sample_id=f"tx-broad-{rx_id}-{scenario}",
                tx_id="tx-broad",
                rx_id=rx_id,
                day_id="day-1",
                scenario=scenario,
            )

    ranked = split.rank_new_classes(connection, profile, scenario_by_sample)

    assert ranked[20].index("tx-broad") < ranked[20].index("tx-narrow")


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("new_class_sizes", [5, 10, 20.5]),
        ("new_class_sizes", [5, "10", 20]),
        ("new_class_sizes", [False, 10, 20]),
        ("k_values", [1, 5, 10, 20.0]),
        ("k_values", [1, "5", 10, 20]),
        ("k_values", [True, 5, 10, 20]),
        ("k_max", 20.0),
        ("k_max", "20"),
        ("k_max", True),
    ],
)
def test_profile_rejects_non_exact_integer_json_values(field, invalid_value):
    split = _split_module()
    payload = _profile_payload()
    payload[field] = invalid_value

    with pytest.raises(ValueError, match="exact integers"):
        split.CanonicalProfile.from_mapping(payload)


def test_ineligible_and_missing_or_unknown_scene_records_do_not_count():
    split = _split_module()
    payload = _profile_payload(simple_ids=True)
    payload["new_tx_candidates"][:2] = ["tx-a", "tx-b"]
    profile = split.CanonicalProfile.from_mapping(payload)
    connection = _connection()
    scenario_by_sample = {}
    for index in range(10):
        _add_record(
            connection,
            scenario_by_sample,
            sample_id=f"tx-a-ineligible-{index}",
            tx_id="tx-a",
            rx_id="rx-a",
            day_id="day-1",
            scenario=FORMAL_SCENARIOS[0],
            eligible=0,
        )
        _add_record(
            connection,
            scenario_by_sample,
            sample_id=f"tx-a-unmapped-{index}",
            tx_id="tx-a",
            rx_id="rx-a",
            day_id="day-1",
            scenario=None,
        )
        _add_record(
            connection,
            scenario_by_sample,
            sample_id=f"tx-a-unknown-{index}",
            tx_id="tx-a",
            rx_id="rx-a",
            day_id="day-1",
            scenario="not-formal",
        )
    _add_record(
        connection,
        scenario_by_sample,
        sample_id="tx-a-one-valid",
        tx_id="tx-a",
        rx_id="rx-a",
        day_id="day-1",
        scenario=FORMAL_SCENARIOS[0],
    )
    for index in range(2):
        _add_record(
            connection,
            scenario_by_sample,
            sample_id=f"tx-b-valid-{index}",
            tx_id="tx-b",
            rx_id="rx-a",
            day_id="day-1",
            scenario=FORMAL_SCENARIOS[index],
        )

    ranked = split.rank_new_classes(connection, profile, scenario_by_sample)
    eligible = split.eligible_receivers(
        connection,
        registered_tx_ids=("tx-a",),
        candidate_receivers=("rx-a",),
        scenario_by_sample=scenario_by_sample,
        k=1,
    )

    assert ranked[20].index("tx-b") < ranked[20].index("tx-a")
    assert eligible == ()
