import copy
import importlib
import json
import sqlite3
from collections import Counter
from dataclasses import FrozenInstanceError
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
    iq_sha256="digest",
    source_asset="ManyTx",
    source_record_index=0,
):
    connection.execute(
        """
        INSERT INTO canonical_records (
          physical_sample_id, tx_id, rx_id, day_id, eq_id, sig_id, iq_sha256,
          preferred_asset, preferred_source_record_index, eligible
        ) VALUES (?, ?, ?, ?, '1', ?, ?, ?, ?, ?)
        """,
        (
            sample_id,
            tx_id,
            rx_id,
            day_id,
            sample_id,
            iq_sha256,
            source_asset,
            source_record_index,
            eligible,
        ),
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


def _eligible_record_mappings(connection):
    columns = (
        "physical_sample_id",
        "tx_id",
        "rx_id",
        "day_id",
        "preferred_asset",
        "preferred_source_record_index",
        "eligible",
    )
    return [
        dict(zip(columns, row))
        for row in connection.execute(
            """
            SELECT physical_sample_id, tx_id, rx_id, day_id,
                   preferred_asset, preferred_source_record_index, eligible
            FROM canonical_records
            WHERE eligible = 1
            ORDER BY physical_sample_id
            """
        )
    ]


@pytest.fixture(scope="module")
def canonical_split_context():
    split = _split_module()
    payload = _profile_payload(simple_ids=True)
    payload["receiver_tiers"] = {
        "dense": ["rx-a", "rx-b"],
        "single_day": [],
        "many_tx": [],
    }
    profile = split.CanonicalProfile.from_mapping(payload)
    connection = _connection()
    ignored_scene_map = {}
    source_index = 0
    expected_registered = profile.old_tx_ids + tuple(f"tx-{index:02d}" for index in range(20))

    # Sixty-three rows in each TX/RX/day group produce exactly 21 rows per
    # formal scene.  Even after the global first-20 support reserve, every
    # required four-day BAL4D cell retains at least one query candidate.
    for tx_id in expected_registered:
        for day_index in range(4):
            for sample_index in range(63):
                _add_record(
                    connection,
                    ignored_scene_map,
                    sample_id=f"p-{tx_id}-rx-a-d{day_index}-{sample_index:02d}",
                    tx_id=tx_id,
                    rx_id="rx-a",
                    day_id=f"day-{day_index}",
                    scenario=None,
                    source_record_index=source_index,
                )
                source_index += 1

    # rx-b is a declared target receiver but deliberately lacks complete
    # registered-class coverage and must be excluded at every K.
    for sample_index in range(3):
        _add_record(
            connection,
            ignored_scene_map,
            sample_id=f"p-rx-b-{sample_index}",
            tx_id=profile.old_tx_ids[0],
            rx_id="rx-b",
            day_id="day-0",
            scenario=None,
            source_record_index=source_index,
        )
        source_index += 1

    _add_record(
        connection,
        ignored_scene_map,
        sample_id="p-source-receiver",
        tx_id=profile.old_tx_ids[0],
        rx_id="source-rx",
        day_id="day-0",
        scenario=None,
        source_record_index=source_index,
    )
    source_index += 1
    _add_record(
        connection,
        ignored_scene_map,
        sample_id="p-unregistered-tx",
        tx_id="outside-registered-pool",
        rx_id="rx-a",
        day_id="day-0",
        scenario=None,
        source_record_index=source_index,
    )
    source_index += 1
    _add_record(
        connection,
        ignored_scene_map,
        sample_id="p-ineligible",
        tx_id=profile.old_tx_ids[0],
        rx_id="rx-a",
        day_id="day-0",
        scenario=None,
        eligible=0,
        source_record_index=source_index,
    )
    connection.commit()

    eligible_records = _eligible_record_mappings(connection)
    assignments = split.assign_scenes(eligible_records, seed=713101)
    plain_scenes = {
        sample_id: metadata.scene for sample_id, metadata in assignments.items()
    }
    class_selection = split.rank_new_classes(connection, profile, plain_scenes)
    assert class_selection[20] == tuple(f"tx-{index:02d}" for index in range(20))
    retained_ids = {
        row[0]
        for row in connection.execute(
            """
            SELECT physical_sample_id
            FROM canonical_records
            WHERE eligible = 1 AND rx_id = 'rx-a'
            """
        )
        if row[0].startswith("p-old-") or row[0].startswith("p-tx-")
    }
    try:
        yield {
            "split": split,
            "connection": connection,
            "profile": profile,
            "eligible_records": eligible_records,
            "assignments": assignments,
            "class_selection": class_selection,
            "registered_tx_ids": expected_registered,
            "retained_ids": retained_ids,
        }
    finally:
        connection.close()


def test_scene_assignment_is_deterministic_disjoint_near_even_and_rotated():
    split = _split_module()
    records = [
        {
            "physical_sample_id": f"group-{group_index}-sample-{sample_index}",
            "tx_id": f"tx-{group_index}",
            "rx_id": "rx-a",
            "day_id": "day-0",
        }
        for group_index in range(12)
        for sample_index in range(8)
    ]

    first = split.assign_scenes(records, seed=713101)
    second = split.assign_scenes(list(reversed(records)), seed=713101)

    assert first == second
    assert tuple(first) == tuple(sorted(first))
    assert len(first) == len(records)
    assert {metadata.scene for metadata in first.values()} == set(FORMAL_SCENARIOS)
    group_distributions = set()
    for group_index in range(12):
        group_ids = {
            f"group-{group_index}-sample-{sample_index}" for sample_index in range(8)
        }
        counts = Counter(first[sample_id].scene for sample_id in group_ids)
        assert max(counts.values()) - min(counts.values()) <= 1
        group_distributions.add(tuple(counts[scene] for scene in FORMAL_SCENARIOS))
        for scene in FORMAL_SCENARIOS:
            ranks = sorted(
                first[sample_id].scene_rank
                for sample_id in group_ids
                if first[sample_id].scene == scene
            )
            assert ranks == list(range(counts[scene]))
    assert len(group_distributions) > 1
    with pytest.raises(TypeError):
        first["new-id"] = next(iter(first.values()))
    with pytest.raises(FrozenInstanceError):
        next(iter(first.values())).scene = FORMAL_SCENARIOS[0]


def test_scene_assignment_rejects_duplicate_and_malformed_identity_rows():
    split = _split_module()
    valid = {
        "physical_sample_id": "sample-0",
        "tx_id": "tx-0",
        "rx_id": "rx-0",
        "day_id": "day-0",
    }
    with pytest.raises(ValueError, match="duplicate physical"):
        split.assign_scenes([valid, dict(valid)])
    for malformed in (
        {key: value for key, value in valid.items() if key != "day_id"},
        {**valid, "tx_id": ""},
        {**valid, "rx_id": None},
    ):
        with pytest.raises(ValueError, match="identity"):
            split.assign_scenes([malformed])


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_maxq_uses_every_non_support_record_with_exact_k_and_exclusions(
    canonical_split_context,
    k,
):
    context = canonical_split_context
    manifest = context["split"].build_split_manifest(
        context["connection"],
        context["profile"],
        k,
        "MAXQ_ALL_UNIQUE",
        scene_assignments=context["assignments"],
        class_selection=context["class_selection"],
    )

    assert manifest.eligible_receivers == ("rx-a",)
    assert manifest.registered_tx_ids == context["registered_tx_ids"]
    assert set(manifest.eligible_ids) == context["retained_ids"]
    assert set(manifest.support_ids).isdisjoint(manifest.query_ids)
    assert len(manifest.query_ids) == len(manifest.eligible_ids) - len(manifest.support_ids)
    support_counts = Counter(
        (row.rx_id, row.scene, row.tx_id)
        for row in manifest.rows
        if row.role == "support"
    )
    assert len(support_counts) == len(context["registered_tx_ids"]) * len(FORMAL_SCENARIOS)
    assert set(support_counts.values()) == {k}
    assert {row.role for row in manifest.rows} == {"support", "query"}
    assert {row.rx_id for row in manifest.rows} == {"rx-a"}
    assert {row.tx_id for row in manifest.rows} == set(context["registered_tx_ids"])
    assert "p-source-receiver" not in manifest.eligible_ids
    assert "p-unregistered-tx" not in manifest.eligible_ids
    assert "p-ineligible" not in manifest.eligible_ids


def test_support_is_nested_across_all_k_values(canonical_split_context):
    context = canonical_split_context
    manifests = {
        k: context["split"].build_split_manifest(
            context["connection"],
            context["profile"],
            k,
            "MAXQ_ALL_UNIQUE",
            scene_assignments=context["assignments"],
            class_selection=context["class_selection"],
        )
        for k in context["profile"].k_values
    }

    support_sets = {k: set(manifest.support_ids) for k, manifest in manifests.items()}
    assert support_sets[1] < support_sets[5] < support_sets[10] < support_sets[20]
    for k, manifest in manifests.items():
        assert {
            row.rank for row in manifest.rows if row.role == "support"
        } == set(range(k))


def test_balanced_four_day_core_has_one_frozen_cell_capacity_and_query_pool(
    canonical_split_context,
):
    context = canonical_split_context
    manifests = {
        k: context["split"].build_split_manifest(
            context["connection"],
            context["profile"],
            k,
            "BALANCED_4DAY_CORE",
            scene_assignments=context["assignments"],
            class_selection=context["class_selection"],
        )
        for k in context["profile"].k_values
    }

    query_sets = {k: set(manifest.query_ids) for k, manifest in manifests.items()}
    assert query_sets[1] == query_sets[5] == query_sets[10] == query_sets[20]
    query_cell_counts = Counter(
        (row.tx_id, row.rx_id, row.day_id, row.scene)
        for row in manifests[1].rows
        if row.role == "query"
    )
    assert len(query_cell_counts) == len(context["registered_tx_ids"]) * 4 * len(FORMAL_SCENARIOS)
    assert len(set(query_cell_counts.values())) == 1
    assert next(iter(query_cell_counts.values())) > 0

    k20_reserve = set(manifests[20].support_ids)
    for k, manifest in manifests.items():
        support = set(manifest.support_ids)
        query = set(manifest.query_ids)
        assert len(support) == len(context["registered_tx_ids"]) * len(FORMAL_SCENARIOS) * k
        assert support.isdisjoint(query)
        assert (k20_reserve - support).isdisjoint(query)
        assert manifest.eligible_receivers == ("rx-a",)


def test_capsule_and_split_ids_are_deterministic_and_sensitive(canonical_split_context):
    context = canonical_split_context
    kwargs = {
        "scene_assignments": context["assignments"],
        "class_selection": context["class_selection"],
    }
    first = context["split"].build_split_manifest(
        context["connection"], context["profile"], 5, "MAXQ_ALL_UNIQUE", **kwargs
    )
    repeated = context["split"].build_split_manifest(
        context["connection"], context["profile"], 5, "MAXQ_ALL_UNIQUE", **kwargs
    )
    k10 = context["split"].build_split_manifest(
        context["connection"], context["profile"], 10, "MAXQ_ALL_UNIQUE", **kwargs
    )
    balanced = context["split"].build_split_manifest(
        context["connection"], context["profile"], 5, "BALANCED_4DAY_CORE", **kwargs
    )
    changed_support = context["split"].build_split_manifest(
        context["connection"],
        context["profile"],
        5,
        "MAXQ_ALL_UNIQUE",
        support_seed=713102,
        **kwargs,
    )
    changed_assignments = context["split"].assign_scenes(
        context["eligible_records"], seed=713102
    )
    changed_plain = {
        sample_id: metadata.scene for sample_id, metadata in changed_assignments.items()
    }
    changed_selection = context["split"].rank_new_classes(
        context["connection"], context["profile"], changed_plain
    )
    changed_scene = context["split"].build_split_manifest(
        context["connection"],
        context["profile"],
        5,
        "MAXQ_ALL_UNIQUE",
        scene_assignments=changed_assignments,
        class_selection=changed_selection,
    )

    assert first == repeated
    assert first.capsule_id == k10.capsule_id == balanced.capsule_id == changed_support.capsule_id
    assert len({first.split_id, k10.split_id, balanced.split_id, changed_support.split_id}) == 4
    assert changed_scene.capsule_id != first.capsule_id
    assert changed_scene.split_id != first.split_id


def test_split_rejects_undeclared_k_policy_and_drifted_class_selection(
    canonical_split_context,
):
    context = canonical_split_context
    base_kwargs = {
        "scene_assignments": context["assignments"],
        "class_selection": context["class_selection"],
    }
    with pytest.raises(ValueError, match="K"):
        context["split"].build_split_manifest(
            context["connection"], context["profile"], 2, "MAXQ_ALL_UNIQUE", **base_kwargs
        )
    with pytest.raises(ValueError, match="policy"):
        context["split"].build_split_manifest(
            context["connection"], context["profile"], 1, "UNDECLARED", **base_kwargs
        )
    drifted = dict(context["class_selection"])
    drifted[20] = tuple(reversed(drifted[20]))
    with pytest.raises(ValueError, match="class_selection"):
        context["split"].build_split_manifest(
            context["connection"],
            context["profile"],
            1,
            "MAXQ_ALL_UNIQUE",
            scene_assignments=context["assignments"],
            class_selection=drifted,
        )


@pytest.fixture
def query_truth_serialization_fixture():
    split = _split_module()
    sentinel = "QUERY_TRUTH_SENTINEL_7F3A9C"
    manifest = split.SplitManifest(
        protocol_schema="p2_min_v1",
        profile_id="serialization-test",
        query_policy="MAXQ_ALL_UNIQUE",
        k=1,
        registered_tx_ids=("authorized-support-label", sentinel),
        eligible_receivers=("rx-a",),
        rows=(
            split.SplitRow(
                physical_sample_id="support-id",
                source_asset="ManyTx",
                source_record_index=10,
                tx_id="authorized-support-label",
                rx_id="rx-a",
                day_id="day-0",
                scene=FORMAL_SCENARIOS[0],
                role="support",
                rank=0,
            ),
            split.SplitRow(
                physical_sample_id="opaque-query-id",
                source_asset="ManyTx",
                source_record_index=11,
                tx_id=sentinel,
                rx_id="rx-a",
                day_id="day-0",
                scene=FORMAL_SCENARIOS[0],
                role="query",
                rank=1,
            ),
        ),
        capsule_id="capsule-id",
        split_id="split-id",
    )
    return manifest, sentinel


def test_manifest_serialization_keeps_authorized_support_tx_id(
    query_truth_serialization_fixture,
):
    manifest, _ = query_truth_serialization_fixture
    support_row = next(
        row for row in manifest.to_mapping()["rows"] if row["role"] == "support"
    )

    assert support_row["tx_id"] == "authorized-support-label"
    assert set(support_row) == {
        "physical_sample_id",
        "source_asset",
        "source_record_index",
        "tx_id",
        "rx_id",
        "day_id",
        "scene",
        "role",
        "rank",
    }


def test_manifest_serialization_omits_query_truth_and_unique_sentinel(
    query_truth_serialization_fixture,
):
    manifest, sentinel = query_truth_serialization_fixture
    query_row = next(
        row for row in manifest.to_mapping()["rows"] if row["role"] == "query"
    )

    assert set(query_row) == {
        "physical_sample_id",
        "source_asset",
        "source_record_index",
        "rx_id",
        "day_id",
        "scene",
        "role",
        "rank",
    }
    assert sentinel not in json.dumps(query_row, sort_keys=True)
