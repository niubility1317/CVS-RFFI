from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.somph_cache_build_matrix import (
    FIXED_N607_CACHE_OUTPUT_ROOT,
    REQUIRED_SAMPLES_PER_TX,
    TOTAL_REQUIRED_SAMPLES_PER_TX,
    SomphCacheBuildMatrixError,
    validate_cache_build_manifest,
    validate_registered_cache_coverage,
    validate_cache_spec,
    write_cache_build_matrix,
)
from cvsrffi.leo_weak_cache import (
    LEO_WEAK_CACHE_SET_SCHEMA,
    canonical_json_sha256,
    ids_sha256,
    physical_sample_id_from_values,
)
from cvsrffi.somph_formal_matrix import (
    CONFIRMATION_SEEDS,
    FORMAL_RECEIVERS,
    NEW_TX_IDS,
    OLD_TX_IDS,
)


def _resign(payload: dict) -> None:
    from cvsrffi.somph_cache_build_matrix import _canonical_sha256

    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    payload["manifest_sha256"] = _canonical_sha256(unsigned)


def _build(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "somph_specs"
    manifest = write_cache_build_matrix(
        output_root=root,
        manysig_pkl="/datasets/ManySig.pkl",
        manytx_pkl="/datasets/ManyTx.pkl",
    )
    return root, manifest


def _write_cache_set(
    root: Path,
    *,
    receiver: str = "20-1",
    count_override: dict[tuple[str, str], int] | None = None,
    drift_scenario_ids: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    hashes: dict[str, str] = {}
    ids_by_scenario: dict[str, list[str]] = {}
    first_scenario_first_row: dict[str, object] | None = None
    scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    for scenario_index, scenario in enumerate(scenarios):
        scenario_rows: list[dict[str, object]] = []
        serial = scenario_index * 10_000
        for role, tx_ids in (("target_old", OLD_TX_IDS), ("target_new", NEW_TX_IDS)):
            dataset_sha = ("a" if role == "target_old" else "b") * 64
            for tx_id in tx_ids:
                count = (count_override or {}).get((role, tx_id), 40)
                for rank in range(count):
                    day_id = str((rank + scenario_index) % 3)
                    sig_id = str(serial)
                    sample_id = physical_sample_id_from_values(
                        dataset_sha256=dataset_sha,
                        source_record_index=serial,
                        role=role,
                        tx_id=tx_id,
                        rx_id=receiver,
                        day_id=day_id,
                        eq_id="1",
                        sig_id=sig_id,
                    )
                    scenario_rows.append(
                        {
                            "role": role,
                            "tx_id": tx_id,
                            "rx_id": receiver,
                            "day_id": day_id,
                            "eq_id": "1",
                            "sig_id": sig_id,
                            "dataset_sha256": dataset_sha,
                            "source_record_index": serial,
                            "sample_id": sample_id,
                        }
                    )
                    serial += 1
        if scenario_index == 0:
            first_scenario_first_row = dict(scenario_rows[0])
        if scenario == drift_scenario_ids:
            assert first_scenario_first_row is not None
            scenario_rows[0] = dict(first_scenario_first_row)
        path = root / f"{scenario}.npz"
        np.savez(
            path,
            leo_weak_iq=np.zeros((len(scenario_rows), 2, 1), dtype=np.float32),
            dataset_role=np.asarray([row["role"] for row in scenario_rows]),
            tx_ids=np.asarray([row["tx_id"] for row in scenario_rows]),
            rx_ids=np.asarray([row["rx_id"] for row in scenario_rows]),
            day_ids=np.asarray([row["day_id"] for row in scenario_rows]),
            eq_ids=np.asarray([row["eq_id"] for row in scenario_rows]),
            sig_ids=np.asarray([row["sig_id"] for row in scenario_rows]),
            source_dataset_sha256=np.asarray(
                [row["dataset_sha256"] for row in scenario_rows]
            ),
            source_record_indices=np.asarray(
                [row["source_record_index"] for row in scenario_rows],
                dtype=np.int64,
            ),
            sample_ids=np.asarray([row["sample_id"] for row in scenario_rows]),
            sat_scenarios=np.asarray([scenario] * len(scenario_rows)),
        )
        hashes[scenario] = hashlib.sha256(path.read_bytes()).hexdigest()
        ids_by_scenario[scenario] = [
            str(row["sample_id"]) for row in scenario_rows
        ]
    payload = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "cache_scope": "stage2_registered",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_equalization_augmentation_transform_allowed": True,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
        "phase2_physical_sample_root_id_policy": (
            "immutable_preoverlay_lineage_token"
        ),
        "phase2_query_post_reception_view_fit_access": False,
        "physical_sample_scenario_assignment_policy": (
            "disjoint_preoverlay_tx_day_stratified_v1"
        ),
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": [
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        ],
        "output_roles": ["target_old", "target_new"],
        "cache_npz_by_scenario": {
            scenario: f"{scenario}.npz"
            for scenario in (
                "leo_clear_weak",
                "leo_low_elev_weak",
                "leo_rain_weak",
            )
        },
        "cache_sha256_by_scenario": hashes,
        "physical_sample_ids_sha256_by_scenario": {
            scenario: ids_sha256(ids_by_scenario[scenario])
            for scenario in scenarios
        },
        "physical_sample_scenario_assignment_sha256": canonical_json_sha256(
            ids_by_scenario
        ),
    }
    manifest = root / "cache_set.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_writes_exact_30_cell_registered_maxk20_matrix(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path)
    assert manifest["cell_count"] == 30
    assert manifest["receivers"] == list(FORMAL_RECEIVERS)
    assert manifest["confirmation_seeds"] == list(CONFIRMATION_SEEDS)
    assert manifest["seeds"] == [713101, 713102, 713103, 713104, 713105, 713106]
    assert manifest["cache_scope"] == "stage2_registered"
    assert manifest["cache_output_root"] == FIXED_N607_CACHE_OUTPUT_ROOT
    assert manifest["support_pool_max_k"] == 20
    assert manifest["query_samples_per_tx"] == 20
    assert manifest["required_samples_per_tx"] == 40
    assert manifest["required_physical_samples_per_tx_all_scenarios"] == 120
    assert manifest["cross_scenario_physical_sample_reuse"] is False
    assert manifest["estimated_rows_per_scenario"] == 26 * 40
    assert manifest["estimated_rows_all_scenarios_per_cell"] == 26 * 40 * 3
    assert manifest["disk_budget_bytes_total"] == (
        manifest["disk_budget_bytes_per_cell"] * 30
    )
    assert manifest["control_status"] == "LOCAL_PROTOCOL_REPAIR_REQUIRED"
    assert manifest["formal_launch_authority"] is False
    assert manifest["ssh_performed"] is False
    assert manifest["cache_builder_executed"] is False
    assert manifest["post_build_coverage_required"] is True
    assert (
        manifest["post_build_coverage_status"]
        == "NOT_RUN_BLOCKS_FORMAL_LAUNCH"
    )
    assert (root / "manifest.json").is_file()

    output_roots = {cell["cache_output_root"] for cell in manifest["cells"]}
    assert len(output_roots) == 30
    for cell in manifest["cells"]:
        spec_path = root / cell["spec_path"]
        raw = spec_path.read_bytes()
        spec = json.loads(raw.decode("utf-8"))
        assert hashlib.sha256(raw).hexdigest() == cell["spec_file_sha256"]
        assert cell["spec_canonical_sha256"] != cell["spec_file_sha256"]
        assert spec["cache_scope"] == "stage2_registered"
        assert [role["role"] for role in spec["role_specs"]] == [
            "target_old",
            "target_new",
        ]
        assert spec["role_specs"][0]["tx_ids"] == ",".join(OLD_TX_IDS)
        assert spec["role_specs"][1]["tx_ids"] == ",".join(NEW_TX_IDS)
        assert all(
            role["max_samples_per_tx"] == TOTAL_REQUIRED_SAMPLES_PER_TX
            for role in spec["role_specs"]
        )
        assert all(role["days"] == "0,1,2" for role in spec["role_specs"])
        assert list(spec["satellite_seed_by_scenario"]) == [
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        ]
        assert spec["clean_sample_access"] is False
        assert spec["clean_derived_signal_access"] is False
        assert str(spec["out_manifest"]).startswith(FIXED_N607_CACHE_OUTPUT_ROOT)


def test_explicit_cache_output_root_is_bound_into_all_30_exact_cells(
    tmp_path: Path,
) -> None:
    root = tmp_path / "custom_specs"
    cache_output_root = "/offline/formal/somph-cache-v2"
    manifest = write_cache_build_matrix(
        output_root=root,
        manysig_pkl="/datasets/ManySig.pkl",
        manytx_pkl="/datasets/ManyTx.pkl",
        cache_output_root=cache_output_root,
    )

    assert manifest["cell_count"] == 30
    assert manifest["cache_output_root"] == cache_output_root
    validate_cache_build_manifest(manifest, manifest_root=root)
    for cell in manifest["cells"]:
        expected_root = (
            f"{cache_output_root}/rx_{cell['receiver'].replace('-', '_')}"
            f"/seed_{cell['seed']}"
        )
        assert cell["cache_output_root"] == expected_root
        spec = json.loads(
            (root / cell["spec_path"]).read_text(encoding="utf-8")
        )
        assert spec["out_manifest"] == f"{expected_root}/cache_set.json"


def test_all_specs_are_accepted_by_the_real_leo_weak_cache_builder(
    tmp_path: Path,
) -> None:
    from scripts import build_cvs_leo_weak_iq_cache as real_builder

    root, manifest = _build(tmp_path)
    for cell in manifest["cells"]:
        spec = json.loads(
            (root / cell["spec_path"]).read_text(encoding="utf-8")
        )
        checked = real_builder.validate_build_spec(spec)
        assert checked["cache_scope"] == "stage2_registered"
        assert [role["role"] for role in checked["role_specs"]] == [
            "target_old",
            "target_new",
        ]


def test_only_locked_development_cell_has_k10_selection_eligibility(
    tmp_path: Path,
) -> None:
    _, manifest = _build(tmp_path)
    eligible = [
        (
            cell["receiver"],
            cell["seed"],
            cell["development_selection_k_shot"],
        )
        for cell in manifest["cells"]
        if cell["development_selection_eligible"]
    ]
    assert eligible == [("20-1", 713101, 10)]
    assert all(
        cell["nondevelopment_selection_authority"] is False
        for cell in manifest["cells"]
    )
    assert sum(cell["seed_role"] == "development" for cell in manifest["cells"]) == 5
    assert (
        sum(
            cell["seed_role"] == "independent_confirmation"
            for cell in manifest["cells"]
        )
        == 25
    )


def test_all_90_receiver_seed_scenario_base_seeds_are_unique(
    tmp_path: Path,
) -> None:
    root, manifest = _build(tmp_path)
    values: list[int] = []
    for cell in manifest["cells"]:
        spec = json.loads((root / cell["spec_path"]).read_text(encoding="utf-8"))
        values.extend(spec["satellite_seed_by_scenario"].values())
    assert len(values) == 90
    assert len(set(values)) == 90


def test_manifest_and_spec_reject_bool_as_integer(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path)
    broken_manifest = copy.deepcopy(manifest)
    broken_manifest["cells"][0]["seed"] = True
    _resign(broken_manifest)
    with pytest.raises(SomphCacheBuildMatrixError, match="booleans are forbidden"):
        validate_cache_build_manifest(broken_manifest)

    spec = json.loads((root / manifest["cells"][0]["spec_path"]).read_text())
    spec["role_specs"][0]["max_samples_per_tx"] = True
    with pytest.raises(SomphCacheBuildMatrixError, match="booleans are forbidden"):
        validate_cache_spec(spec, receiver="20-1", seed=713101)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manysig_pkl", "/datasets/clean/ManySig.pkl"),
        ("manytx_pkl", "/datasets/raw/ManyTx.pkl"),
        ("manysig_pkl", "/datasets/phase2/ManySig.pkl"),
        ("manytx_pkl", "/datasets/predictor/ManyTx.pkl"),
        ("manysig_pkl", "/datasets/scorer/ManySig.pkl"),
        ("manytx_pkl", "/datasets/package/ManyTx.pkl"),
    ],
)
def test_rejects_forbidden_dataset_paths(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    kwargs = {
        "output_root": tmp_path / f"plan_{field}",
        "manysig_pkl": "/datasets/ManySig.pkl",
        "manytx_pkl": "/datasets/ManyTx.pkl",
    }
    kwargs[field] = value
    with pytest.raises(SomphCacheBuildMatrixError, match="forbidden path token"):
        write_cache_build_matrix(**kwargs)


def test_rejects_caller_dataset_family_and_overwrite(tmp_path: Path) -> None:
    with pytest.raises(SomphCacheBuildMatrixError, match="fixed ManySig.pkl"):
        write_cache_build_matrix(
            output_root=tmp_path / "wrong_dataset",
            manysig_pkl="/datasets/Other.pkl",
            manytx_pkl="/datasets/ManyTx.pkl",
        )
    root, _ = _build(tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_cache_build_matrix(
            output_root=root,
            manysig_pkl="/datasets/ManySig.pkl",
            manytx_pkl="/datasets/ManyTx.pkl",
        )


def test_manifest_rejects_extra_cell_selection_authority(tmp_path: Path) -> None:
    _, manifest = _build(tmp_path)
    broken = copy.deepcopy(manifest)
    confirmation = next(cell for cell in broken["cells"] if cell["seed"] == 713102)
    confirmation["development_selection_eligible"] = True
    confirmation["development_selection_k_shot"] = 10
    _resign(broken)
    with pytest.raises(
        SomphCacheBuildMatrixError, match="development selection eligibility drift"
    ):
        validate_cache_build_manifest(broken)


def test_spec_rejects_forbidden_extra_fields(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path)
    spec = json.loads((root / manifest["cells"][0]["spec_path"]).read_text())
    spec["clean_cache_path"] = "/runtime/cache"
    with pytest.raises(SomphCacheBuildMatrixError, match="exact schema drift"):
        validate_cache_spec(spec, receiver="20-1", seed=713101)
    spec.pop("clean_cache_path")
    spec["raw_dataset_path"] = "/datasets/ManySig.pkl"
    with pytest.raises(SomphCacheBuildMatrixError, match="exact schema drift"):
        validate_cache_spec(spec, receiver="20-1", seed=713101)


def test_post_build_gate_accepts_exact40_for_all_roles_txs_and_scenarios(
    tmp_path: Path,
) -> None:
    manifest = _write_cache_set(tmp_path / "coverage_ok")
    audit = validate_registered_cache_coverage(
        manifest.resolve(),
        expected_receiver="20-1",
    )
    assert audit["coverage_pass"] is True
    assert audit["row_count_per_scenario"] == 1040
    assert audit["exact_rows_per_role_tx_receiver"] == 40
    assert len(audit["scenario_audits"]) == 3


def test_post_build_gate_rejects_39_rows_for_one_tx_even_when_total_is_1040(
    tmp_path: Path,
) -> None:
    manifest = _write_cache_set(
        tmp_path / "coverage_39",
        count_override={
            ("target_old", OLD_TX_IDS[0]): 39,
            ("target_old", OLD_TX_IDS[1]): 41,
        },
    )
    with pytest.raises(SomphCacheBuildMatrixError, match="exactly 40"):
        validate_registered_cache_coverage(
            manifest.resolve(),
            expected_receiver="20-1",
        )


def test_post_build_gate_rejects_cross_scenario_physical_id_overlap(
    tmp_path: Path,
) -> None:
    manifest = _write_cache_set(
        tmp_path / "coverage_drift",
        drift_scenario_ids="leo_rain_weak",
    )
    with pytest.raises(
        SomphCacheBuildMatrixError,
        match="PROTOCOL_INVALID_FOR_PHASE2_SINGLE_OBSERVATION",
    ):
        validate_registered_cache_coverage(
            manifest.resolve(),
            expected_receiver="20-1",
        )


def test_post_build_gate_rejects_relative_escape_npz_path(tmp_path: Path) -> None:
    manifest = _write_cache_set(tmp_path / "coverage_escape")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cache_npz_by_scenario"]["leo_clear_weak"] = "../leo_clear_weak.npz"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SomphCacheBuildMatrixError, match="exact sibling"):
        validate_registered_cache_coverage(
            manifest.resolve(),
            expected_receiver="20-1",
        )


def test_post_build_gate_rejects_symlinked_npz(tmp_path: Path) -> None:
    manifest = _write_cache_set(tmp_path / "coverage_link")
    target = manifest.parent / "leo_clear_weak.npz"
    replacement = manifest.parent / "replacement.npz"
    target.replace(replacement)
    try:
        target.symlink_to(replacement)
    except OSError:
        replacement.replace(target)
        pytest.skip("symlink creation is unavailable on this Windows host")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cache_sha256_by_scenario"]["leo_clear_weak"] = hashlib.sha256(
        replacement.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SomphCacheBuildMatrixError, match="symlink component"):
        validate_registered_cache_coverage(
            manifest.resolve(),
            expected_receiver="20-1",
        )


def test_output_root_must_be_absolute_and_not_a_symlink(tmp_path: Path) -> None:
    with pytest.raises(SomphCacheBuildMatrixError, match="must be absolute"):
        write_cache_build_matrix(
            output_root=Path("relative/specs"),
            manysig_pkl="/datasets/ManySig.pkl",
            manytx_pkl="/datasets/ManyTx.pkl",
        )
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(SomphCacheBuildMatrixError, match="symlink component"):
        write_cache_build_matrix(
            output_root=linked_parent / "specs",
            manysig_pkl="/datasets/ManySig.pkl",
            manytx_pkl="/datasets/ManyTx.pkl",
        )


def test_cli_only_exposes_cache_root_not_custom_formal_axes_and_writes_specs(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "build_cvs_somph_cache_specs.py"
    )
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for forbidden in (
        "--receiver",
        "--seed",
        "--scenario",
        "--tx-ids",
        "--k-shot",
    ):
        assert forbidden not in help_result.stdout
    assert "--cache-output-root" in help_result.stdout

    output = tmp_path / "cli_specs"
    cache_output_root = "/offline/formal/cli-cache"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(output),
            "--manysig-pkl",
            "/datasets/ManySig.pkl",
            "--manytx-pkl",
            "/datasets/ManyTx.pkl",
            "--cache-output-root",
            cache_output_root,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["cell_count"] == 30
    assert summary["formal_launch_authority"] is False
    assert summary["control_status"] == "LOCAL_PROTOCOL_REPAIR_REQUIRED"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_output_root"] == cache_output_root
    validate_cache_build_manifest(manifest, manifest_root=output)
