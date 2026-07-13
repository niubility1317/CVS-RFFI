from paper_reproduction.scripts.evaluate_cvs_phase1_ssdg_detailed import (
    _checkpoint_with_dataset_override,
    parse_sat_scenarios,
)


def test_ssdg_detailed_eval_formal_scenarios_are_registered() -> None:
    assert tuple(
        parse_sat_scenarios("leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    ) == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def test_dataset_override_changes_only_checkpoint_dataset_path(tmp_path) -> None:
    dataset = tmp_path / "ManySig.pkl"
    dataset.write_bytes(b"fixture")
    original = {"args": {"wisig_pkl": "/remote/ManySig.pkl", "seed": 7}, "epoch": 3}
    updated, original_path = _checkpoint_with_dataset_override(original, str(dataset))
    assert original_path == "/remote/ManySig.pkl"
    assert updated["args"]["wisig_pkl"] == str(dataset.resolve())
    assert updated["args"]["seed"] == 7
    assert updated["epoch"] == 3
    assert original["args"]["wisig_pkl"] == "/remote/ManySig.pkl"
