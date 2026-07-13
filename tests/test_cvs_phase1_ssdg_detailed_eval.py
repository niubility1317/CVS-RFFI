from paper_reproduction.scripts.evaluate_cvs_phase1_ssdg_detailed import (
    _checkpoint_with_dataset_override,
    _metadata_from_extra,
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


def test_metadata_from_standard_four_field_wisig_batch() -> None:
    metadata = {"rx_i": [1, 2], "day_i": [0, 1], "sig_i": [7, 8]}
    assert _metadata_from_extra(([10, 11], metadata)) is metadata


def test_metadata_from_extra_rejects_incomplete_mapping() -> None:
    try:
        _metadata_from_extra(({"rx_i": [1]},))
    except KeyError as exc:
        assert "day_i" in str(exc) and "sig_i" in str(exc)
    else:
        raise AssertionError("incomplete WiSig metadata must be rejected")
