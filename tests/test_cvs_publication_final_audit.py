from tools.validate_cvs_publication_comparison import DETAIL_LEVELS, SCENARIOS


def test_final_audit_requires_all_formal_scenarios_and_detail_levels() -> None:
    assert SCENARIOS == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    assert "per_receiver_transmitter_day" in DETAIL_LEVELS
    assert "per_receiver" in DETAIL_LEVELS
    assert "per_transmitter" in DETAIL_LEVELS
