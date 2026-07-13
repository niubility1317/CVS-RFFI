from paper_reproduction.scripts.evaluate_cvs_phase1_ssdg_detailed import parse_sat_scenarios


def test_ssdg_detailed_eval_formal_scenarios_are_registered() -> None:
    assert tuple(
        parse_sat_scenarios("leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    ) == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
