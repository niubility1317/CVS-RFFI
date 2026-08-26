from cvsrffi.jmrs02_rx2_scoring import score_rx2_records


SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _record(row, scenario, sample, truth, base, candidate, final, selected=False):
    sample_id = f"{row}:{scenario}:{sample}"
    return ({
        "sample_id": sample_id,
        "row": row,
        "scenario": scenario,
        "receiver": sample % 2,
        "base_predicted_class": base,
        "candidate_predicted_class": candidate,
        "final_predicted_class": final,
        "gate_selected": selected,
        "target_or_query_access": False,
    }, {"sample_id": sample_id, "true_class": truth})


def _matrix(rx0_gain=False, rx2_gain=True):
    predictions, truths = [], []
    for scenario in SCENARIOS:
        for sample in range(4):
            truth = 1
            base = 1 if sample < 3 else 0
            for row in ("B0", "RX0", "RX2"):
                improve = scenario != "clean" and sample == 3 and (
                    (row == "RX0" and rx0_gain) or (row == "RX2" and rx2_gain)
                )
                candidate = truth if improve else base
                final = candidate
                item = _record(row, scenario, sample, truth, base, candidate, final, selected=improve)
                predictions.append(item[0]); truths.append(item[1])
    return predictions, truths


def test_rx2_must_beat_both_core90_and_same_capacity_rx0():
    predictions, truths = _matrix(rx0_gain=False, rx2_gain=True)
    result = score_rx2_records(predictions, truths)
    decision = result["decision"]["RX2"]
    assert decision["gain_vs_b0_pp"] > 0.0
    assert decision["gain_vs_rx0_pp"] > 0.0
    assert decision["passes_rx2"] is True


def test_rx2_does_not_pass_when_same_capacity_control_matches_it():
    predictions, truths = _matrix(rx0_gain=True, rx2_gain=True)
    result = score_rx2_records(predictions, truths)
    decision = result["decision"]["RX2"]
    assert decision["gain_vs_b0_pp"] > 0.0
    assert decision["gain_vs_rx0_pp"] == 0.0
    assert decision["passes_rx2"] is False

