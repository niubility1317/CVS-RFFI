from cvsrffi.jmrs02_j1_scoring import score_j1_records


def _row(sample, row, scenario, truth, base, candidate, final, selected=False, nuisance=None, target=None):
    prediction = {
        "sample_id": f"{row}:{scenario}:{sample}", "row": row, "scenario": scenario,
        "held_receiver": 0, "receiver": 0, "day": 0,
        "base_predicted_class": base, "candidate_predicted_class": candidate,
        "final_predicted_class": final, "gate_selected": selected, "gate_utility": 0.2,
        "parameter_count": 10, "target_or_query_access": False,
    }
    if nuisance is not None:
        prediction["nuisance_prediction"] = nuisance
        prediction["nuisance_target_proxy"] = target
    return prediction, {"sample_id": prediction["sample_id"], "true_class": truth}


def test_scorer_keeps_same_row_metrics_and_rescue_harm():
    predictions, truths = [], []
    for item in (
        _row(1, "B0", "clean", 1, 1, 1, 1),
        _row(2, "B0", "clean", 1, 0, 0, 0),
        _row(1, "D1P", "clean", 1, 1, 1, 1, True),
        _row(2, "D1P", "clean", 1, 0, 1, 1, True),
    ):
        predictions.append(item[0]); truths.append(item[1])
    result = score_j1_records(predictions, truths)
    metric = result["metrics"]["D1P"]["clean"]
    assert metric["final_accuracy"] == 1.0
    assert metric["rescue_count"] == 1
    assert metric["harm_count"] == 0
    assert metric["gate_coverage"] == 1.0


def test_p0_reports_proxy_improvement_without_tx_gain_claim():
    prediction, truth = _row(1, "P0", "leo_rain_weak", 1, 1, 1, 1, nuisance=[0.5, 0, 0, 0], target=[1, 0, 0, 0])
    result = score_j1_records([prediction], [truth])
    assert result["nuisance"]["P0"]["mae"] == 0.125
    assert result["nuisance"]["P0"]["zero_baseline_mae"] == 0.25
    assert result["nuisance"]["P0"]["tx_residual_claim"] is False
