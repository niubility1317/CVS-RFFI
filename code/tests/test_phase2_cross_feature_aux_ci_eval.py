from scripts.phase2_cross_feature_aux_ci_eval import XfaConfig, build_xfa_evidence


def _base_row(event_id: str, *, score: float, margin: float, risk: float = 0.1):
    return {
        "event_id": event_id,
        "receiver_id": "rx-a",
        "role": "old",
        "true_label": "tx-old",
        "predicted_label": "tx-old",
        "known_score": str(score),
        "known_margin": str(margin),
        "support_density": "0.5",
        "reliability": "0.8",
        "unknown_risk": str(risk),
        "bytes": "168",
        "latency_ms": "0.2",
    }


def _aux_row(event_id: str, *, risk: float):
    return {
        "event_id": event_id,
        "receiver_id": "rx-a",
        "role": "old",
        "true_label": "tx-old",
        "predicted_label": "tx-other",
        "unknown_risk": str(risk),
        "bytes": "168",
        "latency_ms": "0.4",
    }


def _config(**overrides):
    values = {
        "aux_weight": 1.0,
        "strong_score": 0.55,
        "strong_margin": 0.08,
        "strong_support_density": 0.35,
        "strong_reliability": 0.60,
        "weak_score_anchor": 0.70,
        "weak_margin_anchor": 0.25,
        "weak_support_anchor": 0.45,
        "weak_reliability_anchor": 0.70,
        "strong_aux_cap": 0.15,
        "aux_bytes_per_receiver": 16.0,
    }
    values.update(overrides)
    return XfaConfig(**values)


def test_xfa_uses_same_matched_subset_and_reports_missing_aux_rows():
    base_rows = [
        _base_row("e0", score=0.3, margin=0.02),
        _base_row("e1", score=0.3, margin=0.02),
    ]
    aux_rows = [_aux_row("e0", risk=0.9)]

    paired_base, xfa_rows, audit = build_xfa_evidence(base_rows, aux_rows, _config())

    assert [row["event_id"] for row in paired_base] == ["e0"]
    assert [row["event_id"] for row in xfa_rows] == ["e0"]
    assert audit["matched_row_count"] == 1
    assert audit["missing_aux_row_count"] == 1
    assert audit["same_subset"] is True


def test_xfa_preserves_base_label_authority():
    base_rows = [_base_row("e0", score=0.3, margin=0.02)]
    aux_rows = [_aux_row("e0", risk=0.9)]

    _, xfa_rows, _ = build_xfa_evidence(base_rows, aux_rows, _config())

    assert xfa_rows[0]["predicted_label"] == "tx-old"
    assert xfa_rows[0]["xfa_label_authority"] == "base_qknn_only"
    assert xfa_rows[0]["unknown_risk"] > float(base_rows[0]["unknown_risk"])


def test_strong_known_rows_cap_aux_component_without_lowering_base_risk():
    base_rows = [_base_row("e0", score=0.9, margin=0.3, risk=0.4)]
    aux_rows = [_aux_row("e0", risk=1.0)]

    _, xfa_rows, audit = build_xfa_evidence(
        base_rows,
        aux_rows,
        _config(strong_aux_cap=0.05),
    )

    assert audit["strong_known_row_count"] == 1
    assert xfa_rows[0]["xfa_base_strong_known"] == 1
    assert xfa_rows[0]["unknown_risk"] == 0.4
    assert xfa_rows[0]["xfa_aux_component"] <= 0.05
