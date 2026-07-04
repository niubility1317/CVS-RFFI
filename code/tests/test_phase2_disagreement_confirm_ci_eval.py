from scripts.phase2_disagreement_confirm_ci_eval import PROFILES, augment_darc_evidence


def _row(event_id: str, receiver: str, label: str, *, score: float, margin: float, risk: float = 0.05):
    return {
        "event_id": event_id,
        "receiver_id": receiver,
        "role": "old",
        "true_label": "old-a",
        "predicted_label": label,
        "class_evidence_top1_label": label,
        "known_score": str(score),
        "known_margin": str(margin),
        "support_density": "0.6",
        "receiver_class_reliability": "0.8",
        "unknown_risk": str(risk),
        "socapr_safety_route_unknown_risk": "0.8",
    }


def _profile(name: str):
    return next(profile for profile in PROFILES if profile.name == name)


def test_darc_keeps_base_label_authority():
    rows = [
        _row("e0", "rx0", "old-a", score=0.2, margin=0.01),
        _row("e0", "rx1", "old-b", score=0.2, margin=0.01),
    ]

    out = augment_darc_evidence(rows, _profile("darc_balanced"), old_labels={"old-a", "old-b"})

    assert [row["predicted_label"] for row in out] == ["old-a", "old-b"]
    assert all(row["darc_label_authority"] == "base_qknn_only" for row in out)


def test_darc_caps_strong_old_candidate():
    rows = [
        _row("e0", "rx0", "old-a", score=0.9, margin=0.3, risk=0.7),
        _row("e0", "rx1", "old-a", score=0.9, margin=0.3, risk=0.7),
    ]

    out = augment_darc_evidence(rows, _profile("darc_light"), old_labels={"old-a"})

    assert all(row["darc_strong_old_candidate"] == 1 for row in out)
    assert all(row["unknown_risk"] <= 0.30 for row in out)


def test_darc_disagreement_raises_weak_unknown_risk():
    rows = [
        _row("e0", "rx0", "old-a", score=0.2, margin=0.01, risk=0.05),
        _row("e0", "rx1", "old-b", score=0.2, margin=0.01, risk=0.05),
        _row("e0", "rx2", "new-c", score=0.2, margin=0.01, risk=0.05),
    ]

    out = augment_darc_evidence(rows, _profile("darc_unknown_push"), old_labels={"old-a", "old-b"})

    assert all(row["darc_label_disagreement"] > 0.0 for row in out)
    assert max(row["unknown_risk"] for row in out) > 0.5
