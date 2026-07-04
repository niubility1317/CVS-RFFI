import numpy as np

from scripts.phase2_support_metric_energy_ci_eval import (
    SmecConfig,
    build_support_model,
    augment_smec_evidence,
)


def _row(
    event_id: str,
    *,
    score: float,
    margin: float,
    risk: float = 0.05,
    label: str = "old-a",
    role: str = "old",
    true_label: str = "old-a",
):
    return {
        "event_id": event_id,
        "receiver_id": "rx-a",
        "role": role,
        "true_label": true_label,
        "predicted_label": label,
        "class_evidence_top1_label": label,
        "known_score": str(score),
        "known_margin": str(margin),
        "support_density": "0.6",
        "receiver_class_reliability": "0.8",
        "unknown_risk": str(risk),
        "bytes": "40",
        "latency_ms": "0.2",
    }


def _config(**overrides):
    values = {
        "proto_weight": 0.70,
        "knn_weight": 0.50,
        "energy_weight": 0.0,
        "proto_temperature": 0.03,
        "knn_temperature": 0.03,
        "energy_temperature": 0.20,
        "proto_quantile": 0.90,
        "knn_quantile": 0.90,
        "proto_slack": 0.01,
        "knn_slack": 0.01,
        "strong_score": 0.60,
        "strong_margin": 0.08,
        "strong_support_density": 0.35,
        "strong_reliability": 0.60,
        "weak_score_anchor": 0.70,
        "weak_margin_anchor": 0.20,
        "weak_support_anchor": 0.45,
        "weak_reliability_anchor": 0.70,
        "strong_aux_cap": 0.15,
        "aux_bytes_per_receiver": 24.0,
        "aux_latency_ms": 0.03,
    }
    values.update(overrides)
    return SmecConfig(**values)


def _support_model(config=None):
    cfg = config or _config()
    support_features = np.asarray(
        [
            [1.00, 0.00],
            [0.99, 0.04],
            [0.98, -0.04],
        ],
        dtype=np.float32,
    )
    return build_support_model(
        receiver_id="rx-a",
        support_features=support_features,
        support_labels=["old-a", "old-a", "old-a"],
        support_logits=None,
        old_labels={"old-a"},
        config=cfg,
    )


def _two_class_support_model(config=None):
    cfg = config or _config()
    support_features = np.asarray(
        [
            [1.00, 0.00],
            [0.98, 0.05],
            [0.98, -0.05],
            [0.00, 1.00],
            [0.05, 0.98],
            [-0.05, 0.98],
        ],
        dtype=np.float32,
    )
    return build_support_model(
        receiver_id="rx-a",
        support_features=support_features,
        support_labels=["old-a", "old-a", "old-a", "new-a", "new-a", "new-a"],
        support_logits=None,
        old_labels={"old-a"},
        config=cfg,
    )


def test_smec_builds_old_boundary_margin_from_support_only():
    cfg = _config(old_boundary_quantile=0.05, old_boundary_slack=0.0)
    model = _two_class_support_model(cfg)

    assert "old-a" in model.old_boundary_margin_thresholds
    assert model.old_boundary_margin_thresholds["old-a"] > 0.80
    assert model.global_old_boundary_margin_threshold > 0.80


def test_smec_builds_obace_conformal_scores_from_support_only():
    cfg = _config()
    model = _two_class_support_model(cfg)

    assert "old-a" in model.obace_conformal_scores
    assert "new-a" in model.obace_conformal_scores
    assert len(model.obace_conformal_scores["old-a"]) == 3
    assert len(model.obace_conformal_scores["new-a"]) == 3


def test_smec_obace_guard_lifts_high_consensus_unknown_with_absolute_failures():
    cfg = _config(
        old_label_aux_policy="obace_guard",
        obace_conformal_weight=1.0,
        obace_conformal_min_risk=0.70,
        obace_old_min_abs_failures=2,
        old_lift_min_weakness=0.50,
    )
    rows = [
        _row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a", role="unknown", true_label="unknown-a"),
        {
            **_row(
                "e0",
                score=0.20,
                margin=0.01,
                risk=0.05,
                label="old-a",
                role="unknown",
                true_label="unknown-a",
            ),
            "receiver_id": "rx-b",
        },
    ]
    models = {"rx-a": _two_class_support_model(cfg), "rx-b": _two_class_support_model(cfg)}
    query_features = {
        ("e0", "rx-a"): np.asarray([0.70, 0.70], dtype=np.float32),
        ("e0", "rx-b"): np.asarray([0.70, 0.70], dtype=np.float32),
    }

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert all(row["smec_event_label_agreement"] == 1.0 for row in out)
    assert all(row["smec_obace_conformal_risk"] >= 0.70 for row in out)
    assert all(row["smec_obace_absolute_fail_count"] >= 2 for row in out)
    assert all(row["smec_old_label_lift_blocked"] == 0 for row in out)
    assert all(row["unknown_risk"] > 0.50 for row in out)


def test_smec_obace_guard_blocks_old_label_when_only_boundary_fails():
    cfg = _config(
        old_label_aux_policy="obace_guard",
        proto_weight=0.0,
        knn_weight=0.0,
        old_boundary_weight=1.0,
        obace_conformal_weight=0.0,
        obace_old_min_abs_failures=2,
        old_boundary_min_risk=0.80,
        old_lift_min_weakness=0.50,
    )
    rows = [
        _row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"),
        {**_row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"), "receiver_id": "rx-b"},
    ]
    models = {"rx-a": _two_class_support_model(cfg), "rx-b": _two_class_support_model(cfg)}
    query_features = {
        ("e0", "rx-a"): np.asarray([0.70, 0.70], dtype=np.float32),
        ("e0", "rx-b"): np.asarray([0.70, 0.70], dtype=np.float32),
    }

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert all(row["smec_old_boundary_risk"] >= 0.80 for row in out)
    assert all(row["smec_obace_absolute_fail_count"] == 1 for row in out)
    assert all(row["smec_old_label_lift_blocked"] == 1 for row in out)
    assert all(row["unknown_risk"] == 0.05 for row in out)


def test_smec_obace_guard_blocks_seen_new_label_without_enough_absolute_failures():
    cfg = _config(
        old_label_aux_policy="obace_guard",
        proto_weight=1.0,
        knn_weight=0.0,
        old_boundary_weight=0.0,
        obace_conformal_weight=1.0,
        obace_conformal_min_risk=0.70,
        obace_nonold_min_abs_failures=3,
    )
    rows = [_row("e0", score=0.20, margin=0.01, risk=0.05, label="new-a", role="seen_new", true_label="new-a")]
    models = {"rx-a": _two_class_support_model(cfg)}
    query_features = {("e0", "rx-a"): np.asarray([0.70, 0.70], dtype=np.float32)}

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert out[0]["smec_obace_absolute_fail_count"] < 3
    assert out[0]["smec_old_label_lift_blocked"] == 1
    assert out[0]["unknown_risk"] == 0.05


def test_smec_obace_event_guard_lifts_consistent_unknown_from_event_evidence():
    cfg = _config(
        old_label_aux_policy="obace_event_guard",
        proto_weight=0.0,
        knn_weight=0.0,
        old_boundary_weight=0.0,
        obace_conformal_weight=0.0,
        obace_event_weight=1.0,
        obace_event_vote_min_risk=0.50,
        obace_event_min_votes=2,
        obace_event_min_mean_risk=0.50,
        old_lift_min_weakness=0.50,
    )
    rows = [
        _row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a", role="unknown", true_label="unknown-a"),
        {
            **_row(
                "e0",
                score=0.20,
                margin=0.01,
                risk=0.05,
                label="old-a",
                role="unknown",
                true_label="unknown-a",
            ),
            "receiver_id": "rx-b",
        },
        {
            **_row(
                "e0",
                score=0.20,
                margin=0.01,
                risk=0.05,
                label="old-a",
                role="unknown",
                true_label="unknown-a",
            ),
            "receiver_id": "rx-c",
        },
    ]
    models = {rx: _two_class_support_model(cfg) for rx in ["rx-a", "rx-b", "rx-c"]}
    query_features = {
        ("e0", "rx-a"): np.asarray([0.70, 0.70], dtype=np.float32),
        ("e0", "rx-b"): np.asarray([0.70, 0.70], dtype=np.float32),
        ("e0", "rx-c"): np.asarray([0.70, 0.70], dtype=np.float32),
    }

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert all(row["smec_obace_event_vote_count"] >= 2 for row in out)
    assert all(row["smec_obace_event_risk"] > 0.50 for row in out)
    assert all(row["smec_old_label_lift_blocked"] == 0 for row in out)
    assert all(row["unknown_risk"] > 0.50 for row in out)


def test_smec_obace_event_guard_blocks_strong_old_despite_event_evidence():
    cfg = _config(
        old_label_aux_policy="obace_event_guard",
        proto_weight=0.0,
        knn_weight=0.0,
        old_boundary_weight=0.0,
        obace_conformal_weight=0.0,
        obace_event_weight=1.0,
        obace_event_vote_min_risk=0.50,
        obace_event_min_votes=2,
        obace_event_min_mean_risk=0.50,
    )
    rows = [
        _row("e0", score=0.95, margin=0.30, risk=0.05, label="old-a"),
        {**_row("e0", score=0.95, margin=0.30, risk=0.05, label="old-a"), "receiver_id": "rx-b"},
        {**_row("e0", score=0.95, margin=0.30, risk=0.05, label="old-a"), "receiver_id": "rx-c"},
    ]
    models = {rx: _two_class_support_model(cfg) for rx in ["rx-a", "rx-b", "rx-c"]}
    query_features = {
        ("e0", "rx-a"): np.asarray([0.70, 0.70], dtype=np.float32),
        ("e0", "rx-b"): np.asarray([0.70, 0.70], dtype=np.float32),
        ("e0", "rx-c"): np.asarray([0.70, 0.70], dtype=np.float32),
    }

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert all(row["smec_obace_event_vote_count"] >= 2 for row in out)
    assert all(row["smec_strong_known_candidate"] == 1 for row in out)
    assert all(row["smec_old_label_lift_blocked"] == 1 for row in out)
    assert all(row["unknown_risk"] == 0.05 for row in out)


def test_smec_old_boundary_guard_lifts_agreed_old_label_near_foreign_boundary():
    cfg = _config(
        old_label_aux_policy="old_boundary_guard",
        old_boundary_weight=1.0,
        old_boundary_min_risk=0.80,
        old_lift_min_weakness=0.50,
    )
    rows = [
        _row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"),
        {**_row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"), "receiver_id": "rx-b"},
    ]
    models = {"rx-a": _two_class_support_model(cfg), "rx-b": _two_class_support_model(cfg)}
    query_features = {
        ("e0", "rx-a"): np.asarray([0.70, 0.70], dtype=np.float32),
        ("e0", "rx-b"): np.asarray([0.70, 0.70], dtype=np.float32),
    }

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert all(row["smec_event_label_agreement"] == 1.0 for row in out)
    assert all(row["smec_old_boundary_risk"] >= 0.80 for row in out)
    assert all(row["smec_old_label_lift_blocked"] == 0 for row in out)
    assert all(row["unknown_risk"] > 0.50 for row in out)


def test_smec_old_boundary_guard_blocks_agreed_old_label_inside_old_margin():
    cfg = _config(
        old_label_aux_policy="old_boundary_guard",
        old_boundary_weight=1.0,
        old_boundary_min_risk=0.80,
        old_lift_min_weakness=0.50,
    )
    rows = [
        _row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"),
        {**_row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"), "receiver_id": "rx-b"},
    ]
    models = {"rx-a": _two_class_support_model(cfg), "rx-b": _two_class_support_model(cfg)}
    query_features = {
        ("e0", "rx-a"): np.asarray([1.0, 0.0], dtype=np.float32),
        ("e0", "rx-b"): np.asarray([1.0, 0.0], dtype=np.float32),
    }

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert all(row["smec_old_boundary_risk"] < 0.20 for row in out)
    assert all(row["smec_old_label_lift_blocked"] == 1 for row in out)
    assert all(row["unknown_risk"] == 0.05 for row in out)


def test_smec_preserves_base_label_authority():
    cfg = _config()
    rows = [_row("e0", score=0.25, margin=0.01, label="old-a")]
    models = {"rx-a": _support_model(cfg)}
    query_features = {("e0", "rx-a"): np.asarray([0.0, 1.0], dtype=np.float32)}

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert out[0]["predicted_label"] == "old-a"
    assert out[0]["class_evidence_top1_label"] == "old-a"
    assert out[0]["smec_label_authority"] == "base_qknn_only"


def test_smec_caps_strong_known_aux_without_lowering_base_risk():
    cfg = _config(strong_aux_cap=0.05)
    rows = [_row("e0", score=0.95, margin=0.30, risk=0.40)]
    models = {"rx-a": _support_model(cfg)}
    query_features = {("e0", "rx-a"): np.asarray([0.0, 1.0], dtype=np.float32)}

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert out[0]["smec_strong_known_candidate"] == 1
    assert out[0]["unknown_risk"] == 0.40
    assert out[0]["smec_aux_component"] <= 0.05


def test_smec_raises_weak_far_from_support_unknown_risk():
    cfg = _config()
    rows = [_row("e0", score=0.20, margin=0.01, risk=0.05)]
    models = {"rx-a": _support_model(cfg)}
    query_features = {("e0", "rx-a"): np.asarray([0.0, 1.0], dtype=np.float32)}

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert out[0]["smec_proto_risk"] > 0.90
    assert out[0]["smec_knn_risk"] > 0.90
    assert out[0]["unknown_risk"] > 0.50
    assert out[0]["class_evidence_top1_unknown_risk"] == out[0]["unknown_risk"]


def test_smec_old_lossless_policy_never_lifts_old_label_risk():
    cfg = _config(old_label_aux_policy="never")
    rows = [_row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a")]
    models = {"rx-a": _support_model(cfg)}
    query_features = {("e0", "rx-a"): np.asarray([0.0, 1.0], dtype=np.float32)}

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert out[0]["unknown_risk"] == 0.05
    assert out[0]["smec_old_label_aux_policy"] == "never"
    assert out[0]["smec_old_label_lift_blocked"] == 1


def test_smec_old_lossless_policy_still_lifts_seen_new_label_risk():
    cfg = _config(old_label_aux_policy="never")
    rows = [_row("e0", score=0.20, margin=0.01, risk=0.05, label="new-a")]
    models = {"rx-a": _support_model(cfg)}
    query_features = {("e0", "rx-a"): np.asarray([0.0, 1.0], dtype=np.float32)}

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert out[0]["unknown_risk"] > 0.50
    assert out[0]["smec_old_label_lift_blocked"] == 0


def test_smec_old_consensus_guard_blocks_agreed_old_label_risk():
    cfg = _config(old_label_aux_policy="consensus_guard")
    rows = [
        _row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"),
        {**_row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"), "receiver_id": "rx-b"},
    ]
    models = {"rx-a": _support_model(cfg), "rx-b": _support_model(cfg)}
    query_features = {
        ("e0", "rx-a"): np.asarray([0.0, 1.0], dtype=np.float32),
        ("e0", "rx-b"): np.asarray([0.0, 1.0], dtype=np.float32),
    }

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    assert all(row["unknown_risk"] == 0.05 for row in out)
    assert all(row["smec_event_label_agreement"] == 1.0 for row in out)
    assert all(row["smec_old_label_lift_blocked"] == 1 for row in out)


def test_smec_old_consensus_guard_lifts_disagreed_weak_old_label_risk():
    cfg = _config(
        old_label_aux_policy="consensus_guard",
        old_lift_max_label_agreement=0.60,
        old_lift_min_weakness=0.50,
    )
    rows = [
        _row("e0", score=0.20, margin=0.01, risk=0.05, label="old-a"),
        {**_row("e0", score=0.20, margin=0.01, risk=0.05, label="new-a"), "receiver_id": "rx-b"},
    ]
    models = {"rx-a": _support_model(cfg), "rx-b": _support_model(cfg)}
    query_features = {
        ("e0", "rx-a"): np.asarray([0.0, 1.0], dtype=np.float32),
        ("e0", "rx-b"): np.asarray([0.0, 1.0], dtype=np.float32),
    }

    out = augment_smec_evidence(rows, models, query_features, {}, cfg, old_labels={"old-a"})

    old_row = next(row for row in out if row["receiver_id"] == "rx-a")
    assert old_row["smec_event_label_agreement"] == 0.5
    assert old_row["smec_old_label_lift_blocked"] == 0
    assert old_row["unknown_risk"] > 0.50
