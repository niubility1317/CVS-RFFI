import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.spaceborne_fewshot import (  # noqa: E402
    ClassState,
    OrbitAdaptiveMSEHead,
    OpenSetGateConfig,
    PredictionResult,
    PrototypeSet,
    SiameseAnchorVerifier,
    UNKNOWN_LABEL,
    apply_density_shell_inlier_gate,
    apply_identity_consensus_arbitration,
    apply_seen_new_registration_override,
    apply_old_unknown_acceptance_guard,
    apply_old_primary_acceptance_gate,
    apply_pseudo_unknown_void_gate,
    apply_retention_rescue_gate,
    apply_siamese_verifier_to_ambiguous,
    apply_source_looo_unknown_risk_arbitration,
    apply_support_conformal_arbitration,
    apply_support_reconstruction_arbitration,
    apply_two_branch_background_guard,
    build_prototype_set,
    calibrate_anchor_density_gates,
    calibrate_thresholds,
    fit_low_compute_target_adapter,
    generate_pseudo_unknown_features,
    predict_with_oa_mse_head,
    register_old_classes,
)
from eval_spaceborne_fewshot import _run_oa_mse_protocol, parse_args  # noqa: E402


def _state(label: int, prototype: list[float]) -> ClassState:
    p = torch.tensor(prototype, dtype=torch.float32)
    return ClassState(
        class_id=label,
        group="old",
        prototype=p,
        mask=torch.ones_like(p),
        subspace=torch.zeros((p.numel(), 0), dtype=torch.float32),
        covariance_diag=torch.ones_like(p),
        thresholds={"min_margin": -1.0e6},
        evt_params={},
    )


def test_oa_mse_multiproto_score_uses_same_class_support_anchor_mixture():
    class0 = _state(0, [1.0, 0.0])
    class0.support_anchors = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    class0.thresholds.update(
        {
            "soft_mixture_score_enabled": True,
            "soft_mixture_topk": 1,
            "soft_mixture_temperature": 0.05,
            "soft_mixture_score_weight": 1.0,
        }
    )
    states = {
        0: class0,
        1: _state(1, [-1.0, 0.0]),
    }

    head = OrbitAdaptiveMSEHead(dim=2, class_states=states, beta_residual=0.0, eta_mahalanobis=0.0)
    result = predict_with_oa_mse_head(torch.tensor([[0.0, 1.0]], dtype=torch.float32), head)

    assert int(result.candidate_labels[0].item()) == 0
    assert result.diagnostics["best_old_score"][0].item() > 0.9


def test_source_looo_unknown_risk_rejects_impostor_accept_without_unknown_labels():
    states = {
        0: _state(0, [1.0, 0.0]),
        1: _state(1, [0.55, 0.83]),
        2: _state(2, [-1.0, 0.0]),
    }
    head = OrbitAdaptiveMSEHead(dim=2, class_states=states, beta_residual=0.0, eta_mahalanobis=0.0)
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    base = PredictionResult(
        predicted_labels=torch.tensor([0, 0], dtype=torch.long),
        candidate_labels=torch.tensor([0, 0], dtype=torch.long),
        scores=torch.tensor([1.0, 0.0], dtype=torch.float32),
        accepted=torch.tensor([True, True]),
        diagnostics={"old_support_evidence_delta": torch.tensor([0.10, -0.20], dtype=torch.float32)},
        decisions=["accept", "accept"],
        gate_reasons=["base_accept", "base_accept"],
    )
    source_features = torch.tensor(
        [[1.0, 0.0], [0.98, 0.02], [0.55, 0.83], [0.58, 0.80], [-1.0, 0.0], [-0.98, 0.02]],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    result = apply_source_looo_unknown_risk_arbitration(
        query,
        base,
        head,
        source_features,
        source_labels,
        torch.empty((0, 2), dtype=torch.float32),
        enabled=True,
        risk_quantile=0.90,
        risk_slack=0.0,
        min_score_margin=0.05,
        min_known_evidence_delta=-0.08,
        reject_min_failures=2,
    )

    assert result.accepted[0].item() is True
    assert result.accepted[1].item() is False
    assert int(result.predicted_labels[1].item()) == UNKNOWN_LABEL
    assert result.gate_reasons[1] == "source_looo_unknown_risk_reject"
    assert "source_looo_risk_margin" in result.diagnostics


def test_negative_anchor_background_loss_is_reported_in_adapter_telemetry():
    source = build_prototype_set(
        torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]], dtype=torch.float32),
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        gate_config=OpenSetGateConfig(mode="oa_mse"),
    )
    support = torch.tensor([[0.97, 0.03], [0.03, 0.97]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(
        source,
        support,
        labels,
        source_adapter_features=source.vectors,
        source_adapter_labels=source.labels,
        steps=2,
        negative_anchor_weight=0.20,
        negative_anchor_margin=0.08,
        pseudo_unknown_samples_per_pair=2,
    )

    assert telemetry["negative_anchor_weight"] == 0.20
    assert telemetry["negative_anchor_count"] > 0
    assert "negative_anchor_background_basin" in telemetry["loss_profile"]
    assert telemetry["loss_terms"]["negative_anchor_background_basin_weighted"] >= 0.0


def test_anchor_density_gate_rejects_off_manifold_old_like_query():
    class0 = _state(0, [1.0, 0.0])
    class0.support_anchors = torch.tensor([[1.0, 0.0], [0.98, 0.02]], dtype=torch.float32)
    class1 = _state(1, [0.0, 1.0])
    class1.support_anchors = torch.tensor([[0.0, 1.0], [0.02, 0.98]], dtype=torch.float32)
    states = {0: class0, 1: class1}
    known = torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    telemetry = calibrate_anchor_density_gates(
        states,
        known,
        labels,
        enabled=True,
        topk=2,
        temperature=0.05,
        min_quantile=0.25,
        margin_quantile=0.25,
        action="reject",
    )
    head = OrbitAdaptiveMSEHead(dim=2, class_states=states, beta_residual=0.0, eta_mahalanobis=0.0)
    result = predict_with_oa_mse_head(torch.tensor([[0.72, 0.70], [1.0, 0.0]], dtype=torch.float32), head)

    assert telemetry["class_count"] == 2
    assert result.decisions[0] == "reject"
    assert result.gate_reasons[0] == "anchor_density_reject"
    assert result.accepted[1].item() is True
    assert "anchor_density_margin_delta" in result.diagnostics


def test_density_shell_gate_accepts_class_inliers_before_rejecting_open_space():
    class0 = _state(0, [1.0, 0.0])
    class0.support_anchors = torch.tensor([[1.0, 0.0], [0.98, 0.02]], dtype=torch.float32)
    class1 = _state(10, [0.0, 1.0])
    class1.group = "seen_new"
    class1.support_anchors = torch.tensor([[0.0, 1.0], [0.02, 0.98]], dtype=torch.float32)
    for state in (class0, class1):
        state.thresholds.update(
            {
                "anchor_density_topk": 1,
                "anchor_density_temperature": 0.05,
                "min_anchor_density": 0.90,
            }
        )
    class0.thresholds.update(
        {
            "min_old_support_anchor_similarity": 0.90,
            "min_old_support_evidence": 0.0,
        }
    )
    class1.thresholds.update(
        {
            "min_seen_new_anchor_similarity": 0.90,
            "min_seen_new_evidence": 0.0,
        }
    )
    head = OrbitAdaptiveMSEHead(dim=2, class_states={0: class0, 10: class1}, beta_residual=0.0, eta_mahalanobis=0.0)
    query = torch.tensor([[0.0, 1.0], [0.7071, 0.7071]], dtype=torch.float32)
    base = predict_with_oa_mse_head(query, head)

    result = apply_density_shell_inlier_gate(
        query,
        base,
        head,
        torch.tensor([[0.7071, 0.7071]], dtype=torch.float32),
        enabled=True,
        seen_new_min_evidence_delta=0.0,
        seen_new_min_anchor_delta=0.0,
        seen_new_min_density_delta=0.0,
        old_min_evidence_delta=0.0,
        old_min_anchor_delta=0.0,
        old_min_density_delta=0.0,
        reject_background_score=0.95,
        reject_background_margin=0.20,
        reject_min_failed_shells=2,
    )

    assert result.predicted_labels.tolist()[0] == 10
    assert result.accepted.tolist()[0] is True
    assert result.gate_reasons[0] == "density_shell_inlier_accept"
    assert result.predicted_labels.tolist()[1] == UNKNOWN_LABEL
    assert result.accepted.tolist()[1] is False
    assert result.gate_reasons[1] == "density_shell_open_space_reject"
    assert "density_shell_old_density_delta" in result.diagnostics


def test_identity_consensus_arbitration_prefers_known_identity_before_background_reject():
    class0 = _state(0, [1.0, 0.0])
    class0.support_anchors = torch.tensor([[1.0, 0.0], [0.99, 0.01]], dtype=torch.float32)
    class1 = _state(10, [0.0, 1.0])
    class1.group = "seen_new"
    class1.support_anchors = torch.tensor([[0.0, 1.0], [0.01, 0.99]], dtype=torch.float32)
    for state in (class0, class1):
        state.thresholds.update(
            {
                "anchor_density_topk": 1,
                "anchor_density_temperature": 0.05,
                "min_anchor_density": 0.90,
            }
        )
    class0.thresholds.update(
        {
            "min_old_support_anchor_similarity": 0.90,
            "min_old_support_evidence": 0.0,
        }
    )
    class1.thresholds.update(
        {
            "min_seen_new_anchor_similarity": 0.90,
            "min_seen_new_evidence": 0.0,
        }
    )
    head = OrbitAdaptiveMSEHead(dim=2, class_states={0: class0, 10: class1}, beta_residual=0.0, eta_mahalanobis=0.0)
    query = torch.tensor([[0.0, 1.0], [0.7071, 0.7071]], dtype=torch.float32)
    base = predict_with_oa_mse_head(query, head)

    result = apply_identity_consensus_arbitration(
        query,
        base,
        head,
        torch.tensor([[0.7071, 0.7071]], dtype=torch.float32),
        enabled=True,
        old_min_evidence_delta=0.0,
        old_min_anchor_delta=0.0,
        old_min_density_delta=0.0,
        seen_new_min_evidence_delta=0.0,
        seen_new_min_anchor_delta=0.0,
        seen_new_min_density_delta=0.0,
        background_accept_margin=0.30,
        reject_background_score=0.95,
        reject_background_margin=0.20,
        reject_min_identity_failures=4,
    )

    assert result.predicted_labels.tolist()[0] == 10
    assert result.accepted.tolist()[0] is True
    assert result.gate_reasons[0] == "identity_consensus_accept"
    assert result.predicted_labels.tolist()[1] == UNKNOWN_LABEL
    assert result.accepted.tolist()[1] is False
    assert result.gate_reasons[1] == "identity_consensus_open_space_reject"
    assert "identity_consensus_margin" in result.diagnostics


def test_support_conformal_arbitration_rejects_accepted_row_outside_class_support():
    states = {
        0: _state(0, [1.0, 0.0, 0.0]),
        1: _state(1, [0.0, 1.0, 0.0]),
    }
    states[0].support_anchors = torch.tensor([[1.0, 0.0, 0.0], [0.98, 0.05, 0.0]], dtype=torch.float32)
    states[1].support_anchors = torch.tensor([[0.0, 1.0, 0.0], [0.05, 0.98, 0.0]], dtype=torch.float32)
    for state in states.values():
        state.thresholds.update(
            {
                "min_old_support_evidence": -1.0e6,
                "min_old_support_anchor_similarity": 0.70,
                "min_seen_new_evidence": -1.0e6,
                "min_seen_new_anchor_similarity": 0.70,
                "min_anchor_density": -1.0e6,
            }
        )
    head = OrbitAdaptiveMSEHead(dim=3, class_states=states)
    features = torch.tensor([[0.55, 0.52, 0.65], [0.99, 0.02, 0.0]], dtype=torch.float32)
    base = PredictionResult(
        predicted_labels=torch.tensor([0, 0], dtype=torch.long),
        scores=torch.tensor([0.8, 0.9], dtype=torch.float32),
        accepted=torch.tensor([True, True]),
        candidate_labels=torch.tensor([0, 0], dtype=torch.long),
        decisions=["accept", "accept"],
        gate_reasons=["base_accept", "base_accept"],
    )

    result = apply_support_conformal_arbitration(
        features,
        base,
        head,
        torch.tensor([[0.60, 0.58, 0.55]], dtype=torch.float32),
        enabled=True,
        calibration_quantile=0.05,
        conformity_slack=0.02,
        anchor_margin_slack=0.02,
        background_score=0.70,
        background_margin=-0.20,
        hard_reject_margin=0.05,
        reject_min_failures=2,
    )

    assert result.accepted.tolist() == [False, True]
    assert result.predicted_labels.tolist()[0] == UNKNOWN_LABEL
    assert result.gate_reasons[0] == "support_conformal_open_space_reject"
    assert "support_conformal_margin" in result.diagnostics


def test_support_reconstruction_arbitration_rejects_class_local_residual_outlier():
    states = {
        0: _state(0, [1.0, 0.0, 0.0]),
        1: _state(1, [0.0, 1.0, 0.0]),
    }
    states[0].support_anchors = torch.tensor([[1.0, 0.0, 0.0], [0.98, 0.05, 0.0]], dtype=torch.float32)
    states[1].support_anchors = torch.tensor([[0.0, 1.0, 0.0], [0.05, 0.98, 0.0]], dtype=torch.float32)
    head = OrbitAdaptiveMSEHead(dim=3, class_states=states)
    features = torch.tensor([[0.78, 0.02, 0.62], [0.99, 0.02, 0.0]], dtype=torch.float32)
    base = PredictionResult(
        predicted_labels=torch.tensor([0, 0], dtype=torch.long),
        scores=torch.tensor([0.8, 0.9], dtype=torch.float32),
        accepted=torch.tensor([True, True]),
        candidate_labels=torch.tensor([0, 0], dtype=torch.long),
        decisions=["accept", "accept"],
        gate_reasons=["base_accept", "base_accept"],
    )

    result = apply_support_reconstruction_arbitration(
        features,
        base,
        head,
        torch.tensor([[0.74, 0.08, 0.67]], dtype=torch.float32),
        enabled=True,
        rank=1,
        residual_quantile=0.95,
        residual_slack=0.02,
        min_residual_floor=0.02,
        negative_margin=0.30,
        hard_residual_margin=0.04,
        background_score=0.70,
        background_margin=-0.20,
        reject_min_failures=2,
    )

    assert result.accepted.tolist() == [False, True]
    assert result.predicted_labels.tolist()[0] == UNKNOWN_LABEL
    assert result.gate_reasons[0] == "support_reconstruction_open_space_reject"
    assert result.diagnostics["support_reconstruction_residual_margin"][0].item() < 0.0
    assert "support_reconstruction_negative_margin" in result.diagnostics


def test_siamese_unknown_risk_veto_rejects_low_old_evidence_uncertain_rows():
    states = {0: _state(0, [1.0, 0.0]), 1: _state(1, [0.0, 1.0])}
    features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    verifier = SiameseAnchorVerifier(
        anchor_features=features,
        anchor_labels=torch.tensor([0], dtype=torch.long),
        threshold=0.0,
        scale=20.0,
    )
    base = PredictionResult(
        predicted_labels=torch.tensor([UNKNOWN_LABEL], dtype=torch.long),
        scores=torch.tensor([0.0], dtype=torch.float32),
        accepted=torch.tensor([False]),
        candidate_labels=torch.tensor([UNKNOWN_LABEL], dtype=torch.long),
        diagnostics={
            "old_support_evidence_delta": torch.tensor([-0.20], dtype=torch.float32),
            "old_surrogate_reject_evidence_delta": torch.tensor([-0.10], dtype=torch.float32),
        },
        margins=torch.tensor([0.0], dtype=torch.float32),
        gate_reasons=["old_surrogate_evidence_uncertain"],
        decisions=["uncertain"],
    )

    accepted = apply_siamese_verifier_to_ambiguous(features, base, states, verifier, threshold=0.50)
    assert accepted.predicted_labels.tolist() == [0]
    assert accepted.accepted.tolist() == [True]
    assert accepted.gate_reasons == ["siamese_verified"]

    rejected = apply_siamese_verifier_to_ambiguous(
        features,
        base,
        states,
        verifier,
        threshold=0.50,
        unknown_risk_veto=True,
        min_old_support_evidence_delta=0.0,
        min_old_surrogate_reject_delta=0.0,
    )
    assert rejected.predicted_labels.tolist() == [UNKNOWN_LABEL]
    assert rejected.accepted.tolist() == [False]
    assert rejected.decisions == ["reject"]
    assert rejected.gate_reasons == ["siamese_unknown_risk_reject"]


def test_coupled_siamese_veto_uses_anchor_margin_before_rejecting():
    states = {0: _state(0, [1.0, 0.0]), 1: _state(1, [0.0, 1.0])}
    features = torch.tensor([[1.0, 0.0], [0.72, 0.69]], dtype=torch.float32)
    verifier = SiameseAnchorVerifier(
        anchor_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        anchor_labels=torch.tensor([0, 1], dtype=torch.long),
        threshold=0.0,
        scale=20.0,
    )
    base = PredictionResult(
        predicted_labels=torch.tensor([UNKNOWN_LABEL, UNKNOWN_LABEL], dtype=torch.long),
        scores=torch.tensor([0.0, 0.0], dtype=torch.float32),
        accepted=torch.tensor([False, False]),
        candidate_labels=torch.tensor([UNKNOWN_LABEL, UNKNOWN_LABEL], dtype=torch.long),
        diagnostics={
            "old_support_evidence_delta": torch.tensor([-0.20, -0.20], dtype=torch.float32),
            "old_surrogate_reject_evidence_delta": torch.tensor([-0.10, -0.10], dtype=torch.float32),
            "energy_delta": torch.tensor([-0.20, -0.20], dtype=torch.float32),
            "old_support_anchor_margin": torch.tensor([0.80, 0.01], dtype=torch.float32),
        },
        margins=torch.tensor([0.0, 0.0], dtype=torch.float32),
        gate_reasons=["old_surrogate_evidence_uncertain", "old_surrogate_evidence_uncertain"],
        decisions=["uncertain", "uncertain"],
    )

    result = apply_siamese_verifier_to_ambiguous(
        features,
        base,
        states,
        verifier,
        threshold=0.50,
        unknown_risk_veto=True,
        unknown_risk_veto_mode="coupled",
        min_old_support_evidence_delta=0.0,
        min_old_surrogate_reject_delta=0.0,
        min_energy_delta=0.0,
        min_old_support_anchor_margin=0.05,
        min_veto_failures=2,
    )

    assert result.predicted_labels.tolist() == [0, UNKNOWN_LABEL]
    assert result.accepted.tolist() == [True, False]
    assert result.gate_reasons == ["siamese_verified", "siamese_unknown_risk_coupled_reject"]


def test_old_unknown_acceptance_guard_rejects_weak_old_like_accepts():
    states = {0: _state(0, [1.0, 0.0]), 1: _state(1, [0.0, 1.0])}
    base = PredictionResult(
        predicted_labels=torch.tensor([0, 0], dtype=torch.long),
        scores=torch.tensor([0.80, 0.10], dtype=torch.float32),
        accepted=torch.tensor([True, True]),
        candidate_labels=torch.tensor([0, 0], dtype=torch.long),
        diagnostics={
            "old_surrogate_reject_evidence_delta": torch.tensor([0.35, -0.10], dtype=torch.float32),
            "best_old_score": torch.tensor([0.80, -1.20], dtype=torch.float32),
            "old_support_anchor_margin": torch.tensor([0.20, 0.01], dtype=torch.float32),
        },
        margins=torch.tensor([5.0, 0.2], dtype=torch.float32),
        gate_reasons=["accepted", "siamese_verified"],
        decisions=["accept", "accept"],
    )

    guarded = apply_old_unknown_acceptance_guard(
        base,
        states,
        enabled=True,
        min_old_surrogate_reject_delta=0.20,
        min_best_old_score=0.0,
        min_old_support_anchor_margin=0.05,
        min_margin=1.0,
        min_guard_failures=2,
    )

    assert guarded.predicted_labels.tolist() == [0, UNKNOWN_LABEL]
    assert guarded.accepted.tolist() == [True, False]
    assert guarded.gate_reasons == ["accepted", "old_unknown_acceptance_guard_reject"]
    assert bool(guarded.diagnostics["old_unknown_acceptance_guard_reject_mask"][1].item())


def test_old_primary_gate_preserves_only_high_consistency_old_accepts():
    class0 = _state(0, [1.0, 0.0])
    class0.thresholds.update({"class_envelope_gate_enabled": True, "class_envelope_min_failures": 1})
    class1 = _state(1, [0.0, 1.0])
    head = OrbitAdaptiveMSEHead(dim=2, class_states={0: class0, 1: class1}, beta_residual=0.0, eta_mahalanobis=0.0)
    features = torch.tensor([[1.0, 0.0], [0.98, 0.02]], dtype=torch.float32)
    base = PredictionResult(
        predicted_labels=torch.tensor([0, 0], dtype=torch.long),
        scores=torch.tensor([1.0, 0.9], dtype=torch.float32),
        accepted=torch.tensor([True, True]),
        candidate_labels=torch.tensor([0, 0], dtype=torch.long),
        diagnostics={
            "old_support_evidence_delta": torch.tensor([0.20, 0.20], dtype=torch.float32),
            "old_support_anchor_delta": torch.tensor([0.10, 0.10], dtype=torch.float32),
            "old_support_anchor_margin": torch.tensor([0.12, 0.12], dtype=torch.float32),
            "old_drift_cos": torch.tensor([0.80, 0.80], dtype=torch.float32),
            "old_drift_dist": torch.tensor([0.20, 0.20], dtype=torch.float32),
            "support_knn_label": torch.tensor([0, 1], dtype=torch.long),
            "support_knn_margin": torch.tensor([0.20, 0.20], dtype=torch.float32),
            "support_knn_seen_new_minus_old": torch.tensor([-0.20, -0.20], dtype=torch.float32),
            "soft_mixture_score_margin": torch.tensor([0.25, 0.25], dtype=torch.float32),
            "soft_mixture_cos": torch.tensor([0.90, 0.90], dtype=torch.float32),
            "soft_mixture_residual": torch.tensor([0.05, 0.05], dtype=torch.float32),
            "soft_mixture_consistency_pass_mask": torch.tensor([True, True]),
            "class_envelope_label": torch.tensor([0, 0], dtype=torch.long),
            "class_envelope_failure_count": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "class_envelope_reject_mask": torch.tensor([False, False]),
        },
        margins=torch.tensor([0.80, 0.80], dtype=torch.float32),
        gate_reasons=["accepted", "identity_consensus_accept"],
        decisions=["accept", "accept"],
    )

    gated = apply_old_primary_acceptance_gate(
        features,
        base,
        head,
        torch.empty((0, 2), dtype=torch.float32),
        enabled=True,
        require_soft_mixture=True,
        require_support_knn=True,
        require_class_envelope=True,
        min_old_support_evidence_delta=0.0,
        min_old_support_anchor_delta=0.0,
        min_old_support_anchor_margin=0.0,
        min_score_margin=0.0,
        min_soft_mixture_margin=0.0,
        min_soft_mixture_cos=0.50,
        max_soft_mixture_residual=0.20,
        min_support_knn_margin=0.0,
        max_support_knn_seen_new_minus_old=0.0,
        min_old_drift_cos=0.50,
        max_old_drift_dist=0.50,
        fail_action="defer",
    )

    assert gated.predicted_labels.tolist() == [0, UNKNOWN_LABEL]
    assert gated.accepted.tolist() == [True, False]
    assert gated.gate_reasons == ["accepted", "old_primary_consistency_defer"]
    assert bool(gated.diagnostics["old_primary_consistency_pass_mask"][0].item())
    assert not bool(gated.diagnostics["old_primary_support_knn_pass_mask"][1].item())
    assert bool(gated.diagnostics["old_primary_blocked_accept_mask"][1].item())


def test_old_primary_gate_makes_unknown_veto_dominate_retention_rescue():
    class0 = _state(0, [1.0, 0.0])
    class0.thresholds.update({"class_envelope_gate_enabled": True, "class_envelope_min_failures": 1})
    head = OrbitAdaptiveMSEHead(dim=2, class_states={0: class0}, beta_residual=0.0, eta_mahalanobis=0.0)
    features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    rescued = PredictionResult(
        predicted_labels=torch.tensor([0], dtype=torch.long),
        scores=torch.tensor([0.95], dtype=torch.float32),
        accepted=torch.tensor([True]),
        candidate_labels=torch.tensor([0], dtype=torch.long),
        diagnostics={
            "old_support_evidence_delta": torch.tensor([0.30], dtype=torch.float32),
            "old_support_anchor_delta": torch.tensor([0.20], dtype=torch.float32),
            "old_support_anchor_margin": torch.tensor([0.20], dtype=torch.float32),
            "old_drift_cos": torch.tensor([0.90], dtype=torch.float32),
            "old_drift_dist": torch.tensor([0.10], dtype=torch.float32),
            "support_knn_label": torch.tensor([0], dtype=torch.long),
            "support_knn_margin": torch.tensor([0.30], dtype=torch.float32),
            "support_knn_seen_new_minus_old": torch.tensor([-0.30], dtype=torch.float32),
            "soft_mixture_score_margin": torch.tensor([0.30], dtype=torch.float32),
            "soft_mixture_cos": torch.tensor([0.95], dtype=torch.float32),
            "soft_mixture_residual": torch.tensor([0.02], dtype=torch.float32),
            "soft_mixture_consistency_pass_mask": torch.tensor([True]),
            "class_envelope_label": torch.tensor([0], dtype=torch.long),
            "class_envelope_failure_count": torch.tensor([0.0], dtype=torch.float32),
            "class_envelope_reject_mask": torch.tensor([False]),
            "old_unknown_acceptance_guard_reject_mask": torch.tensor([True]),
            "retention_rescue_accept_mask": torch.tensor([True]),
        },
        margins=torch.tensor([1.0], dtype=torch.float32),
        gate_reasons=["retention_rescue_accept"],
        decisions=["accept"],
    )

    gated = apply_old_primary_acceptance_gate(
        features,
        rescued,
        head,
        torch.empty((0, 2), dtype=torch.float32),
        enabled=True,
        require_soft_mixture=True,
        require_support_knn=True,
        require_class_envelope=True,
        min_old_support_evidence_delta=0.0,
        min_old_support_anchor_delta=0.0,
        min_old_support_anchor_margin=0.0,
        min_score_margin=0.0,
        min_soft_mixture_margin=0.0,
        min_soft_mixture_cos=0.50,
        max_soft_mixture_residual=0.20,
        min_support_knn_margin=0.0,
        min_old_drift_cos=0.50,
        max_old_drift_dist=0.50,
        unknown_veto_action="reject",
    )

    assert gated.predicted_labels.tolist() == [UNKNOWN_LABEL]
    assert gated.accepted.tolist() == [False]
    assert gated.decisions == ["reject"]
    assert gated.gate_reasons == ["old_primary_unknown_veto"]
    assert bool(gated.diagnostics["old_primary_consistency_pass_mask"][0].item())
    assert bool(gated.diagnostics["old_primary_unknown_veto_applied_mask"][0].item())


def test_old_primary_gate_vetoes_pre_reject_risk_after_retention_rescue():
    class0 = _state(0, [1.0, 0.0])
    class0.thresholds.update({"class_envelope_gate_enabled": True, "class_envelope_min_failures": 1})
    head = OrbitAdaptiveMSEHead(dim=2, class_states={0: class0}, beta_residual=0.0, eta_mahalanobis=0.0)
    features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    pre_reject_veto_sources = (
        "pre_reject_arbitration_reject_mask",
        "pre_reject_arbitration_background_reject_risk_mask",
        "pre_reject_arbitration_background_defer_risk_mask",
        "pre_reject_arbitration_extreme_background_mask",
    )

    for source_name in pre_reject_veto_sources:
        rescued = PredictionResult(
            predicted_labels=torch.tensor([0], dtype=torch.long),
            scores=torch.tensor([0.95], dtype=torch.float32),
            accepted=torch.tensor([True]),
            candidate_labels=torch.tensor([0], dtype=torch.long),
            diagnostics={
                "old_support_evidence_delta": torch.tensor([0.30], dtype=torch.float32),
                "old_support_anchor_delta": torch.tensor([0.20], dtype=torch.float32),
                "old_support_anchor_margin": torch.tensor([0.20], dtype=torch.float32),
                "old_drift_cos": torch.tensor([0.90], dtype=torch.float32),
                "old_drift_dist": torch.tensor([0.10], dtype=torch.float32),
                "support_knn_label": torch.tensor([0], dtype=torch.long),
                "support_knn_margin": torch.tensor([0.30], dtype=torch.float32),
                "support_knn_seen_new_minus_old": torch.tensor([-0.30], dtype=torch.float32),
                "soft_mixture_score_margin": torch.tensor([0.30], dtype=torch.float32),
                "soft_mixture_cos": torch.tensor([0.95], dtype=torch.float32),
                "soft_mixture_residual": torch.tensor([0.02], dtype=torch.float32),
                "soft_mixture_consistency_pass_mask": torch.tensor([True]),
                "class_envelope_label": torch.tensor([0], dtype=torch.long),
                "class_envelope_failure_count": torch.tensor([0.0], dtype=torch.float32),
                "class_envelope_reject_mask": torch.tensor([False]),
                "retention_rescue_accept_mask": torch.tensor([True]),
                source_name: torch.tensor([True]),
            },
            margins=torch.tensor([1.0], dtype=torch.float32),
            gate_reasons=["retention_rescue_accept"],
            decisions=["accept"],
        )

        gated = apply_old_primary_acceptance_gate(
            features,
            rescued,
            head,
            torch.empty((0, 2), dtype=torch.float32),
            enabled=True,
            require_soft_mixture=True,
            require_support_knn=True,
            require_class_envelope=True,
            min_old_support_evidence_delta=0.0,
            min_old_support_anchor_delta=0.0,
            min_old_support_anchor_margin=0.0,
            min_score_margin=0.0,
            min_soft_mixture_margin=0.0,
            min_soft_mixture_cos=0.50,
            max_soft_mixture_residual=0.20,
            min_support_knn_margin=0.0,
            min_old_drift_cos=0.50,
            max_old_drift_dist=0.50,
            unknown_veto_action="reject",
        )

        assert gated.predicted_labels.tolist() == [UNKNOWN_LABEL], source_name
        assert gated.accepted.tolist() == [False], source_name
        assert gated.decisions == ["reject"], source_name
        assert gated.gate_reasons == ["old_primary_unknown_veto"], source_name
        assert bool(gated.diagnostics["old_primary_consistency_pass_mask"][0].item()), source_name
        assert bool(gated.diagnostics["old_primary_unknown_veto_applied_mask"][0].item()), source_name
        assert gated.diagnostics["old_primary_prior_veto_count"][0].item() == 1.0


def test_retention_rescue_candidate_requires_old_primary_consensus_before_accept():
    class0 = _state(0, [1.0, 0.0])
    class0.thresholds.update({"class_envelope_gate_enabled": True, "class_envelope_min_failures": 1})
    class1 = _state(1, [0.0, 1.0])
    head = OrbitAdaptiveMSEHead(dim=2, class_states={0: class0, 1: class1}, beta_residual=0.0, eta_mahalanobis=0.0)
    features = torch.tensor([[1.0, 0.0], [0.98, 0.02]], dtype=torch.float32)
    base = PredictionResult(
        predicted_labels=torch.tensor([UNKNOWN_LABEL, UNKNOWN_LABEL], dtype=torch.long),
        scores=torch.tensor([0.0, 0.0], dtype=torch.float32),
        accepted=torch.tensor([False, False]),
        candidate_labels=torch.tensor([0, 0], dtype=torch.long),
        diagnostics={
            "old_support_evidence_delta": torch.tensor([0.30, 0.30], dtype=torch.float32),
            "old_support_anchor_delta": torch.tensor([0.20, 0.20], dtype=torch.float32),
            "old_support_anchor_margin": torch.tensor([0.20, 0.20], dtype=torch.float32),
            "old_drift_cos": torch.tensor([0.90, 0.10], dtype=torch.float32),
            "old_drift_dist": torch.tensor([0.10, 0.90], dtype=torch.float32),
            "support_knn_label": torch.tensor([0, 0], dtype=torch.long),
            "support_knn_margin": torch.tensor([0.30, 0.30], dtype=torch.float32),
            "support_knn_seen_new_minus_old": torch.tensor([-0.30, -0.30], dtype=torch.float32),
            "soft_mixture_score_margin": torch.tensor([0.30, 0.30], dtype=torch.float32),
            "soft_mixture_cos": torch.tensor([0.95, 0.95], dtype=torch.float32),
            "soft_mixture_residual": torch.tensor([0.02, 0.02], dtype=torch.float32),
            "soft_mixture_consistency_pass_mask": torch.tensor([True, True]),
            "class_envelope_label": torch.tensor([0, 0], dtype=torch.long),
            "class_envelope_failure_count": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "class_envelope_reject_mask": torch.tensor([False, False]),
        },
        margins=torch.tensor([0.0, 0.0], dtype=torch.float32),
        gate_reasons=["base_reject", "base_reject"],
        decisions=["reject", "reject"],
    )

    rescued = apply_retention_rescue_gate(
        features,
        base,
        head,
        torch.empty((0, 2), dtype=torch.float32),
        enabled=True,
        old_min_evidence_delta=0.0,
        old_min_anchor_delta=0.0,
        old_min_anchor_margin=0.0,
        old_min_score_margin=0.0,
        max_background_score=1.0,
        max_background_margin=1.0,
        direct_accept=False,
    )

    assert rescued.accepted.tolist() == [False, False]
    assert rescued.gate_reasons == ["base_reject", "base_reject"]
    assert rescued.diagnostics["retention_rescue_eligible_mask"].tolist() == [True, True]
    assert rescued.diagnostics["retention_rescue_accept_mask"].tolist() == [False, False]

    gated = apply_old_primary_acceptance_gate(
        features,
        rescued,
        head,
        torch.empty((0, 2), dtype=torch.float32),
        enabled=True,
        promote_rescue_candidates=True,
        require_soft_mixture=True,
        require_support_knn=True,
        require_class_envelope=True,
        min_old_support_evidence_delta=0.0,
        min_old_support_anchor_delta=0.0,
        min_old_support_anchor_margin=0.0,
        min_score_margin=0.0,
        min_soft_mixture_margin=0.0,
        min_soft_mixture_cos=0.50,
        max_soft_mixture_residual=0.20,
        min_support_knn_margin=0.0,
        max_support_knn_seen_new_minus_old=0.0,
        min_old_drift_cos=0.50,
        max_old_drift_dist=0.50,
        fail_action="defer",
    )

    assert gated.predicted_labels.tolist() == [0, UNKNOWN_LABEL]
    assert gated.accepted.tolist() == [True, False]
    assert gated.gate_reasons == ["old_primary_rescue_consensus_accept", "old_primary_consistency_defer"]
    assert bool(gated.diagnostics["old_primary_rescue_promoted_mask"][0].item())
    assert bool(gated.diagnostics["old_primary_rescue_blocked_mask"][1].item())


def test_old_primary_gate_blocks_each_missing_consistency_source():
    class0 = _state(0, [1.0, 0.0])
    class0.thresholds.update({"class_envelope_gate_enabled": True, "class_envelope_min_failures": 1})
    class1 = _state(1, [0.0, 1.0])
    head = OrbitAdaptiveMSEHead(dim=2, class_states={0: class0, 1: class1}, beta_residual=0.0, eta_mahalanobis=0.0)
    features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    base_diagnostics = {
        "old_support_evidence_delta": torch.tensor([0.30], dtype=torch.float32),
        "old_support_anchor_delta": torch.tensor([0.20], dtype=torch.float32),
        "old_support_anchor_margin": torch.tensor([0.20], dtype=torch.float32),
        "old_drift_cos": torch.tensor([0.90], dtype=torch.float32),
        "old_drift_dist": torch.tensor([0.10], dtype=torch.float32),
        "support_knn_label": torch.tensor([0], dtype=torch.long),
        "support_knn_margin": torch.tensor([0.30], dtype=torch.float32),
        "support_knn_seen_new_minus_old": torch.tensor([-0.30], dtype=torch.float32),
        "soft_mixture_score_margin": torch.tensor([0.30], dtype=torch.float32),
        "soft_mixture_cos": torch.tensor([0.95], dtype=torch.float32),
        "soft_mixture_residual": torch.tensor([0.02], dtype=torch.float32),
        "soft_mixture_consistency_pass_mask": torch.tensor([True]),
        "class_envelope_label": torch.tensor([0], dtype=torch.long),
        "class_envelope_failure_count": torch.tensor([0.0], dtype=torch.float32),
        "class_envelope_reject_mask": torch.tensor([False]),
    }
    cases = [
        ("soft_mixture_score_margin", torch.tensor([-0.10], dtype=torch.float32), "old_primary_soft_mixture_pass_mask"),
        ("support_knn_margin", torch.tensor([-0.10], dtype=torch.float32), "old_primary_support_knn_pass_mask"),
        ("support_knn_label", torch.tensor([1], dtype=torch.long), "old_primary_support_knn_pass_mask"),
        ("old_drift_cos", torch.tensor([0.10], dtype=torch.float32), "old_primary_drift_pass_mask"),
        ("old_drift_dist", torch.tensor([0.90], dtype=torch.float32), "old_primary_drift_pass_mask"),
        ("class_envelope_failure_count", torch.tensor([1.0], dtype=torch.float32), "old_primary_class_envelope_pass_mask"),
        ("class_envelope_reject_mask", torch.tensor([True]), "old_primary_class_envelope_pass_mask"),
    ]

    for key, bad_value, pass_mask_name in cases:
        diagnostics = dict(base_diagnostics)
        diagnostics[key] = bad_value
        base = PredictionResult(
            predicted_labels=torch.tensor([0], dtype=torch.long),
            scores=torch.tensor([1.0], dtype=torch.float32),
            accepted=torch.tensor([True]),
            candidate_labels=torch.tensor([0], dtype=torch.long),
            diagnostics=diagnostics,
            margins=torch.tensor([1.0], dtype=torch.float32),
            gate_reasons=["accepted"],
            decisions=["accept"],
        )

        gated = apply_old_primary_acceptance_gate(
            features,
            base,
            head,
            torch.empty((0, 2), dtype=torch.float32),
            enabled=True,
            require_soft_mixture=True,
            require_support_knn=True,
            require_class_envelope=True,
            min_old_support_evidence_delta=0.0,
            min_old_support_anchor_delta=0.0,
            min_old_support_anchor_margin=0.0,
            min_score_margin=0.0,
            min_soft_mixture_margin=0.0,
            min_soft_mixture_cos=0.50,
            max_soft_mixture_residual=0.20,
            min_support_knn_margin=0.0,
            max_support_knn_seen_new_minus_old=0.0,
            min_old_drift_cos=0.50,
            max_old_drift_dist=0.50,
            fail_action="defer",
        )

        assert gated.predicted_labels.tolist() == [UNKNOWN_LABEL], key
        assert gated.accepted.tolist() == [False], key
        assert gated.gate_reasons == ["old_primary_consistency_defer"], key
        assert not bool(gated.diagnostics[pass_mask_name][0].item()), key


def test_two_branch_background_guard_keeps_strong_old_support_and_rejects_weak_background_risk():
    state = _state(0, [1.0, 0.0])
    base = PredictionResult(
        predicted_labels=torch.tensor([0, 0], dtype=torch.long),
        scores=torch.tensor([0.70, 0.70], dtype=torch.float32),
        accepted=torch.tensor([True, True]),
        candidate_labels=torch.tensor([0, 0], dtype=torch.long),
        diagnostics={
            "old_support_evidence_delta": torch.tensor([-0.50, 0.20], dtype=torch.float32),
            "old_support_anchor_delta": torch.tensor([-0.50, 0.08], dtype=torch.float32),
            "old_support_anchor_margin": torch.tensor([0.00, 0.05], dtype=torch.float32),
        },
        gate_reasons=["accepted", "accepted"],
        decisions=["accept", "accept"],
    )

    guarded = apply_two_branch_background_guard(
        torch.tensor([[0.60, 0.80], [0.60, 0.80]], dtype=torch.float32),
        base,
        {0: state},
        pseudo_unknown=torch.tensor([[0.60, 0.80]], dtype=torch.float32),
        enabled=True,
        min_background_score=0.90,
        min_background_margin=0.10,
        old_support_evidence_delta=0.0,
        old_support_anchor_delta=0.0,
        old_support_anchor_margin=0.02,
    )

    assert guarded.predicted_labels.tolist() == [UNKNOWN_LABEL, 0]
    assert guarded.accepted.tolist() == [False, True]
    assert guarded.gate_reasons == ["two_branch_background_guard_reject", "accepted"]
    assert bool(guarded.diagnostics["two_branch_background_reject_mask"][0].item())
    assert bool(guarded.diagnostics["two_branch_support_override_mask"][1].item())


def test_seen_new_registration_override_accepts_registered_support_but_respects_background_risk():
    old_state = _state(0, [1.0, 0.0])
    old_state.support_anchors = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    old_state.thresholds["min_old_support_evidence"] = 0.90
    seen_state = _state(10, [0.0, 1.0])
    seen_state.group = "seen_new"
    seen_state.support_anchors = torch.tensor([[0.0, 1.0], [0.02, 0.98]], dtype=torch.float32)
    seen_state.thresholds.update(
        {
            "min_seen_new_evidence": 0.55,
            "min_seen_new_support_affinity": 0.55,
            "max_seen_new_support_residual": 0.60,
            "min_seen_new_anchor_similarity": 0.80,
        }
    )
    head = OrbitAdaptiveMSEHead(dim=2, class_states={0: old_state, 10: seen_state}, beta_residual=0.0, eta_mahalanobis=0.0)
    features = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    base = PredictionResult(
        predicted_labels=torch.tensor([UNKNOWN_LABEL], dtype=torch.long),
        scores=torch.tensor([0.0], dtype=torch.float32),
        accepted=torch.tensor([False]),
        candidate_labels=torch.tensor([0], dtype=torch.long),
        diagnostics={},
        gate_reasons=["unknown_reject"],
        decisions=["reject"],
    )

    accepted = apply_seen_new_registration_override(
        features,
        base,
        head,
        pseudo_unknown=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        enabled=True,
        min_evidence_delta=0.0,
        min_anchor_delta=0.0,
        min_affinity_delta=0.0,
        min_residual_delta=0.0,
        min_score_margin=-0.50,
        min_seen_vs_old_evidence_margin=0.0,
        max_background_score=0.90,
        max_background_margin=0.0,
    )

    assert accepted.predicted_labels.tolist() == [10]
    assert accepted.candidate_labels.tolist() == [10]
    assert accepted.accepted.tolist() == [True]
    assert accepted.gate_reasons == ["seen_new_registration_override"]
    assert bool(accepted.diagnostics["seen_new_registration_override_mask"][0].item())
    assert accepted.diagnostics["seen_new_override_seen_minus_old_evidence"][0].item() > 0.0

    rejected = apply_seen_new_registration_override(
        features,
        base,
        head,
        pseudo_unknown=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        enabled=True,
        min_evidence_delta=0.0,
        min_anchor_delta=0.0,
        min_affinity_delta=0.0,
        min_residual_delta=0.0,
        min_score_margin=-0.50,
        min_seen_vs_old_evidence_margin=0.0,
        max_background_score=0.90,
        max_background_margin=-0.10,
    )

    assert rejected.predicted_labels.tolist() == [UNKNOWN_LABEL]
    assert rejected.accepted.tolist() == [False]
    assert bool(rejected.diagnostics["seen_new_override_background_risk_mask"][0].item())


def test_surrogate_unknown_sets_accept_energy_gate_and_does_not_accept_boundary_sample():
    states = {
        0: _state(0, [1.0, 0.0]),
        1: _state(1, [0.0, 1.0]),
    }
    known = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    boundary = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    updated = calibrate_thresholds(
        states,
        known,
        labels,
        surrogate_unknown=boundary,
        target_far=0.05,
        evt_mode="weibull",
    )
    for state in updated.values():
        assert "max_energy" in state.thresholds
        assert state.thresholds["max_energy"] == state.thresholds["reject_energy"]
        assert "surrogate_reject_energy" in state.thresholds
        state.thresholds["max_residual"] = 10.0
        state.thresholds["reject_residual"] = 10.0
        state.thresholds["max_mahalanobis"] = 1.0e6
        state.thresholds["reject_mahalanobis"] = 1.0e6

    head = OrbitAdaptiveMSEHead(dim=2, class_states=updated)
    result = predict_with_oa_mse_head(boundary, head)

    assert result.accepted.tolist() == [False]
    assert result.decisions[0] in {"reject", "uncertain"}
    assert result.energy is not None
    assert float(result.energy[0].item()) >= updated[0].thresholds["surrogate_reject_energy"]


def test_old_surrogate_reject_relax_creates_uncertain_band_below_support_evidence_gate():
    states = {
        0: _state(0, [1.0, 0.0, 0.0]),
        1: _state(1, [0.0, 1.0, 0.0]),
    }
    states[0].support_anchors = torch.tensor([[1.0, 0.0, 0.0], [0.98, 0.05, 0.0]], dtype=torch.float32)
    states[1].support_anchors = torch.tensor([[0.0, 1.0, 0.0], [0.05, 0.98, 0.0]], dtype=torch.float32)
    known = torch.tensor(
        [[1.0, 0.0, 0.0], [0.98, 0.05, 0.0], [0.0, 1.0, 0.0], [0.05, 0.98, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    surrogate_unknown = torch.tensor(
        [[0.7, 0.7, 0.0], [0.7, -0.7, 0.0], [-0.7, 0.7, 0.0]],
        dtype=torch.float32,
    )

    updated = calibrate_thresholds(
        states,
        known,
        labels,
        surrogate_unknown=surrogate_unknown,
        target_far=0.05,
        evt_mode="weibull",
        old_surrogate_evidence_margin=0.04,
        old_surrogate_reject_relax=0.25,
    )

    for state in updated.values():
        gate = state.evt_params["old_surrogate_evidence_gate"]
        assert state.thresholds["min_old_surrogate_reject_evidence"] == (
            state.thresholds["min_old_surrogate_evidence"] - 0.25
        )
        assert gate["reject_relax"] == 0.25
        assert gate["surrogate_unknown_reject_threshold"] == state.thresholds["min_old_surrogate_reject_evidence"]
        assert "uncertain_between" in gate["action"]


def test_target_adapter_reports_source_retention_and_unknown_moat_terms():
    source_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.02, 0.98, 0.0],
        ],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)
    assert isinstance(source, PrototypeSet)

    support = torch.tensor(
        [
            [0.96, 0.04, 0.0],
            [0.04, 0.96, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.08, 0.92],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(
        source,
        support,
        labels,
        source_adapter_features=source_features,
        source_adapter_labels=source_labels,
        source_boundary_pseudo_unknown_samples_per_pair=1,
        source_boundary_pseudo_unknown_offset_scale=0.18,
        steps=2,
        source_ce_weight=0.20,
        unknown_moat_weight=0.30,
        unknown_moat_margin=0.40,
        pseudo_unknown_target_shift_samples_per_class=2,
        pseudo_unknown_target_shift_offset_scale=0.20,
        pseudo_unknown_target_halo_samples_per_class=2,
        pseudo_unknown_target_halo_offset_scale=0.35,
        pseudo_unknown_target_ring_samples_per_class=2,
        pseudo_unknown_target_ring_offset_scale=0.45,
        old_bridge_weight=0.15,
        old_bridge_samples_per_class=2,
        support_contrast_weight=0.20,
        support_contrast_negative_margin=0.78,
        support_contrast_positive_margin=0.88,
        soft_proto_weight=0.25,
        soft_proto_topk=2,
        soft_proto_temperature=0.10,
        adapter_selection_policy="proxy_line_search",
    )

    assert telemetry["loss_profile"] == "target_support_ce+source_old_retention_ce+source_anchor+old_target_bridge_ce+pseudo_unknown_moat+old_neighborhood_retention+old_surrogate_margin+target_support_contrast+class_constrained_soft_prototype_mixture"
    assert telemetry["adapter_selection_policy"] == "proxy_line_search"
    assert telemetry["selected_alpha"] in {0.0, 0.25, 0.5, 0.75, 1.0}
    assert len(telemetry["adapter_selection"]["candidates"]) == 5
    assert telemetry["source_ce_weight"] == 0.20
    assert telemetry["source_adapter_feature_count"] == 4
    assert telemetry["source_adapter_label_count"] == 4
    assert telemetry["source_boundary_pseudo_unknown_samples_per_pair"] == 1
    assert telemetry["pseudo_unknown_source_boundary_count"] > 0
    assert telemetry["unknown_moat_weight"] == 0.30
    assert telemetry["unknown_moat_margin"] == 0.40
    assert telemetry["pseudo_unknown_count"] > 0
    assert telemetry["pseudo_unknown_geometry_count"] > 0
    assert telemetry["pseudo_unknown_target_shift_count"] == 4
    assert telemetry["pseudo_unknown_target_halo_count"] == 4
    assert telemetry["pseudo_unknown_target_ring_count"] == 4
    assert telemetry["old_bridge_count"] == 4
    assert telemetry["support_contrast_weight"] == 0.20
    assert telemetry["support_contrast_anchor_count"] == 2
    assert telemetry["soft_proto_weight"] == 0.25
    assert telemetry["soft_proto_topk"] == 2
    assert telemetry["soft_proto_anchor_count"] > 0
    assert telemetry["soft_proto_train_count"] > 0
    assert telemetry["old_neighborhood_count"] > 0
    assert telemetry["loss_terms"]["source_old_ce_weighted"] >= 0.0
    assert telemetry["loss_terms"]["old_target_bridge_ce_weighted"] >= 0.0
    assert telemetry["loss_terms"]["pseudo_unknown_moat_weighted"] >= 0.0
    assert telemetry["loss_terms"]["old_neighborhood_retention_weighted"] >= 0.0
    assert telemetry["loss_terms"]["old_surrogate_margin_weighted"] >= 0.0
    assert telemetry["loss_terms"]["target_support_contrast_weighted"] >= 0.0
    assert telemetry["loss_terms"]["class_constrained_soft_prototype_mixture_weighted"] >= 0.0


def test_target_adapter_reports_soft_prototype_mixture_boundary_terms():
    source_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.02, 0.98, 0.0],
        ],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)

    support = torch.tensor(
        [
            [0.96, 0.04, 0.0],
            [0.04, 0.96, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.08, 0.92],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(
        source,
        support,
        labels,
        steps=2,
        soft_proto_weight=0.20,
        soft_proto_topk=2,
        soft_proto_temperature=0.10,
        soft_proto_boundary_weight=0.30,
        soft_proto_boundary_margin=0.18,
    )

    assert "+soft_prototype_mixture_boundary" in telemetry["loss_profile"]
    assert telemetry["soft_proto_boundary_weight"] == 0.30
    assert telemetry["soft_proto_boundary_margin"] == 0.18
    assert telemetry["loss_terms"]["soft_prototype_mixture_boundary_weighted"] >= 0.0
    assert telemetry["loss_trace"][-1]["loss_soft_proto_boundary_weighted"] >= 0.0


def test_target_adapter_reports_support_center_geometry_terms():
    source_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
        ],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)

    support = torch.tensor(
        [
            [0.90, 0.10, 0.0],
            [0.88, 0.12, 0.0],
            [0.10, 0.90, 0.0],
            [0.12, 0.88, 0.0],
            [0.0, 0.10, 0.90],
            [0.0, 0.12, 0.88],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(
        source,
        support,
        labels,
        steps=2,
        support_center_ce_weight=0.35,
        support_center_temperature=0.07,
        support_center_margin=0.18,
    )

    assert "+support_center_leave_one_out_ce_margin" in telemetry["loss_profile"]
    assert telemetry["support_center_ce_weight"] == 0.35
    assert telemetry["support_center_temperature"] == 0.07
    assert telemetry["support_center_margin"] == 0.18
    assert telemetry["support_center_class_count"] == 3
    assert telemetry["loss_terms"]["support_center_leave_one_out_weighted"] >= 0.0
    assert telemetry["loss_trace"][-1]["loss_support_center_weighted"] >= 0.0


def test_target_adapter_reports_void_background_competition_terms():
    source_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.02, 0.98, 0.0],
        ],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)
    support = torch.tensor(
        [
            [0.96, 0.04, 0.0],
            [0.04, 0.96, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(
        source,
        support,
        labels,
        steps=2,
        unknown_moat_weight=0.20,
        pseudo_unknown_samples_per_pair=2,
        pseudo_unknown_target_halo_samples_per_class=1,
        void_background_weight=0.25,
    )

    assert "+void_background_competition" in telemetry["loss_profile"]
    assert telemetry["void_background_weight"] == 0.25
    assert telemetry["loss_terms"]["void_background_competition_weighted"] >= 0.0
    assert telemetry["loss_trace"][-1]["loss_void_background_weighted"] >= 0.0


def test_pseudo_unknown_void_gate_rejects_void_like_accepted_rows():
    states = {0: _state(0, [1.0, 0.0]), 1: _state(1, [0.0, 1.0])}
    features = torch.tensor([[0.60, 0.80]], dtype=torch.float32)
    base = PredictionResult(
        predicted_labels=torch.tensor([1], dtype=torch.long),
        scores=torch.tensor([0.80], dtype=torch.float32),
        accepted=torch.tensor([True]),
        candidate_labels=torch.tensor([1], dtype=torch.long),
        diagnostics={},
        margins=torch.tensor([0.20], dtype=torch.float32),
        gate_reasons=["accepted"],
        decisions=["accept"],
    )

    gated = apply_pseudo_unknown_void_gate(
        features,
        base,
        states,
        pseudo_unknown=torch.tensor([[0.60, 0.80]], dtype=torch.float32),
        enabled=True,
        min_void_score=0.90,
        min_void_margin=0.10,
    )

    assert gated.predicted_labels.tolist() == [UNKNOWN_LABEL]
    assert gated.accepted.tolist() == [False]
    assert gated.decisions == ["reject"]
    assert gated.gate_reasons == ["pseudo_unknown_void_gate_reject"]
    assert bool(gated.diagnostics["void_background_reject_mask"][0].item())
    assert gated.diagnostics["void_background_margin"][0].item() > 0.10


def test_target_boundary_guard_adapter_selection_reports_conservative_proxy_terms():
    source_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.02, 0.98, 0.0],
        ],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)
    assert isinstance(source, PrototypeSet)

    support = torch.tensor(
        [
            [0.96, 0.04, 0.0],
            [0.04, 0.96, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.08, 0.92],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(
        source,
        support,
        labels,
        source_adapter_features=source_features,
        source_adapter_labels=source_labels,
        steps=2,
        unknown_moat_weight=0.30,
        unknown_moat_margin=0.40,
        pseudo_unknown_target_halo_samples_per_class=2,
        pseudo_unknown_target_ring_samples_per_class=2,
        soft_proto_weight=0.20,
        support_contrast_weight=0.20,
        adapter_selection_policy="target_boundary_guard",
    )

    assert telemetry["adapter_selection_policy"] == "target_boundary_guard"
    assert telemetry["adapter_selection"]["policy"] == "target_boundary_guard"
    assert telemetry["adapter_selection"]["reason"] == "old_source_bridge_retention_with_surrogate_unknown_risk_and_alpha_penalty"
    candidates = telemetry["adapter_selection"]["candidates"]
    assert len(candidates) == 5
    assert all("boundary_guard_score" in item for item in candidates)
    assert all("legacy_proxy_score" in item for item in candidates)
    assert telemetry["selected_alpha"] in {0.0, 0.25, 0.5, 0.75, 1.0}


def test_retention_risk_balanced_adapter_selection_reports_overfit_terms():
    source_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.02, 0.98, 0.0],
        ],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)
    support = torch.tensor(
        [
            [0.96, 0.04, 0.0],
            [0.04, 0.96, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.08, 0.92],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(
        source,
        support,
        labels,
        source_adapter_features=source_features,
        source_adapter_labels=source_labels,
        steps=2,
        unknown_moat_weight=0.30,
        unknown_moat_margin=0.40,
        pseudo_unknown_target_halo_samples_per_class=2,
        pseudo_unknown_target_ring_samples_per_class=2,
        soft_proto_weight=0.20,
        support_contrast_weight=0.20,
        adapter_selection_policy="retention_risk_balanced",
    )

    selection = telemetry["adapter_selection"]
    assert telemetry["adapter_selection_policy"] == "retention_risk_balanced"
    assert selection["policy"] == "retention_risk_balanced"
    assert selection["reason"] == "old_retention_first_with_support_overfit_and_surrogate_unknown_risk_penalty"
    assert all("retention_risk_score" in item for item in selection["candidates"])
    assert all("retention_floor" in item for item in selection["candidates"])
    assert all("support_overfit_penalty" in item for item in selection["candidates"])
    assert telemetry["selected_alpha"] in {0.0, 0.25, 0.5, 0.75, 1.0}


def test_support_cv_adapter_selector_reports_leave_one_out_generalization_proxy():
    source_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.96, 0.04, 0.0],
            [0.0, 1.0, 0.0],
            [0.04, 0.96, 0.0],
        ],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)
    support = torch.tensor(
        [
            [0.96, 0.04, 0.0],
            [0.92, 0.08, 0.0],
            [0.04, 0.96, 0.0],
            [0.08, 0.92, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.08, 0.92],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(
        source,
        support,
        labels,
        source_adapter_features=source_features,
        source_adapter_labels=source_labels,
        steps=2,
        support_center_ce_weight=0.20,
        soft_proto_weight=0.20,
        unknown_moat_weight=0.20,
        pseudo_unknown_target_ring_samples_per_class=2,
        adapter_selection_policy="support_cv_constrained",
        old_acc_target=0.95,
        seen_new_acc_target=0.80,
    )

    selection = telemetry["adapter_selection"]
    assert telemetry["adapter_selection_policy"] == "support_cv_constrained"
    assert selection["policy"] == "support_cv_constrained"
    assert selection["reason"] == "support_leave_one_out_feasibility_then_old_retention_and_surrogate_unknown_risk"
    assert "selected_support_cv_acc" in selection
    assert "selected_support_cv_margin_p10" in selection
    assert all("support_cv_old_acc" in item for item in selection["candidates"])
    assert all("support_cv_overfit_penalty" in item for item in selection["candidates"])


def test_source_open_set_uses_source_prototypes_for_energy_calibration_without_target_support():
    source_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.02, 0.98, 0.0],
        ],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)

    result = _run_oa_mse_protocol(
        protocol="source_open_set",
        source=source,
        support_features=torch.empty((0, 3), dtype=torch.float32),
        support_labels=torch.empty((0,), dtype=torch.long),
        query_features=torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32),
        query_labels=torch.tensor([0, UNKNOWN_LABEL], dtype=torch.long),
        gate_config=OpenSetGateConfig(mode="oa_mse"),
    )

    telemetry = result.telemetry["oa_mse_onboard_adaptation"]["pseudo_unknown_energy"]
    assert telemetry["known_calibration_source"] == "source_old_prototypes_no_target_labels"
    assert telemetry["accept_gate"] == "old_retention_constrained_energy_plus_surrogate_reject_energy"
    assert telemetry["old_evidence_gate"] == "support_derived_old_vs_surrogate_unknown_evidence_hard_reject"


def test_adapter_alpha_eval_sweep_reports_query_metrics_without_changing_primary_result():
    source_features = torch.tensor(
        [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 1.0, 0.0], [0.02, 0.98, 0.0]],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)
    support = torch.tensor([[0.96, 0.04, 0.0], [0.04, 0.96, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    query = torch.tensor([[0.95, 0.05, 0.0], [0.05, 0.95, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32)
    query_labels = torch.tensor([0, 1, UNKNOWN_LABEL], dtype=torch.long)

    result = _run_oa_mse_protocol(
        protocol="ftrc",
        source=source,
        support_features=support,
        support_labels=labels,
        query_features=query,
        query_labels=query_labels,
        source_adapter_features=source_features,
        source_adapter_labels=source_labels,
        gate_config=OpenSetGateConfig(mode="oa_mse"),
        adapter_steps=1,
        adapter_selection_policy="proxy_line_search",
        adapter_alpha_eval_sweep=True,
    )

    target_adapter = result.telemetry["oa_mse_onboard_adaptation"]["target_adapter"]
    sweep = target_adapter["alpha_eval_sweep"]
    assert target_adapter["alpha_eval_sweep_policy"] == "eval_only_no_query_training_or_threshold_fit"
    assert [row["alpha"] for row in sweep] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all("metrics" in row for row in sweep)
    assert all("old_class_accuracy" in row["metrics"] for row in sweep)
    assert result.metrics["old_class_accuracy"] >= 0.0


def test_support_retention_guard_is_reported_in_pseudo_unknown_energy_telemetry():
    source_features = torch.tensor(
        [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 1.0, 0.0], [0.02, 0.98, 0.0]],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)
    support = torch.tensor([[0.96, 0.04, 0.0], [0.04, 0.96, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    query = torch.tensor([[0.95, 0.05, 0.0], [0.05, 0.95, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32)
    query_labels = torch.tensor([0, 1, UNKNOWN_LABEL], dtype=torch.long)

    result = _run_oa_mse_protocol(
        protocol="ftrc",
        source=source,
        support_features=support,
        support_labels=labels,
        query_features=query,
        query_labels=query_labels,
        source_adapter_features=source_features,
        source_adapter_labels=source_labels,
        gate_config=OpenSetGateConfig(mode="oa_mse"),
        adapter_steps=1,
        support_retention_guard=True,
        support_retention_guard_quantile=0.20,
        support_retention_guard_slack=0.03,
        two_branch_background_guard=True,
        two_branch_bg_min_score=0.61,
        two_branch_bg_min_margin=-0.04,
        two_branch_old_support_evidence_delta=-0.02,
        two_branch_old_anchor_delta=-0.03,
        two_branch_old_anchor_margin=0.01,
        seen_new_registration_override=True,
        seen_new_override_min_evidence_delta=0.01,
        seen_new_override_min_anchor_delta=0.02,
        seen_new_override_min_score_margin=-0.05,
    )

    telemetry = result.telemetry["oa_mse_onboard_adaptation"]["pseudo_unknown_energy"]
    assert telemetry["support_retention_guard"] is True
    assert telemetry["support_retention_guard_quantile"] == 0.20
    assert telemetry["support_retention_guard_slack"] == 0.03
    two_branch = result.telemetry["oa_mse_onboard_adaptation"]["two_branch_background_guard"]
    assert two_branch["enabled"] is True
    assert two_branch["unknown_query_threshold_calibration"] is False
    assert two_branch["thresholds"]["background_score"] == 0.61
    assert two_branch["thresholds"]["old_support_anchor_margin"] == 0.01
    seen_override = result.telemetry["oa_mse_onboard_adaptation"]["seen_new_registration_override"]
    assert seen_override["enabled"] is True
    assert seen_override["unknown_query_threshold_calibration"] is False
    assert seen_override["thresholds"]["min_evidence_delta"] == 0.01
    assert seen_override["thresholds"]["min_anchor_delta"] == 0.02
    assert seen_override["thresholds"]["min_score_margin"] == -0.05


def test_old_surrogate_evidence_gate_marks_boundary_old_candidate_uncertain():
    states = {
        0: _state(0, [1.0, 0.0]),
        1: _state(1, [0.0, 1.0]),
    }
    states[0].support_anchors = torch.tensor([[1.0, 0.0], [0.98, 0.02]], dtype=torch.float32)
    states[1].support_anchors = torch.tensor([[0.0, 1.0], [0.02, 0.98]], dtype=torch.float32)
    known = torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    boundary = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    updated = calibrate_thresholds(
        states,
        known,
        labels,
        surrogate_unknown=boundary,
        target_far=0.05,
        evt_mode="weibull",
        old_surrogate_evidence_margin=0.05,
    )
    for state in updated.values():
        assert "min_old_support_evidence" in state.thresholds
        assert "min_old_surrogate_evidence" in state.thresholds
        state.thresholds["max_residual"] = 10.0
        state.thresholds["reject_residual"] = 10.0
        state.thresholds["max_mahalanobis"] = 1.0e6
        state.thresholds["reject_mahalanobis"] = 1.0e6
        state.thresholds["max_energy"] = 1.0e6
        state.thresholds["reject_energy"] = 1.0e6
        state.thresholds["surrogate_reject_energy"] = 1.0e6
        state.thresholds["min_old_support_anchor_similarity"] = -1.0
        state.thresholds["min_old_support_evidence"] = -1.0e6

    head = OrbitAdaptiveMSEHead(dim=2, class_states=updated)
    result = predict_with_oa_mse_head(boundary, head)

    assert result.accepted.tolist() == [False]
    assert result.decisions == ["reject"]
    assert result.gate_reasons == ["old_surrogate_evidence_reject"]
    assert "old_support_evidence" in result.diagnostics
    assert "old_surrogate_evidence_delta" in result.diagnostics
    assert "old_surrogate_reject_evidence_delta" in result.diagnostics


def test_register_old_classes_shrinks_low_quality_target_old_support():
    source_features = torch.tensor(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]],
        dtype=torch.float32,
    )
    source_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source = build_prototype_set(source_features, source_labels)
    conflicting_support = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    support_labels = torch.tensor([0], dtype=torch.long)

    states, _ = register_old_classes(
        source,
        conflicting_support,
        support_labels,
        stage="Stage2-B",
        kappa=3.0,
        old_anchor_override_min_quality=0.55,
    )

    state = states[0]
    assert state.support_quality < 0.55
    assert state.thresholds["effective_rho"] < state.thresholds["base_rho"]
    assert state.thresholds["support_source_mean_similarity"] < 0.25


def test_low_quality_old_support_cannot_override_surrogate_energy_reject():
    states = {
        0: _state(0, [1.0, 0.0]),
        1: _state(1, [0.0, 1.0]),
    }
    states[0].support_anchors = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    states[0].support_quality = 0.40
    states[0].thresholds.update(
        {
            "min_old_support_anchor_similarity": 0.50,
            "min_old_support_evidence": -1.0e6,
            "min_old_surrogate_evidence": -1.0e6,
            "min_old_anchor_override_quality": 0.55,
            "max_residual": 10.0,
            "reject_residual": 10.0,
            "max_mahalanobis": 1.0e6,
            "reject_mahalanobis": 1.0e6,
            "max_energy": 1.0e6,
            "reject_energy": 1.0e6,
            "surrogate_reject_energy": -1.0e6,
        }
    )
    states[1].thresholds.update(states[0].thresholds)
    states[1].support_quality = 1.0
    head = OrbitAdaptiveMSEHead(dim=2, class_states=states)

    result = predict_with_oa_mse_head(torch.tensor([[1.0, 0.0]], dtype=torch.float32), head)

    assert result.accepted.tolist() == [False]
    assert result.decisions == ["reject"]
    assert result.gate_reasons == ["surrogate_energy_reject"]
    assert abs(float(result.diagnostics["old_support_quality"][0].item()) - 0.40) < 1.0e-6
    assert float(result.diagnostics["old_support_quality_delta"][0].item()) < 0.0


def test_pseudo_unknown_generation_includes_boundary_and_old_shell_samples():
    states = {
        0: _state(0, [1.0, 0.0, 0.0]),
        1: _state(1, [0.0, 1.0, 0.0]),
    }

    samples = generate_pseudo_unknown_features(states, samples_per_pair=4, offset_scale=0.20)
    assert samples.shape == (4, 3)
    midpoint = torch.nn.functional.normalize(torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32), dim=1)[0]
    assert torch.max(samples @ midpoint) > 0.95
    assert torch.max(samples @ torch.tensor([1.0, 0.0, 0.0])) > torch.max(samples[:2] @ torch.tensor([1.0, 0.0, 0.0]))


def test_target_shift_pseudo_unknown_uses_allowed_old_support_only():
    states = {
        0: _state(0, [1.0, 0.0, 0.0]),
        1: _state(1, [0.0, 1.0, 0.0]),
    }
    states[0].support_anchors = torch.tensor([[0.92, 0.20, 0.0], [0.90, 0.22, 0.0]], dtype=torch.float32)
    states[1].support_anchors = torch.tensor([[0.20, 0.92, 0.0], [0.22, 0.90, 0.0]], dtype=torch.float32)

    samples = generate_pseudo_unknown_features(
        states,
        samples_per_pair=4,
        offset_scale=0.20,
        target_shift_samples_per_class=2,
        target_shift_offset_scale=0.25,
    )

    assert samples.shape == (8, 3)
    target_shift_samples = samples[4:]
    assert torch.max(target_shift_samples @ torch.tensor([0.0, 1.0, 0.0])) > 0.30
    assert torch.max(target_shift_samples @ torch.tensor([1.0, 0.0, 0.0])) > 0.30


def test_target_halo_pseudo_unknown_uses_allowed_old_support_only():
    states = {
        0: _state(0, [1.0, 0.0, 0.0]),
        1: _state(1, [0.0, 1.0, 0.0]),
    }
    states[0].support_anchors = torch.tensor([[0.92, 0.20, 0.0], [0.90, 0.22, 0.0]], dtype=torch.float32)
    states[1].support_anchors = torch.tensor([[0.20, 0.92, 0.0], [0.22, 0.90, 0.0]], dtype=torch.float32)

    samples = generate_pseudo_unknown_features(
        states,
        samples_per_pair=4,
        offset_scale=0.20,
        target_halo_samples_per_class=2,
        target_halo_offset_scale=0.35,
    )

    assert samples.shape == (8, 3)
    target_halo_samples = samples[4:]
    assert torch.max(target_halo_samples @ torch.tensor([0.0, 1.0, 0.0])) > 0.45
    assert torch.max(target_halo_samples @ torch.tensor([1.0, 0.0, 0.0])) > 0.45


def test_target_ring_pseudo_unknown_uses_allowed_old_support_boundary_only():
    states = {
        0: _state(0, [1.0, 0.0, 0.0]),
        1: _state(1, [0.0, 1.0, 0.0]),
    }
    states[0].support_anchors = torch.tensor([[0.92, 0.20, 0.0], [0.90, 0.22, 0.0]], dtype=torch.float32)
    states[1].support_anchors = torch.tensor([[0.20, 0.92, 0.0], [0.22, 0.90, 0.0]], dtype=torch.float32)

    samples = generate_pseudo_unknown_features(
        states,
        samples_per_pair=4,
        offset_scale=0.20,
        target_ring_samples_per_class=2,
        target_ring_offset_scale=0.45,
    )

    assert samples.shape == (8, 3)
    target_ring_samples = samples[4:]
    assert torch.max(target_ring_samples @ torch.tensor([0.92, 0.20, 0.0])) > 0.80
    assert torch.min(target_ring_samples @ torch.tensor([0.92, 0.20, 0.0])) < 0.98


def test_oa_mse_matrix_launcher_emits_target_shift_pseudo_unknown_args():
    from tools.spaceborne_fewshot_da_matrix import make_candidates, render_launcher  # noqa: E402

    candidates = make_candidates(plan="OA_MSE_CARD3")
    script = render_launcher("stage2_unit_target_shift", candidates[:1])

    assert "--pseudo_unknown_target_shift_samples_per_class 2" in script
    assert "--pseudo_unknown_target_shift_offset_scale 0.2" in script
    assert "--pseudo_unknown_source_boundary_samples_per_pair 2" in script
    assert "--pseudo_unknown_source_boundary_offset_scale 0.18" in script
    assert "--pseudo_unknown_target_halo_samples_per_class 2" in script
    assert "--pseudo_unknown_target_halo_offset_scale 0.35" in script
    assert "--pseudo_unknown_target_ring_samples_per_class 3" in script
    assert "--pseudo_unknown_target_ring_offset_scale 0.45" in script
    assert "--oa_mse_adapter_selection_policy proxy_line_search" in script
    assert "--oa_mse_adapter_alpha_eval_sweep" in script
    assert "--oa_mse_old_bridge_weight 0.15" in script
    assert "--old_bridge_samples_per_class 3" in script
    assert "--oa_mse_support_contrast_weight 0.12" in script
    assert "--old_support_contrast_negative_margin 0.78" in script
    assert "--oa_mse_soft_proto_weight 0.08" in script
    assert "--soft_proto_topk 2" in script
    assert "--soft_proto_temperature 0.1" in script


def test_eval_parser_accepts_identity_preserving_adapter_policies(monkeypatch, tmp_path):
    for policy in (
        "identity_preserving",
        "identity_preserving_risk",
        "support_cv_constrained",
        "support_cv_risk_balanced",
        "identity_preserving_cv",
        "identity_preserving_risk_cv",
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_spaceborne_fewshot.py",
                "--protocol",
                "sfe",
                "--output_json",
                str(tmp_path / f"{policy}.json"),
                "--gate_mode",
                "oa_mse",
                "--oa_mse_adapter_selection_policy",
                policy,
            ],
        )

        args = parse_args()

        assert args.oa_mse_adapter_selection_policy == policy


def test_eval_parser_accepts_identity_consensus_support_background_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_spaceborne_fewshot.py",
            "--protocol",
            "sfe",
            "--output_json",
            str(tmp_path / "bgcap.json"),
            "--gate_mode",
            "oa_mse",
            "--oa_mse_identity_consensus_arbitration",
            "--identity_consensus_support_background_cap",
            "--identity_consensus_support_background_cap_quantile",
            "0.86",
            "--identity_consensus_support_background_cap_slack",
            "0.04",
            "--identity_consensus_support_background_cap_min_anchors",
            "3",
        ],
    )

    args = parse_args()

    assert args.identity_consensus_support_background_cap is True
    assert args.identity_consensus_support_background_cap_quantile == 0.86
    assert args.identity_consensus_support_background_cap_slack == 0.04
    assert args.identity_consensus_support_background_cap_min_anchors == 3


def test_eval_parser_accepts_pre_reject_support_neighborhood_retention(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_spaceborne_fewshot.py",
            "--protocol",
            "sfe",
            "--output_json",
            str(tmp_path / "kret.json"),
            "--gate_mode",
            "oa_mse",
            "--oa_mse_pre_reject_defer_arbitration",
            "--pre_reject_support_neighborhood_retention",
            "--pre_reject_support_retention_old_min_evidence_delta",
            "0.03",
            "--pre_reject_support_retention_seen_new_min_score_margin",
            "-0.06",
            "--pre_reject_support_retention_max_background_score",
            "0.91",
            "--pre_reject_support_retention_max_background_margin",
            "0.18",
        ],
    )

    args = parse_args()

    assert args.oa_mse_pre_reject_defer_arbitration is True
    assert args.pre_reject_support_neighborhood_retention is True
    assert args.pre_reject_support_retention_old_min_evidence_delta == 0.03
    assert args.pre_reject_support_retention_seen_new_min_score_margin == -0.06
    assert args.pre_reject_support_retention_max_background_score == 0.91
    assert args.pre_reject_support_retention_max_background_margin == 0.18


def test_eval_parser_accepts_source_risk_constrained_support_retention(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_spaceborne_fewshot.py",
            "--protocol",
            "sfe",
            "--output_json",
            str(tmp_path / "riskret.json"),
            "--gate_mode",
            "oa_mse",
            "--oa_mse_pre_reject_defer_arbitration",
            "--pre_reject_support_neighborhood_retention",
            "--pre_reject_support_retention_require_source_looo_pass",
            "--pre_reject_support_retention_source_looo_max_failures",
            "1",
        ],
    )

    args = parse_args()

    assert args.pre_reject_support_neighborhood_retention is True
    assert args.pre_reject_support_retention_require_source_looo_pass is True
    assert args.pre_reject_support_retention_source_looo_max_failures == 1


def test_eval_parser_accepts_old_primary_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_spaceborne_fewshot.py",
            "--protocol",
            "sfe",
            "--output_json",
            str(tmp_path / "old_primary.json"),
            "--gate_mode",
            "oa_mse",
            "--oa_mse_old_primary_gate",
            "--old_primary_require_soft_mixture",
            "--old_primary_require_support_knn",
            "--old_primary_require_class_envelope",
            "--old_primary_promote_rescue_candidates",
            "--old_primary_min_old_support_evidence_delta",
            "0.03",
            "--old_primary_min_support_knn_margin",
            "0.04",
            "--old_primary_min_old_drift_cos",
            "0.55",
            "--old_primary_max_old_drift_dist",
            "0.45",
            "--old_primary_unknown_veto_action",
            "defer",
            "--retention_rescue_candidate_only",
        ],
    )

    args = parse_args()

    assert args.oa_mse_old_primary_gate is True
    assert args.old_primary_require_soft_mixture is True
    assert args.old_primary_require_support_knn is True
    assert args.old_primary_require_class_envelope is True
    assert args.old_primary_promote_rescue_candidates is True
    assert args.retention_rescue_candidate_only is True
    assert args.old_primary_min_old_support_evidence_delta == 0.03
    assert args.old_primary_min_support_knn_margin == 0.04
    assert args.old_primary_min_old_drift_cos == 0.55
    assert args.old_primary_max_old_drift_dist == 0.45
    assert args.old_primary_unknown_veto_action == "defer"
