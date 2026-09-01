from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from model_dual_cvsincnet import NuisanceHeteroscedasticHead, build_dual_model
from SSDG.train_ssdg import (
    _compute_daot_labeled_step,
    _compute_daot_unlabeled_step,
    _resolve_source_target_axes,
    _validate_daot_config,
    build_arg_parser,
)
from cvsrffi.deployment_orbit import (
    DeploymentOrbitConfig,
    apply_fingerprint_intervention,
    apply_local_nuisance_tangent,
    apply_named_local_nuisance_tangent,
    default_teacher_view_specs,
    daot_ablation_overrides,
    daot_loss_ablation_overrides,
    DAOT_NUISANCE_TANGENT_NAMES,
    physical_reliability_from_meta,
    sample_sparse_joint_direction,
    stable_orbit_key_tensor,
    teacher_importance_matrix,
)
from cvsrffi.daot_training import compute_daot_batch_objective
from cvsrffi.orbit_teacher import (
    EMALossScaleNormalizer,
    TemporalOrbitMemory,
    adv3b02_daot_schedule,
    orbit_logit_distillation_loss,
    orbit_prototype_distillation_loss,
    orbit_relation_loss,
    orbit_feature_loss,
    robust_spherical_orbit_target,
)
from cvsrffi.selective_tangent import (
    angular_sensitivity,
    fingerprint_keep_loss,
    gradient_norm_ratio,
    heteroscedastic_nuisance_loss,
    selective_tangent_loss,
    worst_channel_bucket_accuracy,
)


def test_real_checkpoint_smoke_entrypoint_imports_from_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "code" / "scripts" / "smoke_adv3b02_daot_real_checkpoint.py"),
            "--help",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert "Real-checkpoint, source-shaped, no-query DAOT smoke" in completed.stdout


def test_a0_a8_launcher_binds_the_fasttrust_eff_baseline_route() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    launcher = (
        repository_root / "code" / "scripts" / "launch_phase1_adv3b02_daot_stn_a0_a8_20260901.sh"
    ).read_text(encoding="utf-8")

    assert "FASTTRUST_RC4=true" in launcher
    assert "RC4_ENABLE_NEGATIVE=false" in launcher
    assert "RC4_TOTAL_IDENTITY_EFFECTIVE_BUDGET=0.15" in launcher
    assert "RC4_CLASS_RX_CAP=true" in launcher
    worker = (
        repository_root / "code" / "scripts" / "launch_phase1_adv3b02_muse_ssdg_20260819.sh"
    ).read_text(encoding="utf-8")
    assert "--daot_loss_ablation" in worker
    assert "none|no_z|no_logit|no_proto|relation_on" in worker


def test_manysig_release_keeps_overlapping_days_when_receivers_are_disjoint() -> None:
    source_days, target_days, source_rxs, target_rxs = _resolve_source_target_axes(
        requested_source_days=[1, 2, 3],
        requested_target_days=[0, 1, 2, 3],
        requested_source_receivers=[1, 3, 4, 6, 8],
        requested_target_receivers=[0, 2, 5, 7, 9, 10, 11],
        allow_day_overlap=True,
    )

    assert source_days == [1, 2, 3]
    assert target_days == [0, 1, 2, 3]
    assert source_rxs == [1, 3, 4, 6, 8]
    assert target_rxs == [0, 2, 5, 7, 9, 10, 11]


def test_day_overlap_release_rejects_any_source_target_receiver_overlap() -> None:
    with pytest.raises(ValueError, match="receiver-disjoint"):
        _resolve_source_target_axes(
            requested_source_days=[1, 2, 3],
            requested_target_days=[0, 1, 2, 3],
            requested_source_receivers=[1, 3, 4, 6, 8],
            requested_target_receivers=[0, 2, 5, 7, 8, 10, 11],
            allow_day_overlap=True,
        )


def test_manysig_a0_a7_release_freezes_approved_matrix_and_single_v_protocol() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    launcher = (
        repository_root
        / "code"
        / "scripts"
        / "launch_phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901.sh"
    ).read_text(encoding="utf-8")

    assert "ROWS=(A0 A1 A2 A3 A4 A5 A6 A7)" in launcher
    assert "GPUS=(0 1 2 3 4 5 6 7)" in launcher
    assert "SEED=\"${SEED:-392005}\"" in launcher
    assert "WISIG_TRAIN_DAYS=1,2,3" in launcher
    assert "WISIG_TEST_DAYS=0,1,2,3" in launcher
    assert "WISIG_TRAIN_RXS=1,3,4,6,8" in launcher
    assert "WISIG_TEST_RXS=0,2,5,7,9,10,11" in launcher
    assert "PHASE1_SOURCE_ROLE_PROTOCOL=legacy_l_u_v" in launcher
    assert "SOURCE_VAL_RATIO=0.30" in launcher
    assert "SOURCE_CAL_RATIO=0" in launcher
    assert "SOURCE_SELECT_RATIO=0" in launcher
    assert "ALLOW_SOURCE_TARGET_DAY_OVERLAP=true" in launcher
    assert "TARGET_GROUP_LOADER=test_all_day_unseen_rx" in launcher
    assert "A8" not in launcher


def test_worker_records_actual_single_v_release_roles_in_run_artifacts() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    worker = (
        repository_root / "code" / "scripts" / "launch_phase1_adv3b02_muse_ssdg_20260819.sh"
    ).read_text(encoding="utf-8")

    assert '"ratios": {"L_s": %s, "U_s": %s, "V": %s}' in worker
    assert '"source_role_protocol": "%s"' in worker
    assert 'roles=${PHASE1_SOURCE_ROLE_PROTOCOL}' in worker
    assert 'ratios=${LABELED_RATIO}/${UNLABELED_RATIO}/${SOURCE_VAL_RATIO}' in worker


def test_default_performance_teacher_uses_clean_medium_hard_views() -> None:
    specs = default_teacher_view_specs()

    assert [spec.name for spec in specs] == ["clean", "medium", "hard"]
    assert [spec.scenario for spec in specs] == [
        "clean",
        "leo_clear_weak",
        "leo_low_elev_weak",
    ]
    assert [spec.severity for spec in specs] == pytest.approx([0.0, 0.5, 1.0])


def test_deployment_config_rejects_an_empirical_claim_without_real_statistics() -> None:
    with pytest.raises(ValueError, match="real LEO statistics"):
        DeploymentOrbitConfig(empirical_weight=0.4, has_real_leo_statistics=False)


def test_sparse_joint_direction_is_standardized_deterministic_and_bounded() -> None:
    covariance = torch.tensor(
        [
            [1.0, 0.6, 0.0, 0.0],
            [0.6, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.3, 1.0],
        ]
    )
    first = sample_sparse_joint_direction(covariance, seed=17, max_active=3)
    second = sample_sparse_joint_direction(covariance, seed=17, max_active=3)

    assert torch.equal(first, second)
    assert 1 <= int((first != 0).sum()) <= 3
    assert float(first.norm()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", DAOT_NUISANCE_TANGENT_NAMES)
def test_named_nuisance_tangents_are_deterministic_common_random_views(name: str) -> None:
    x = torch.randn(2, 2, 64)
    direction = torch.tensor([1.0, -0.5])

    first = apply_named_local_nuisance_tangent(
        x,
        name=name,
        direction=direction,
        delta=0.05,
        sample_rate_hz=25e6,
    )
    second = apply_named_local_nuisance_tangent(
        x,
        name=name,
        direction=direction,
        delta=0.05,
        sample_rate_hz=25e6,
    )

    assert first.shape == x.shape
    assert torch.equal(first, second)
    assert not torch.equal(first, x)


def test_ema_loss_scale_normalizer_roundtrips_and_ignores_inactive_zero() -> None:
    normalizer = EMALossScaleNormalizer(momentum=0.5)
    first, first_scales = normalizer.normalize(
        {"orbit_z": torch.tensor(2.0), "orbit_logit": torch.tensor(0.0)},
        active={"orbit_z": True, "orbit_logit": False},
    )
    second, second_scales = normalizer.normalize(
        {"orbit_z": torch.tensor(4.0), "orbit_logit": torch.tensor(0.0)},
        active={"orbit_z": True, "orbit_logit": False},
    )

    assert float(first["orbit_z"]) == pytest.approx(1.0)
    assert first_scales["orbit_z"] == pytest.approx(2.0)
    assert second_scales["orbit_z"] == pytest.approx(3.0)
    assert float(second["orbit_z"]) == pytest.approx(4.0 / 3.0)
    restored = EMALossScaleNormalizer(momentum=0.1)
    restored.load_state_dict(normalizer.state_dict())
    assert restored.state_dict() == normalizer.state_dict()


def test_loss_level_ablation_overrides_are_independent() -> None:
    assert daot_loss_ablation_overrides("none") == {}
    assert daot_loss_ablation_overrides("no_z") == {"daot_lambda_orbit_z": 0.0}
    assert daot_loss_ablation_overrides("no_logit") == {"daot_lambda_orbit_logit": 0.0}
    assert daot_loss_ablation_overrides("no_proto") == {"daot_lambda_orbit_proto": 0.0}
    assert daot_loss_ablation_overrides("relation_on") == {"daot_enable_relation": True}


def test_gradient_ratio_and_worst_bucket_diagnostics_are_numerically_defined() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    base = parameter.square()
    auxiliary = 2.0 * parameter.square()
    ratio = gradient_norm_ratio(auxiliary, base, [parameter])
    bucket = worst_channel_bucket_accuracy(
        predictions=torch.tensor([0, 0, 1, 1]),
        labels=torch.tensor([0, 1, 1, 0]),
        channel_values=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        bins=2,
    )

    assert ratio == pytest.approx(2.0)
    assert bucket["valid_count"] == 4
    assert bucket["bucket_count"] == 2
    assert bucket["worst_accuracy"] == pytest.approx(0.5)


def test_physical_reliability_and_importance_do_not_make_clean_the_only_teacher() -> None:
    meta = {
        "snr_db": torch.tensor([26.0, 14.0]),
        "theta_deg": torch.tensor([70.0, 12.0]),
        "K_db": torch.tensor([20.0, 8.0]),
    }
    reliability = physical_reliability_from_meta(meta, batch_size=2, device=torch.device("cpu"))
    importance = teacher_importance_matrix(batch_size=2, device=torch.device("cpu"))

    assert reliability[0] > reliability[1]
    assert bool(((reliability >= 0.0) & (reliability <= 1.0)).all())
    assert importance.shape == (2, 3)
    assert float(importance[0, 1]) > float(importance[0, 0])


def test_local_tangent_reuses_the_received_iq_without_new_randomness() -> None:
    x = torch.zeros(2, 2, 16)
    x[:, 0] = 1.0
    directions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    first = apply_local_nuisance_tangent(x, directions, delta=0.05, sample_rate_hz=25e6)
    second = apply_local_nuisance_tangent(x, directions, delta=0.05, sample_rate_hz=25e6)

    assert torch.equal(first, second)
    assert not torch.equal(first, x)
    assert bool(torch.isfinite(first).all())


def test_fingerprint_intervention_changes_tx_hardware_directions_deterministically() -> None:
    x = torch.randn(2, 2, 32)

    first = apply_fingerprint_intervention(x, strength=0.05, sample_rate_hz=25e6)
    second = apply_fingerprint_intervention(x, strength=0.05, sample_rate_hz=25e6)

    assert torch.equal(first, second)
    assert not torch.equal(first, x)


def test_stable_orbit_keys_bind_temporal_memory_to_physical_samples() -> None:
    first = stable_orbit_key_tensor(["rx1/day1/sample9", "rx1/day1/sample10"], device=torch.device("cpu"))
    second = stable_orbit_key_tensor(["rx1/day1/sample9", "rx1/day1/sample10"], device=torch.device("cpu"))

    assert torch.equal(first, second)
    assert int(first[0]) != int(first[1])


@pytest.mark.parametrize("ablation_id", [f"A{i}" for i in range(9)])
def test_daot_ablation_matrix_has_a_reachable_configuration(ablation_id: str) -> None:
    config = daot_ablation_overrides(ablation_id)

    assert config["daot_ablation"] == ablation_id
    assert config["use_adv3b02_daot_stn"] is (ablation_id != "A0")
    if ablation_id == "A8":
        assert config["daot_teacher_mode"] == "temporal_memory"
    elif ablation_id != "A0":
        assert config["daot_teacher_mode"] == "three_view"


def test_robust_spherical_target_applies_coverage_floor_and_stays_on_sphere() -> None:
    features = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )
    reliability = torch.tensor([[1.0, 1.0, 0.0]])
    importance = torch.ones_like(reliability)

    target, weights, diagnostics = robust_spherical_orbit_target(
        features,
        reliability=reliability,
        importance=importance,
        coverage_floor=0.15,
        huber_beta_min=0.30,
    )

    assert float(weights[0, 2]) > 0.0
    assert float(target.norm(dim=-1)[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(target[0, 0]) > float(target[0, 1])
    assert float(diagnostics["effective_views"][0]) > 2.0


def test_orbit_feature_loss_uses_recoverability_without_pseudo_labels() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    recoverability = torch.tensor([1.0, 0.0])

    loss = orbit_feature_loss(student, target, recoverability=recoverability)
    loss.backward()

    assert float(loss.detach()) == pytest.approx(1.0)
    assert float(student.grad[0].abs().sum()) > 0.0
    assert float(student.grad[1].abs().sum()) == pytest.approx(0.0)


def test_logit_and_prototype_distillation_use_only_consensus_rows() -> None:
    student_logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True)
    teacher_logits = torch.tensor([[0.0, 4.0], [4.0, 0.0]])
    consensus = torch.tensor([True, False])

    logit_loss = orbit_logit_distillation_loss(
        student_logits,
        teacher_logits,
        consensus=consensus,
        temperature=2.0,
    )
    proto_loss = orbit_prototype_distillation_loss(
        student_similarity=student_logits,
        teacher_similarity=teacher_logits,
        consensus=consensus,
        temperature=2.0,
    )
    (logit_loss + proto_loss).backward()

    assert float(logit_loss.detach()) > 0.0
    assert float(proto_loss.detach()) > 0.0
    assert float(student_logits.grad[0].abs().sum()) > 0.0
    assert float(student_logits.grad[1].abs().sum()) == pytest.approx(0.0)


def test_relation_loss_preserves_teacher_pair_geometry() -> None:
    student = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    pairs = torch.tensor([[0, 1], [1, 2]])

    loss = orbit_relation_loss(student, teacher, pairs=pairs)

    expected = ((0.6 - 0.0) ** 2 + (0.8 - 0.0) ** 2) / 2.0
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test_temporal_orbit_memory_round_trips_state_and_updates_only_valid_keys() -> None:
    memory = TemporalOrbitMemory(momentum=0.8)
    memory.update(
        keys=torch.tensor([10, 20]),
        features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        valid=torch.tensor([True, False]),
    )
    memory.update(
        keys=torch.tensor([10]),
        features=torch.tensor([[0.0, 1.0]]),
        valid=torch.tensor([True]),
    )

    restored = TemporalOrbitMemory(momentum=0.5)
    restored.load_state_dict(memory.state_dict())
    values, found = restored.lookup(torch.tensor([10, 20]))

    assert found.tolist() == [True, False]
    assert values[0].tolist() == pytest.approx([0.8, 0.2])
    assert restored.momentum == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("epoch", "stage", "orbit_scale", "tangent_scale"),
    [
        (20, "A", 0.0, 0.0),
        (21, "B", 0.025, 0.0),
        (60, "B", 1.0, 0.0),
        (61, "C", 1.0, 0.0125),
        (140, "C", 1.0, 1.0),
        (141, "D", 1.0, 1.0),
    ],
)
def test_daot_schedule_has_report_boundaries(
    epoch: int, stage: str, orbit_scale: float, tangent_scale: float
) -> None:
    state = adv3b02_daot_schedule(epoch, total_epochs=200)

    assert state.stage == stage
    assert state.orbit_scale == pytest.approx(orbit_scale)
    assert state.tangent_scale == pytest.approx(tangent_scale)


def test_selective_tangent_controls_nuisance_budget_without_zeroing_mixed_direction() -> None:
    base = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    perturbed = torch.tensor(
        [[math.cos(0.05), math.sin(0.05)], [math.cos(0.10), math.sin(0.10)]]
    )
    sensitivity = angular_sensitivity(base, perturbed, delta=0.05)

    loss = selective_tangent_loss(
        sensitivity,
        budgets=torch.tensor([0.0, 1.5]),
        valid=torch.tensor([True, True]),
    )

    assert sensitivity.tolist() == pytest.approx([1.0, 2.0], abs=2e-4)
    assert float(loss) == pytest.approx((1.0**2 + 0.5**2) / 2.0, abs=2e-4)


def test_fingerprint_keep_penalizes_only_sensitivity_below_the_floor() -> None:
    loss = fingerprint_keep_loss(
        fingerprint_sensitivity=torch.tensor([0.2, 0.8]),
        minimum=torch.tensor([0.5, 0.5]),
    )

    assert float(loss) == pytest.approx((0.3**2 + 0.0) / 2.0)


def test_heteroscedastic_nuisance_loss_ignores_invalid_rows() -> None:
    mean = torch.tensor([[1.0, 3.0], [100.0, 100.0]])
    log_variance = torch.zeros_like(mean)
    target = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    loss = heteroscedastic_nuisance_loss(
        mean,
        log_variance,
        target,
        valid=torch.tensor([True, False]),
    )

    assert float(loss) == pytest.approx((1.0**2 + 2.0**2) / 2.0)


def test_nuisance_head_returns_bounded_mean_and_log_variance() -> None:
    head = NuisanceHeteroscedasticHead(embedding_dim=8, nuisance_dim=4)
    mean, log_variance = head(torch.randn(3, 8))

    assert mean.shape == (3, 4)
    assert log_variance.shape == (3, 4)
    assert bool((log_variance >= -8.0).all())
    assert bool((log_variance <= 8.0).all())


@pytest.mark.filterwarnings("ignore:.*torch.cuda.amp.autocast.*:FutureWarning")
def test_real_dual_model_exposes_nuisance_prediction_only_when_opted_in() -> None:
    legacy = build_dual_model(3, 2, model_size="S", input_len=256)
    daot = build_dual_model(
        3,
        2,
        model_size="S",
        input_len=256,
        use_daot_nuisance_head=True,
        daot_nuisance_dim=9,
    )
    x = torch.randn(2, 2, 256)
    legacy.eval()
    daot.eval()
    with torch.no_grad():
        legacy_out = legacy(x, return_aux=True)
        daot_out = daot(x, return_aux=True)

    assert "daot_nuisance_mean" not in legacy_out
    assert daot_out["daot_nuisance_mean"].shape == (2, 9)
    assert daot_out["daot_nuisance_log_variance"].shape == (2, 9)
    assert not any(key.startswith("daot_nuisance_head.") for key in legacy.state_dict())


def test_daot_cli_is_opt_in_but_three_view_is_its_default_mode() -> None:
    args = build_arg_parser().parse_args(["--output_dir", "run"])

    assert args.use_adv3b02_daot_stn is False
    assert args.daot_teacher_mode == "three_view"
    assert args.daot_lambda_orbit_z == pytest.approx(0.50)
    assert args.daot_lambda_tangent == pytest.approx(0.05)
    assert args.daot_lambda_fingerprint == pytest.approx(0.10)


def test_daot_validation_enables_only_its_nuisance_head_and_ema_teacher() -> None:
    args = build_arg_parser().parse_args(
        ["--output_dir", "run", "--use_adv3b02_daot_stn", "true"]
    )

    _validate_daot_config(args)

    assert args.use_daot_nuisance_head is True
    assert args.use_ema_teacher is True
    assert args.daot_claim_label == "deployment-proxy matched"


def test_daot_validation_applies_loss_axis_after_mechanism_ablation() -> None:
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "run",
            "--daot_ablation",
            "A3",
            "--daot_loss_ablation",
            "no_z",
        ]
    )

    _validate_daot_config(args)

    assert args.use_adv3b02_daot_stn is True
    assert args.daot_teacher_view_count == 3
    assert args.daot_aggregation == "robust_deployment"
    assert args.daot_lambda_orbit_z == 0.0


def test_daot_batch_objective_keeps_feature_and_consensus_routes_separate() -> None:
    student_clean = {
        "z_id": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        "tx_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
    }
    student_channel = {
        "z_id": torch.tensor([[0.8, 0.2], [0.2, 0.8]], requires_grad=True),
        "tx_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
    }
    teachers = [
        {"z_id": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "tx_logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]])},
        {"z_id": torch.tensor([[0.9, 0.1], [0.1, 0.9]]), "tx_logits": torch.tensor([[3.0, 0.0], [3.0, 0.0]])},
        {"z_id": torch.tensor([[0.8, 0.2], [0.2, 0.8]]), "tx_logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]])},
    ]

    result = compute_daot_batch_objective(
        student_clean=student_clean,
        student_channel=student_channel,
        teacher_views=teachers,
        reliability=torch.ones(2, 3),
        importance=torch.ones(2, 3),
        recoverability=torch.ones(2),
        orbit_scale=1.0,
        tangent_scale=0.0,
        weights={"orbit_z": 0.5, "orbit_logit": 0.2, "orbit_proto": 0.0},
        coverage_floor=0.15,
        huber_beta_min=0.30,
        temperature=3.0,
        loss_normalizer=EMALossScaleNormalizer(momentum=0.95),
    )
    result["loss"].backward()

    assert float(result["components"]["orbit_z"].detach()) >= 0.0
    assert result["diagnostics"]["loss_scale_orbit_z"] > 0.0
    assert result["diagnostics"]["consensus_mask"].tolist() == [True, False]
    assert float(student_channel["z_id"].grad[1].abs().sum()) > 0.0
    assert float(student_channel["tx_logits"].grad[1].abs().sum()) == pytest.approx(0.0)


def test_labeled_step_runs_the_default_three_teacher_views() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.eye_(self.proj.weight)

        def forward(self, x, **kwargs):
            pooled = x.mean(dim=-1)
            z = self.proj(pooled)
            return {"z_id": z, "tx_logits": z, "z_dom": z}

    def fake_sat(x, scenario, args, *, gen, return_meta):
        offsets = {"leo_clear_weak": 0.10, "leo_low_elev_weak": 0.20, "leo_rain_weak": 0.30}
        shifted = x.clone()
        shifted[:, 1] = shifted[:, 1] + offsets[scenario]
        batch = x.shape[0]
        meta = {
            "snr_db": torch.full((batch,), 24.0 if scenario == "leo_clear_weak" else 16.0),
            "theta_deg": torch.full((batch,), 60.0 if scenario == "leo_clear_weak" else 20.0),
            "K_db": torch.full((batch,), 18.0 if scenario == "leo_clear_weak" else 9.0),
            "cfo_hz": torch.zeros(batch),
            "residual_cfo_hz": torch.zeros(batch),
            "fD_hz": torch.zeros(batch),
            "pl_db": torch.zeros(batch),
            "h_km": torch.full((batch,), 1000.0),
            "state": torch.zeros(batch),
        }
        return shifted, meta

    args = build_arg_parser().parse_args(
        ["--output_dir", "run", "--use_adv3b02_daot_stn", "true"]
    )
    _validate_daot_config(args)
    args.daot_diagnostic_epochs = "60"
    model = ToyModel()
    teacher = ToyModel()
    teacher.load_state_dict(model.state_dict())
    x = torch.randn(3, 2, 16)
    clean = model(x)

    result = _compute_daot_labeled_step(
        model=model,
        ema_model=teacher,
        student_clean=clean,
        x_clean=x,
        y_clean=torch.tensor([0, 1, 0]),
        d_clean=None,
        args=args,
        epoch=60,
        batch_idx=1,
        apply_sat_fn=fake_sat,
        prototype_matrix=None,
    )
    result["loss"].backward()

    assert result["diagnostics"]["teacher_view_count"] == 3.0
    assert result["diagnostics"]["teacher_mode"] == "three_view"
    assert set(result["diagnostics"]["named_nuisance_sensitivity"]) == set(
        DAOT_NUISANCE_TANGENT_NAMES
    )
    assert float(model.proj.weight.grad.abs().sum()) > 0.0


def test_unlabeled_step_has_no_label_or_pseudo_label_input() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x, **kwargs):
            z = self.proj(x.mean(dim=-1))
            return {"z_id": z, "tx_logits": z}

    def fake_sat(x, scenario, args, *, gen, return_meta):
        shifted = x + (0.05 if scenario == "leo_clear_weak" else 0.10)
        batch = x.shape[0]
        meta = {
            "snr_db": torch.full((batch,), 20.0),
            "theta_deg": torch.full((batch,), 45.0),
            "K_db": torch.full((batch,), 12.0),
        }
        return shifted, meta

    args = build_arg_parser().parse_args(
        ["--output_dir", "run", "--use_adv3b02_daot_stn", "true"]
    )
    _validate_daot_config(args)
    model = ToyModel()
    teacher = ToyModel()
    teacher.load_state_dict(model.state_dict())
    x = torch.randn(4, 2, 8)
    student = model(x)
    with torch.no_grad():
        teacher_clean = teacher(x)

    result = _compute_daot_unlabeled_step(
        model=model,
        ema_model=teacher,
        teacher_clean=teacher_clean,
        student_strong=student,
        x_unlabeled=x,
        d_unlabeled=None,
        args=args,
        epoch=80,
        batch_idx=1,
        apply_sat_fn=fake_sat,
        prototype_matrix=None,
    )
    result["loss"].backward()

    assert result["diagnostics"]["route"] == "all_recoverable_feature_consensus_logits"
    assert float(model.proj.weight.grad.abs().sum()) > 0.0
