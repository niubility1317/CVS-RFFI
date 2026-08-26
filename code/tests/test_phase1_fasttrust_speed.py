import inspect
import math
from types import SimpleNamespace

import torch
from torch import nn

from SSDG import train_ssdg


def test_fasttrust_defaults_to_independent_u_batch_256_without_changing_l_batch():
    args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "true", "--batch_size", "128"]
    )

    assert args.batch_size == 128
    assert args.muse_unlabeled_batch_size == 256
    assert train_ssdg._resolve_unlabeled_batch_size(args) == 256


def test_non_muse_loader_keeps_the_legacy_batch_size():
    args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "false", "--batch_size", "96"]
    )
    assert train_ssdg._resolve_unlabeled_batch_size(args) == 96


def test_muse_u_loader_keeps_tail_batch_and_m0_uses_fasttrust_lr():
    muse_m0 = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "true", "--muse_level", "M0"]
    )
    legacy = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "false"]
    )

    assert train_ssdg._unlabeled_drop_last(muse_m0) is False
    assert train_ssdg._unlabeled_drop_last(legacy) is True
    assert train_ssdg._fasttrust_lr_enabled(muse_m0) is True


def test_fasttrust_lr_schedule_has_warmup_cosine_and_backbone_tail_scales():
    assert train_ssdg._fasttrust_lr_scales(1) == (0.2, 1.0)
    assert train_ssdg._fasttrust_lr_scales(5) == (1.0, 1.0)
    global_160, backbone_160 = train_ssdg._fasttrust_lr_scales(160)
    assert abs(global_160 - 0.1) < 1e-12
    assert backbone_160 == 1.0
    assert train_ssdg._fasttrust_lr_scales(161) == (0.1, 0.2)
    assert train_ssdg._fasttrust_lr_scales(180) == (0.1, 0.2)
    assert train_ssdg._fasttrust_lr_scales(181) == (0.1, 0.05)
    assert train_ssdg._fasttrust_lr_scales(200) == (0.1, 0.05)


def test_fasttrust_epoch_resource_metrics_report_realized_throughput_and_peak_memory():
    metrics = train_ssdg._fasttrust_epoch_resource_metrics(
        u_samples_per_step=256.0,
        u_forward_samples_per_step=512.0,
        steps=10,
        elapsed_s=5.0,
        peak_memory_bytes=3 * 1024**3,
    )

    assert metrics == {
        "muse/u_samples_per_s": 512.0,
        "muse/u_forward_samples_per_s": 1024.0,
        "muse/peak_cuda_memory_mb": 3072.0,
    }


def test_qb3_gradient_telemetry_uses_first_real_batch_from_one_based_loader():
    muse_state = {"fasttrust_rc4": True}
    telemetry_epochs = {1, 41, 91}

    assert train_ssdg._rc4_should_collect_gradient_telemetry(
        muse_state,
        batch_idx=1,
        epoch=41,
        telemetry_epochs=telemetry_epochs,
    )
    assert not train_ssdg._rc4_should_collect_gradient_telemetry(
        muse_state,
        batch_idx=2,
        epoch=41,
        telemetry_epochs=telemetry_epochs,
    )
    assert not train_ssdg._rc4_should_collect_gradient_telemetry(
        muse_state,
        batch_idx=1,
        epoch=40,
        telemetry_epochs=telemetry_epochs,
    )


def test_qb3_gradient_telemetry_uses_actual_shared_parameter_graph_and_cosine():
    shared = nn.Linear(2, 2, bias=False)
    x = torch.tensor([[1.0, 2.0]])
    logits = shared(x)
    labeled = logits[0, 0]
    aligned = 2.0 * logits[0, 0]
    opposed = -logits[0, 0]

    metrics = train_ssdg._rc4_gradient_relationships(
        {"labeled": labeled, "hard": aligned, "partial_set": opposed},
        tuple(shared.parameters()),
    )

    assert metrics["norms"]["labeled"] > 0.0
    assert metrics["norms"]["hard"] == 2.0 * metrics["norms"]["labeled"]
    assert metrics["cosines_to_labeled"]["hard"] == 1.0
    assert metrics["cosines_to_labeled"]["partial_set"] == -1.0


def test_qb3_gradient_telemetry_marks_disconnected_loss_as_unavailable_not_zero():
    shared = nn.Linear(2, 1, bias=False)
    disconnected = torch.tensor(3.0, requires_grad=True)

    metrics = train_ssdg._rc4_gradient_relationships(
        {"labeled": shared(torch.ones(1, 2)).sum(), "hard": disconnected},
        tuple(shared.parameters()),
    )

    assert math.isnan(metrics["norms"]["hard"])
    assert math.isnan(metrics["cosines_to_labeled"]["hard"])


def test_qb3_first_nonfinite_gradient_names_parameter_and_counts_bad_elements():
    model = nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float("nan"), float("inf")]])

    detail = train_ssdg._first_nonfinite_gradient(model)

    assert detail == {
        "parameter_name": "weight",
        "nonfinite_elements": 2,
        "nan_elements": 1,
        "posinf_elements": 1,
        "neginf_elements": 0,
    }


def test_qb3_nonfinite_gradient_is_captured_before_gradient_clipping():
    source = inspect.getsource(train_ssdg.train)
    unscale_at = source.index("scaler.unscale_(optimizer)")
    inspect_at = source.index(
        "first_nonfinite_gradient = _first_nonfinite_gradient(model)",
        unscale_at,
    )
    clip_at = source.index("torch.nn.utils.clip_grad_norm_", inspect_at)

    assert unscale_at < inspect_at < clip_at


def test_fasttrust_epoch_stage_timing_reports_accounted_and_other_seconds():
    metrics = train_ssdg._fasttrust_epoch_stage_timing_metrics(
        train_batches_s=10.0,
        base_validation_s=2.0,
        heavy_source_validation_s=3.0,
        checkpoint_io_s=4.0,
        epoch_elapsed_s=20.0,
    )

    assert metrics == {
        "muse/time_train_batches_s": 10.0,
        "muse/time_base_validation_s": 2.0,
        "muse/time_heavy_source_validation_s": 3.0,
        "muse/time_checkpoint_io_s": 4.0,
        "muse/time_other_s": 1.0,
    }


def test_rc4_defaults_use_safe_domain_scale_and_nonfinite_guard():
    args = train_ssdg.build_arg_parser().parse_args(["--output_dir", "out"])

    assert args.rc4_lambda_domain == 0.16
    assert args.rc4_nonfinite_guard_min_count == 8
    assert args.rc4_nonfinite_guard_fraction == 0.05


def test_fasttrust_log_snapshot_never_retains_autograd_graphs():
    source = torch.tensor(2.0, requires_grad=True)
    derived = source.square()
    snapshot = train_ssdg._detach_log_mapping({"loss": derived, "count": 3.0})

    assert torch.is_tensor(snapshot["loss"])
    assert snapshot["loss"].item() == 4.0
    assert snapshot["loss"].requires_grad is False
    assert snapshot["loss"].grad_fn is None
    assert snapshot["count"] == 3.0


def test_rc4_nonfinite_guard_requires_both_absolute_count_and_fraction():
    assert not train_ssdg._rc4_nonfinite_guard_triggered(7, 7, min_count=8, fraction=0.05)
    assert not train_ssdg._rc4_nonfinite_guard_triggered(8, 200, min_count=8, fraction=0.05)
    assert train_ssdg._rc4_nonfinite_guard_triggered(8, 100, min_count=8, fraction=0.05)


def test_sparse_heavy_source_val_schedule_runs_56_times_for_e200():
    args = SimpleNamespace(
        source_val_heavy_eval_start_epoch=1,
        source_val_heavy_eval_interval=5,
        source_val_heavy_eval_final_window=20,
        source_val_heavy_eval_final_interval=1,
    )
    selected = [
        epoch
        for epoch in range(1, 201)
        if train_ssdg._should_run_source_val_heavy_eval(epoch, 200, args)
    ]

    assert len(selected) == 56
    assert selected[:3] == [5, 10, 15]
    assert selected[-20:] == list(range(181, 201))


class _CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, **_kwargs):
        self.calls += 1
        flat = x.flatten(1)
        return {
            "tx_logits": flat[:, :3],
            "z_id": flat[:, :4],
            "z_dom": flat[:, -4:],
            "constant": "kept",
        }


def test_fused_student_forward_matches_two_deterministic_views_with_one_call():
    strong = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8)
    nuisance = strong + 100.0
    domains = torch.tensor([0, 1])
    model = _CountingModel()

    outputs = train_ssdg._forward_muse_student_views(
        model,
        strong,
        nuisance,
        domains,
        grl_lambda=0.1,
        fused=True,
    )

    assert model.calls == 1
    assert torch.equal(outputs["strong"]["z_id"], strong.flatten(1)[:, :4])
    assert torch.equal(outputs["nuisance"]["z_id"], nuisance.flatten(1)[:, :4])
    assert outputs["strong"]["constant"] == "kept"


def test_disabled_nuisance_branch_does_not_add_a_forward():
    model = _CountingModel()
    strong = torch.zeros(2, 2, 8)

    outputs = train_ssdg._forward_muse_student_views(
        model,
        strong,
        None,
        torch.tensor([0, 1]),
        grl_lambda=0.1,
        fused=True,
    )

    assert model.calls == 1
    assert outputs["nuisance"] is None


def test_anchor_logit_cache_is_opt_in_and_uses_dense_base_index_lookup():
    defaults = train_ssdg.build_arg_parser().parse_args(["--output_dir", "out"])
    enabled = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--rc4_cache_anchor_logits", "true"]
    )
    assert defaults.rc4_cache_anchor_logits is False
    assert enabled.rc4_cache_anchor_logits is True

    cache = train_ssdg._dense_anchor_logit_cache(
        torch.tensor([7, 2, 5]),
        torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]]),
        device=torch.device("cpu"),
    )
    metadata = {"base_index": torch.tensor([5, 7, 2])}

    actual = train_ssdg._lookup_anchor_logits(
        cache, metadata, expected_count=3, device=torch.device("cpu")
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[0.5, 0.5], [0.7, 0.3], [0.2, 0.8]]),
    )


def test_anchor_logit_cache_fails_closed_for_missing_sample_id():
    cache = train_ssdg._dense_anchor_logit_cache(
        torch.tensor([1]), torch.tensor([[1.0, 0.0]]), device=torch.device("cpu")
    )

    try:
        train_ssdg._lookup_anchor_logits(
            cache,
            {"base_index": torch.tensor([2])},
            expected_count=1,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "missing base_index" in str(exc)
    else:
        raise AssertionError("cache lookup must reject an uncached sample")


def test_anchor_logit_cache_preserves_live_amp_dtype():
    logits = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float16)
    cache = train_ssdg._dense_anchor_logit_cache(
        torch.tensor([3, 1]), logits, device=torch.device("cpu")
    )

    actual = train_ssdg._lookup_anchor_logits(
        cache,
        {"base_index": torch.tensor([1, 3])},
        expected_count=2,
        device=torch.device("cpu"),
    )

    assert cache["logits"].dtype == torch.float16
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, logits.flip(0))


def test_anchor_cache_precomputation_restores_training_rng_state():
    source = inspect.getsource(train_ssdg.train)
    capture_at = source.index("pre_cache_rng = _capture_training_rng_state()")
    build_at = source.index("_build_rc4_anchor_logit_cache(", capture_at)
    restore_at = source.index("_restore_training_rng_state(pre_cache_rng)", build_at)

    assert capture_at < build_at < restore_at


def test_anchor_cache_uses_the_same_amp_context_as_live_anchor_forward():
    builder = inspect.getsource(train_ssdg._build_rc4_anchor_logit_cache)
    training = inspect.getsource(train_ssdg.train)

    assert "with autocast(enabled=bool(amp_enabled))" in builder
    assert "amp_enabled=bool(args.amp and device.type == \"cuda\")" in training
