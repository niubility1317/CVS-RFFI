from pathlib import Path


TRACEABILITY = Path("analysis/adv3b02_muse_ssdg_traceability_20260819.md")
FORMAL_REPORT = Path(
    "automation_reports/CV-SincNet/phase1_adv3b02_muse_ssdg_20260819/report.md"
)


def test_final_fix_traceability_names_real_call_chain_evidence() -> None:
    text = TRACEABILITY.read_text(encoding="utf-8")
    for token in (
        "test_m2_fusion_uses_global_local_prototype_and_l_s_prior_alignment",
        "test_proto_momentum_boundary_is_095_then_099_at_s3b_and_s3c",
        "test_epoch_181_freezes_muse_statistics_prior_and_local_teacher_state",
        "test_m3_sha_mask_selects_exactly_one_identity_student_per_row",
        "test_muse_can_delegate_final_target_eval_without_changing_legacy",
        "test_strict_reconstruction_failure_exits_before_metrics_are_written",
        "test_active_phase1_row_factories_emit_parser_valid_final_only_selection",
    ):
        assert token in text
    assert "|FFR-1|final-fix brief|" in text
    assert "|FFR-7|final-fix brief|" in text
    assert "|pending|待RED→GREEN|" not in text


def test_formal_report_keeps_runtime_performance_pending_after_final_fix() -> None:
    text = FORMAL_REPORT.read_text(encoding="utf-8")
    assert "## Final fix wave（FFR-1至FFR-7）" in text
    assert "strict checkpoint reconstruction" in text
    assert "DELEGATED_TO_MUSE_LAUNCHER" in text
    assert "真实M0–M3训练：未运行" in text
    assert "真实clean及三LEO场景性能：未产生" in text
