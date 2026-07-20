import numpy as np
import pytest

from cvsrffi.stage2_qk_d81_lgf import (
    Phase1LockedConfig,
    QKD81LGFError,
    audit_quantized_margin,
    build_support_bank,
    fuse_with_base_probabilities,
    normalize_three_blocks,
    score_qknn_logits,
)


def _features(rows: int, seed: int = 9701) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, 288)).astype(np.float32)


def _config(**overrides) -> Phase1LockedConfig:
    values = dict(
        beta=8.0,
        temp_base=1.0,
        temp_qk=0.2,
        eta_max=0.5,
        phase1_receipt_sha256="2" * 64,
        margin_audit_sha256="3" * 64,
    )
    values.update(overrides)
    return Phase1LockedConfig(**values)


def test_three_block_normalization_has_equal_block_energy() -> None:
    normalized = normalize_three_blocks(_features(4))
    for start, stop in ((0, 160), (160, 256), (256, 288)):
        np.testing.assert_allclose(
            np.linalg.norm(normalized[:, start:stop], axis=1),
            np.full(4, 1.0 / np.sqrt(3.0)),
            atol=2e-6,
        )
    assert normalized.flags.writeable is False


def test_support_order_is_canonical_and_predictions_are_order_invariant() -> None:
    support = _features(6)
    labels = np.asarray(["a", "a", "b", "b", "c", "c"])
    config = _config()
    bank_a = build_support_bank(
        support,
        labels,
        ("a", "b", "c"),
        config=config,
        support_only_eta=0.25,
        eta_source="support_crossfit_phase1_smoothed",
        support_cv_receipt_sha256="4" * 64,
    )
    order = np.asarray([5, 0, 3, 2, 1, 4])
    bank_b = build_support_bank(
        support[order],
        labels[order],
        ("a", "b", "c"),
        config=config,
        support_only_eta=0.25,
        eta_source="support_crossfit_phase1_smoothed",
        support_cv_receipt_sha256="4" * 64,
    )
    np.testing.assert_array_equal(bank_a.codes_qint8, bank_b.codes_qint8)
    np.testing.assert_array_equal(bank_a.scales_fp16, bank_b.scales_fp16)
    query = _features(3, seed=9702)
    np.testing.assert_allclose(
        score_qknn_logits(bank_a, query), score_qknn_logits(bank_b, query)
    )


def test_duplicate_support_does_not_create_a_class_count_bonus() -> None:
    prototype = _features(2)
    bank = build_support_bank(
        np.stack([prototype[0], prototype[0], prototype[1], prototype[1]]),
        ("a", "a", "b", "b"),
        ("a", "b"),
        config=_config(),
    )
    logits = score_qknn_logits(bank, prototype[:1])
    single = build_support_bank(
        prototype,
        ("a", "b"),
        ("a", "b"),
        config=_config(),
    )
    np.testing.assert_allclose(logits, score_qknn_logits(single, prototype[:1]), atol=1e-6)


def test_k1_cannot_fit_target_eta_and_zero_eta_is_exact_fallback() -> None:
    support = _features(3)
    with pytest.raises(QKD81LGFError, match="K1 cannot fit eta"):
        build_support_bank(
            support,
            ("a", "b", "c"),
            ("a", "b", "c"),
            config=_config(),
            support_only_eta=0.1,
            eta_source="support_crossfit_phase1_smoothed",
        )
    bank = build_support_bank(
        support,
        ("a", "b", "c"),
        ("a", "b", "c"),
        config=_config(),
    )
    base = np.asarray([[0.2, 0.7, 0.1], [0.4, 0.2, 0.4]], dtype=np.float32)
    fused, prediction, audit = fuse_with_base_probabilities(
        bank, base, _features(2, seed=9703), base_classes=("a", "b", "c")
    )
    assert fused is base
    np.testing.assert_array_equal(prediction, np.asarray(["b", "a"]))
    assert audit["qknn_branch_executed"] is False


def test_query_batching_cannot_change_nonzero_eta_predictions() -> None:
    support = _features(6)
    bank = build_support_bank(
        support,
        ("a", "a", "b", "b", "c", "c"),
        ("a", "b", "c"),
        config=_config(),
        support_only_eta=0.3,
        eta_source="support_crossfit_phase1_smoothed",
        support_cv_receipt_sha256="5" * 64,
    )
    query = _features(4, seed=9704)
    base = np.full((4, 3), 1.0 / 3.0, dtype=np.float32)
    together = fuse_with_base_probabilities(
        bank, base, query, base_classes=("a", "b", "c")
    )[0]
    separate = np.concatenate(
        [fuse_with_base_probabilities(
            bank,
            base[i : i + 1],
            query[i : i + 1],
            base_classes=("a", "b", "c"),
        )[0]
         for i in range(4)],
        axis=0,
    )
    np.testing.assert_allclose(together, separate, atol=1e-7)
    assert bank.persistent_state_bytes >= (
        bank.codes_qint8.nbytes
        + bank.scales_fp16.nbytes
        + bank.class_indices_int16.nbytes
    )


def test_phase1_lock_rejects_unbounded_or_unidentified_configuration() -> None:
    with pytest.raises(QKD81LGFError):
        _config(eta_max=1.1)
    with pytest.raises(QKD81LGFError):
        _config(phase1_receipt_sha256="bad")


def test_unbalanced_support_and_unsealed_nonzero_eta_are_rejected() -> None:
    with pytest.raises(QKD81LGFError, match="balanced K-shot"):
        build_support_bank(
            _features(5),
            ("a", "a", "b", "b", "b"),
            ("a", "b"),
            config=_config(),
        )
    with pytest.raises(QKD81LGFError, match="sealed support-CV receipt"):
        build_support_bank(
            _features(4),
            ("a", "a", "b", "b"),
            ("a", "b"),
            config=_config(),
            support_only_eta=0.2,
            eta_source="support_crossfit_phase1_smoothed",
        )


def test_phase1_margin_audit_reports_int8_argmax_and_margin_flips() -> None:
    support = _features(6)
    labels = ("a", "a", "b", "b", "c", "c")
    bank = build_support_bank(support, labels, ("a", "b", "c"), config=_config())
    audit = audit_quantized_margin(bank, support, labels, _features(12, seed=9705))
    assert 0.0 <= audit["top1_agreement"] <= 1.0
    assert 0.0 <= audit["margin_sign_flip_rate"] <= 1.0
    assert audit["validation_row_count"] == 12
