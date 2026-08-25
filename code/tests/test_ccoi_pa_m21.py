import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.ccoi_pa import CCOIPASidecar, PAChallengeEncoder  # noqa: E402
from cvsrffi.ccoi_pa_m21 import (  # noqa: E402
    FactorRow,
    SidecarArchitectureConfig,
    build_fold_records,
    build_relation_indices,
    build_sidecar_v3_payload,
    common_anchor_mask,
    compose_factor_rows,
    duplicate_audit,
    load_sidecar_v3,
    migrate_v2_challenge_encoder,
    evaluate_stage_a,
    fold_macro_nmse,
    run_factor_matrix,
    split_v_select_retro,
)


def _config(**overrides):
    base = SidecarArchitectureConfig(
        input_length=256,
        token_length=64,
        stride=16,
        q_dim=8,
        challenge_hidden_dim=12,
        codebook_size=10,
        response_dim=9,
        operator_dim=7,
        pa_channels=6,
        num_classes=4,
        num_domains=3,
        holdout_anchor_policy="all_nonoverlap_folds",
        conditioned=True,
        pa_map_contract="core90_pa_token_map_v1",
    )
    return replace(base, **overrides)


def _sidecar(config):
    encoder = PAChallengeEncoder(
        token_length=config.token_length,
        stride=config.stride,
        q_dim=config.q_dim,
        codebook_size=config.codebook_size,
        hidden_dim=config.challenge_hidden_dim,
        num_tx=config.num_classes,
        num_rx=config.num_domains,
    )
    return CCOIPASidecar(
        pa_channels=config.pa_channels,
        num_classes=config.num_classes,
        challenge_encoder=encoder,
        q_dim=config.q_dim,
        response_dim=config.response_dim,
        operator_dim=config.operator_dim,
    )


def _v3_payload(config=None):
    config = config or _config()
    return build_sidecar_v3_payload(
        _sidecar(config),
        row="C4p",
        base_checkpoint="base.pth",
        architecture_config=config,
        fusion_alpha=0.0,
        fusion_scale=1.0,
    )


def test_v3_sidecar_round_trip_preserves_parameter_and_nonparameter_geometry():
    config = _config()
    original = _sidecar(config)
    payload = build_sidecar_v3_payload(
        original,
        row="C4p",
        base_checkpoint="base.pth",
        architecture_config=config,
        fusion_alpha=0.0,
        fusion_scale=1.0,
    )

    restored = load_sidecar_v3(payload, expected_config=config, device=torch.device("cpu"))

    assert restored.challenge_encoder.token_length == 64
    assert restored.challenge_encoder.stride == 16
    assert set(restored.state_dict()) == set(original.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize(
    ("field", "value"),
    (("token_length", 32), ("stride", 8), ("pa_map_contract", "wrong_contract")),
)
def test_v3_sidecar_rejects_semantic_geometry_drift(field, value):
    payload = _v3_payload()

    with pytest.raises(ValueError, match=field):
        load_sidecar_v3(
            payload,
            expected_config=_config(**{field: value}),
            device=torch.device("cpu"),
        )


def test_v3_sidecar_requires_architecture_config():
    payload = _v3_payload()
    payload.pop("architecture_config")

    with pytest.raises(ValueError, match="architecture_config"):
        load_sidecar_v3(payload, expected_config=_config(), device=torch.device("cpu"))


def test_v2_challenge_migration_requires_explicit_mode():
    config = _config()
    old_sidecar = _sidecar(config)
    payload = {
        "schema": "cvs.phase1.ccoi_pa_sidecar.v2",
        "row": "C4",
        "state_dict": old_sidecar.state_dict(),
        "sample_level_source_state_included": False,
    }

    with pytest.raises(ValueError, match="legacy_migration_mode"):
        migrate_v2_challenge_encoder(
            payload,
            architecture_config=config,
            device=torch.device("cpu"),
            legacy_migration_mode=False,
        )


def test_v2_migration_loads_only_challenge_encoder_and_freezes_it():
    config = _config()
    old_sidecar = _sidecar(config)
    payload = {
        "schema": "cvs.phase1.ccoi_pa_sidecar.v2",
        "row": "C4",
        "state_dict": old_sidecar.state_dict(),
        "sample_level_source_state_included": False,
    }

    encoder = migrate_v2_challenge_encoder(
        payload,
        architecture_config=config,
        device=torch.device("cpu"),
        legacy_migration_mode=True,
    )

    assert encoder.token_length == config.token_length
    assert encoder.stride == config.stride
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, old_sidecar.state_dict()[f"challenge_encoder.{key}"])


def _split_metadata():
    rows = []
    base_index = 100
    for tx in (0, 1):
        for receiver in (0, 1):
            for block in range(8):
                for offset in (0, 1):
                    rows.append((tx, receiver, 0, 1, block * 10 + offset, base_index))
                    base_index += 1
    columns = tuple(zip(*rows))
    return {
        "tx": torch.tensor(columns[0]),
        "receiver": torch.tensor(columns[1]),
        "day": torch.tensor(columns[2]),
        "eq": torch.tensor(columns[3]),
        "sig_i": torch.tensor(columns[4]),
        "base_index": torch.tensor(columns[5]),
    }


def test_retro_split_is_reproducible_and_disjoint_by_capture_block():
    metadata = _split_metadata()

    first = split_v_select_retro(metadata, seed=17, block_candidates=(10,))
    second = split_v_select_retro(metadata, seed=17, block_candidates=(10,))

    assert first == second
    assert set(first.fit_indices).isdisjoint(first.audit_indices)
    assert set(first.fit_indices).isdisjoint(first.guard_indices)
    assert set(first.audit_indices).isdisjoint(first.guard_indices)
    assert first.base_index_overlap_count == 0
    for group, role in first.role_by_group.items():
        members = [
            i
            for i in range(len(metadata["tx"]))
            if (
                int(metadata["tx"][i]),
                int(metadata["receiver"][i]),
                int(metadata["day"][i]),
                int(metadata["eq"][i]),
                int(metadata["sig_i"][i]) // first.block_size,
            )
            == group
        ]
        expected = set(getattr(first, f"{role}_indices"))
        assert set(members).issubset(expected)


def test_retro_split_places_a_guard_between_fit_and_audit_blocks_per_cell():
    split = split_v_select_retro(_split_metadata(), seed=9, block_candidates=(10,))

    by_cell = {}
    for group, role in split.role_by_group.items():
        by_cell.setdefault(group[:4], {})[group[4]] = role
    for block_roles in by_cell.values():
        for block, role in block_roles.items():
            if role not in {"fit", "audit"}:
                continue
            opposite = "audit" if role == "fit" else "fit"
            assert block_roles.get(block - 1) != opposite
            assert block_roles.get(block + 1) != opposite


def test_duplicate_audit_reports_only_aggregates_and_finds_cross_role_duplicate():
    metadata = _split_metadata()
    split = split_v_select_retro(metadata, seed=17, block_candidates=(10,))
    packet_count = len(metadata["tx"])
    generator = torch.Generator().manual_seed(4)
    iq = torch.randn(packet_count, 2, 16, generator=generator)
    fit_index = split.fit_indices[0]
    audit_index = next(
        index
        for index in split.audit_indices
        if int(metadata["tx"][index]) == int(metadata["tx"][fit_index])
        and int(metadata["receiver"][index]) == int(metadata["receiver"][fit_index])
    )
    iq[audit_index] = iq[fit_index]

    result = duplicate_audit(iq, metadata, split, projection_dim=8, seed=23)

    assert result["base_index_overlap_count"] == 0
    assert result["exact_duplicate_pair_count"] >= 1
    assert result["near_similarity_gt_0_999_rate"] > 0.0
    assert result["sample_level_state_persisted"] is False
    assert "exact_hashes" not in result
    assert "projection_rows" not in result
    assert set(result["nearest_similarity_quantiles"]) == {"q50", "q90", "q95", "q99", "max"}
    assert set(result["nearest_sig_gap_quantiles"]) == {"q50", "q90", "q95", "q99", "max"}


def test_all_four_nonoverlap_folds_are_emitted_with_disjoint_raw_masks():
    config = _config()
    sidecar = _sidecar(config).eval()
    generator = torch.Generator().manual_seed(14)
    x = torch.randn(2, 2, config.input_length, generator=generator)
    pa_map = torch.randn(2, config.pa_channels, 31, generator=generator)

    records = build_fold_records(
        sidecar,
        x,
        pa_map,
        conditioned=True,
        base_index=torch.tensor([31, 47]),
    )

    assert sorted(records.fold_id.unique().tolist()) == [0, 1, 2, 3]
    assert records.theta.shape[0] == 8
    assert records.base_index.tolist() == [31, 47] * 4
    assert not bool((records.support_raw_mask & records.holdout_raw_mask).any())
    assert records.fold_count == 4


def _relation_metadata(rows):
    columns = tuple(zip(*rows))
    return {
        "tx": torch.tensor(columns[0]),
        "receiver": torch.tensor(columns[1]),
        "day": torch.tensor(columns[2]),
        "eq": torch.tensor(columns[3]),
        "sig_i": torch.tensor(columns[4]),
        "base_index": torch.tensor(columns[5]),
    }


def _audit_relation_metadata():
    return _relation_metadata(
        [
            (0, 0, 0, 1, 50, 1000),
            (1, 1, 0, 1, 60, 1001),
            (2, 0, 1, 1, 70, 1002),
            (3, 2, 2, 1, 80, 1003),
        ]
    )


def _bank_relation_metadata():
    return _relation_metadata(
        [
            (0, 0, 0, 1, 10, 2000),  # F2 for audit 0
            (0, 2, 0, 1, 20, 2001),  # F3 for audit 0
            (0, 0, 2, 1, 30, 2002),  # F4 for audit 0
            (4, 0, 0, 1, 40, 2003),  # F5 for audit 0
            (1, 1, 0, 1, 11, 2004),
            (1, 0, 0, 1, 21, 2005),
            (5, 1, 0, 1, 31, 2006),
            (2, 0, 1, 1, 12, 2007),
            (2, 1, 1, 1, 22, 2008),
            (5, 0, 1, 1, 32, 2009),
        ]
    )


@pytest.mark.parametrize("relation", ("F2", "F3", "F4", "F5"))
def test_relation_mapping_uses_strict_metadata_and_never_falls_back(relation):
    audit = _audit_relation_metadata()
    bank = _bank_relation_metadata()

    mapping = build_relation_indices(audit, bank, relation, seed=5)

    assert mapping.fallback_count == 0
    assert mapping.selection_uses_learned_q is False
    assert mapping.index[-1] == -1
    assert mapping.valid[-1] is False
    for audit_index, bank_index in enumerate(mapping.index):
        if bank_index < 0:
            continue
        atx, arx, aday = (int(audit[name][audit_index]) for name in ("tx", "receiver", "day"))
        btx, brx, bday = (int(bank[name][bank_index]) for name in ("tx", "receiver", "day"))
        if relation == "F2":
            assert (btx, brx, bday) == (atx, arx, aday)
            assert int(bank["base_index"][bank_index]) != int(audit["base_index"][audit_index])
        elif relation == "F3":
            assert btx == atx and brx != arx and bday == aday
        elif relation == "F4":
            assert btx == atx and brx == arx and bday != aday
        else:
            assert btx != atx and brx == arx and bday == aday


def test_f6_uses_fixed_physical_features_after_strict_f3_filtering():
    audit = _audit_relation_metadata()
    bank = _bank_relation_metadata()
    audit_features = torch.tensor([[0.0], [10.0], [20.0], [30.0]])
    bank_features = torch.tensor([[99.0], [0.1], [99.0], [99.0], [99.0], [9.9], [99.0], [99.0], [20.2], [99.0]])

    mapping = build_relation_indices(
        audit,
        bank,
        "F6",
        seed=91,
        audit_physical_features=audit_features,
        bank_physical_features=bank_features,
    )

    assert mapping.index[:3] == (1, 5, 8)
    assert mapping.index[3] == -1
    assert mapping.selection_policy == "strict_metadata_then_fixed_pa_distance"
    assert mapping.selection_uses_learned_q is False


def test_common_anchor_mask_is_intersection_of_f2_f3_and_f5():
    audit = _audit_relation_metadata()
    bank = _bank_relation_metadata()
    mappings = {
        row: build_relation_indices(audit, bank, row, seed=3)
        for row in ("F2", "F3", "F5")
    }

    mask = common_anchor_mask(mappings)

    assert mask.tolist() == [True, True, True, False]


def test_factor_rows_expand_sample_relations_over_all_folds_and_common_anchors():
    config = _config()
    sidecar = _sidecar(config).eval()
    generator = torch.Generator().manual_seed(41)
    audit_x = torch.randn(4, 2, config.input_length, generator=generator)
    audit_pa = torch.randn(4, config.pa_channels, 31, generator=generator)
    bank_x = torch.randn(10, 2, config.input_length, generator=generator)
    bank_pa = torch.randn(10, config.pa_channels, 31, generator=generator)
    audit_records = build_fold_records(
        sidecar,
        audit_x,
        audit_pa,
        conditioned=True,
        base_index=torch.tensor([1000, 1001, 1002, 1003]),
    )
    bank_records = build_fold_records(
        sidecar,
        bank_x,
        bank_pa,
        conditioned=True,
        base_index=torch.arange(2000, 2010),
    )
    audit_meta = _audit_relation_metadata()
    bank_meta = _bank_relation_metadata()
    mappings = {
        row: build_relation_indices(audit_meta, bank_meta, row, seed=7)
        for row in ("F2", "F3", "F4", "F5")
    }
    mappings["F6"] = build_relation_indices(
        audit_meta,
        bank_meta,
        "F6",
        seed=7,
        audit_physical_features=torch.arange(4).float().unsqueeze(1),
        bank_physical_features=torch.arange(10).float().unsqueeze(1),
    )
    mappings["F7"] = build_relation_indices(audit_meta, bank_meta, "F7", seed=7)

    rows = compose_factor_rows(audit_records, bank_records, mappings)

    assert set(rows) == {f"F{i}" for i in range(10)}
    assert rows["F3"].valid.shape == (16,)
    assert rows["F3"].valid.tolist() == list(mappings["F3"].valid) * 4
    assert rows["F7"].valid.sum().item() == 0
    assert rows["F9"].valid.all().item()
    assert rows["F3"].common_anchor.tolist() == [True, True, True, False] * 4
    first_bank_index = mappings["F3"].index[0]
    torch.testing.assert_close(rows["F3"].inputs[0, : config.operator_dim], bank_records.theta[first_bank_index])
    torch.testing.assert_close(rows["F8"].inputs[0, : config.operator_dim], audit_records.theta[0])


def test_fold_macro_nmse_is_mean_of_four_fold_nmses_not_global_energy_ratio():
    squared_error = torch.tensor([1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0])
    target_energy = torch.tensor([2.0, 4.0, 4.0, 8.0, 6.0, 12.0, 8.0, 16.0])
    fold_id = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    valid = torch.ones(8, dtype=torch.bool)

    result = fold_macro_nmse(squared_error, target_energy, fold_id, valid)

    expected = ((4 / 6) + (6 / 12) + (8 / 18) + (10 / 24)) / 4
    assert result["macro_nmse"] == pytest.approx(expected)
    assert set(result["per_fold_nmse"]) == {"0", "1", "2", "3"}


def _stage_metrics(*, transfer=True, conditioning=True):
    c4 = {
        "f3_vs_f0_relative_gain": 0.08 if transfer else 0.02,
        "f3_vs_f0_ci_low": 0.01 if transfer else -0.01,
        "f3_vs_f5_relative_gain": 0.09,
        "f3_vs_f5_ci_low": 0.02,
        "f3_vs_f2_relative_degradation": 0.06,
        "f3_common_nmse": 0.80,
    }
    c1 = {"f3_common_nmse": 0.84 if conditioning else 0.805}
    comparison = {
        "c4_vs_c1_f3_relative_gain": 0.0476 if conditioning else 0.0062,
        "c4_vs_c1_f3_ci_low": 0.01 if conditioning else -0.01,
    }
    return c1, c4, comparison


def _coverage():
    return {
        "f3": 0.90,
        "each_tx_two_cross_receiver_relations": True,
        "major_cell_minimum_pass": True,
    }


def _sensitivity():
    return {
        "head_seed_direction_count": 2,
        "candidate_seed_direction_count": 3,
        "satellite_seed_conclusion_reversal": False,
    }


def test_stage_a_pass_requires_transfer_conditioning_coverage_and_seed_stability():
    c1, c4, comparison = _stage_metrics()

    verdict = evaluate_stage_a(c1, c4, comparison, _coverage(), _sensitivity())

    assert verdict.status == "A_PASS"
    assert verdict.next_route == "RUN_M21B_TRUTH_BLIND_EXPERT_GATE"


def test_stage_a_partial_keeps_pa_operator_but_stops_current_conditioning():
    c1, c4, comparison = _stage_metrics(conditioning=False)

    verdict = evaluate_stage_a(c1, c4, comparison, _coverage(), _sensitivity())

    assert verdict.status == "A_PARTIAL"
    assert verdict.next_route == "KEEP_PA_OPERATOR_STOP_CURRENT_CHALLENGE_CONDITIONING"


def test_stage_a_fail_stops_current_theta_transfer_when_f3_does_not_beat_f0():
    c1, c4, comparison = _stage_metrics(transfer=False)

    verdict = evaluate_stage_a(c1, c4, comparison, _coverage(), _sensitivity())

    assert verdict.status == "A_FAIL"
    assert verdict.next_route == "STOP_CURRENT_PA_THETA_TRANSFER"
    assert "F3_NOT_BETTER_THAN_F0" in verdict.reasons


def _synthetic_factor_rows(count, *, seed):
    generator = torch.Generator().manual_seed(seed)
    target = torch.randn(count, 2, generator=generator)
    signal = torch.cat((target, torch.zeros(count, 2)), dim=1)
    noise = torch.randn(count, 4, generator=generator)
    fold_id = torch.arange(count).remainder(4)
    base_index = torch.arange(10000, 10000 + count)
    valid = torch.ones(count, dtype=torch.bool)
    common = torch.ones(count, dtype=torch.bool)

    def make(inputs, row_valid=valid):
        return FactorRow(
            inputs=inputs,
            target=target,
            valid=row_valid,
            common_anchor=common & row_valid,
            base_index=base_index,
            fold_id=fold_id,
        )

    rows = {
        "F0": make(torch.zeros_like(signal)),
        "F1": make(signal),
        "F2": make(signal + 0.02 * noise),
        "F3": make(signal + 0.03 * noise),
        "F4": make(signal + 0.05 * noise),
        "F5": make(noise),
        "F6": make(signal + 0.04 * noise),
        "F7": make(torch.zeros_like(signal), torch.zeros(count, dtype=torch.bool)),
        "F8": make(noise.roll(1, dims=0)),
        "F9": make(torch.zeros_like(signal)),
    }
    return rows


def test_factor_matrix_trains_equal_capacity_heads_and_detects_f3_signal():
    train_rows = _synthetic_factor_rows(96, seed=51)
    eval_rows = _synthetic_factor_rows(48, seed=52)
    groups = torch.stack(
        (
            torch.arange(48).remainder(6),
            torch.arange(48).remainder(4),
            torch.arange(48).remainder(2),
        ),
        dim=1,
    )

    result = run_factor_matrix(
        train_rows,
        eval_rows,
        eval_groups=groups,
        head_seeds=(3, 5, 7),
        steps=180,
        batch_size=32,
        hidden_dim=16,
        bootstrap_resamples=80,
        device=torch.device("cpu"),
    )

    assert result.payload["head_seeds"] == [3, 5, 7]
    assert result.payload["rows"]["F7"]["status"] == "UNAVAILABLE"
    assert result.payload["summary"]["f3_vs_f0_relative_gain"] > 0.50
    assert result.payload["summary"]["f3_vs_f5_relative_gain"] > 0.50
    assert result.payload["summary"]["head_seed_direction_count"] == 3
    assert set(result.squared_errors) == {3, 5, 7}
    assert "per_sample_errors" not in result.payload
