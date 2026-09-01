from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from cvsrffi.marc_ot_support_features import (
    MARC_OT_SUPPORT_FEATURE_CONFIG,
    MARC_OT_SUPPORT_LAYOUT,
    MARC_OT_SUPPORT_ROW_DIM,
    MARC_OT_SUPPORT_ROW_SCHEMA,
    build_marc_ot_support_features,
)


class _ReviewedAuxModel(nn.Module):
    """Small row-wise model exposing the reviewed ADV3B02 aux geometry."""

    def __init__(self, *, bad_key: str | None = None) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bad_key = bad_key

    def forward(self, values: torch.Tensor, return_aux: bool = True):
        assert return_aux is True
        flattened = values.float().flatten(start_dim=1)
        row_mean = flattened.mean(dim=1, keepdim=True) * self.scale
        row_rms = flattened.square().mean(dim=1, keepdim=True).sqrt() * self.scale
        weights = torch.linspace(0.25, 1.25, 160, device=values.device).view(1, -1)
        z_id = (row_mean + 1.0) * weights
        t_emb = (row_rms + 1.5) * weights
        f_emb = (row_mean - row_rms + 2.0) * weights.flip(1)
        output = {
            "z_id": z_id,
            "aux_id": {"t_emb": t_emb, "f_emb": f_emb},
            # Deliberately invalid: the production builder must not consume it.
            "z_dom": torch.full_like(z_id, float("nan")),
            "aux_dom": {"t_emb": torch.full_like(t_emb, float("nan"))},
        }
        if self.bad_key == "z_id_geometry":
            output["z_id"] = z_id[:, :-1]
        elif self.bad_key == "t_emb_nonfinite":
            output["aux_id"]["t_emb"] = t_emb.clone()
            output["aux_id"]["t_emb"][0, 0] = float("inf")
        elif self.bad_key == "missing_aux_id":
            output.pop("aux_id")
        return output


def _tone(*, frequency: float, rows: int = 1, length: int = 256) -> torch.Tensor:
    sample = torch.arange(length, dtype=torch.float32)
    phase = 2.0 * math.pi * frequency * sample
    row = torch.stack((torch.cos(phase), torch.sin(phase)))
    return row.unsqueeze(0).repeat(rows, 1, 1)


def _batch(iq: torch.Tensor, *, nominal_k: int, effective_mask=None):
    rows = int(iq.shape[0])
    class_count = rows // nominal_k
    labels = torch.arange(class_count, dtype=torch.long).repeat_interleave(nominal_k)
    tokens = tuple(f"opaque-{index}" for index in range(rows))
    return build_marc_ot_support_features(
        _ReviewedAuxModel(),
        iq,
        labels,
        tokens,
        nominal_k=nominal_k,
        effective_mask=effective_mask,
        validated_unpadded=effective_mask is None,
        scope="phase2_support",
        fit_scope="full_support",
    )


def test_builder_has_exact_685d_schema_order_and_ignores_domain_aux() -> None:
    iq = _tone(frequency=3.0 / 256.0)
    model = _ReviewedAuxModel()
    built = build_marc_ot_support_features(
        model,
        iq,
        torch.tensor([7]),
        ("opaque-token",),
        nominal_k=1,
        validated_unpadded=True,
        scope="phase2_support",
        fit_scope="full_support",
    )
    with torch.no_grad():
        aux = model(iq, return_aux=True)
    z_id = aux["z_id"]
    t_emb = aux["aux_id"]["t_emb"]
    f_emb = aux["aux_id"]["f_emb"]

    assert built.feature_schema == MARC_OT_SUPPORT_ROW_SCHEMA == "marc_ot.support.row.v1"
    assert built.feature_dim == MARC_OT_SUPPORT_ROW_DIM == 685
    assert built.feature_config == MARC_OT_SUPPORT_FEATURE_CONFIG
    assert built.rows.shape == (1, 685)
    assert tuple(MARC_OT_SUPPORT_LAYOUT) == (
        "z_id",
        "t_emb",
        "f_emb",
        "normalized_time_minus_frequency",
        "embedding_norms",
        "time_frequency_relation",
        "view_stability",
        "phase_clock_proxies",
        "normalized_log_psd_16",
        "rf_lite_10",
        "quality",
        "k_and_mask",
    )
    assert torch.equal(built.rows[:, MARC_OT_SUPPORT_LAYOUT["z_id"]], z_id)
    assert torch.equal(built.rows[:, MARC_OT_SUPPORT_LAYOUT["t_emb"]], t_emb)
    assert torch.equal(built.rows[:, MARC_OT_SUPPORT_LAYOUT["f_emb"]], f_emb)
    expected_delta = torch.nn.functional.normalize(t_emb, dim=1) - torch.nn.functional.normalize(
        f_emb, dim=1
    )
    assert torch.allclose(
        built.rows[:, MARC_OT_SUPPORT_LAYOUT["normalized_time_minus_frequency"]],
        expected_delta,
    )
    expected_norms = torch.stack(
        (z_id.norm(dim=1), t_emb.norm(dim=1), f_emb.norm(dim=1)), dim=1
    )
    assert torch.allclose(
        built.rows[:, MARC_OT_SUPPORT_LAYOUT["embedding_norms"]], expected_norms
    )
    assert built.effective_mask.tolist() == [1.0]
    assert built.rows[0, MARC_OT_SUPPORT_LAYOUT["k_and_mask"]].tolist() == [1.0, 1.0, 1.0]


def test_builder_is_bitwise_deterministic_and_row_permutation_equivariant() -> None:
    iq = torch.cat(
        (
            _tone(frequency=2.0 / 256.0, rows=2),
            _tone(frequency=5.0 / 256.0, rows=2),
        )
    )
    labels = torch.tensor([0, 0, 1, 1])
    tokens = ("a0", "a1", "b0", "b1")
    model = _ReviewedAuxModel()
    first = build_marc_ot_support_features(
        model,
        iq,
        labels,
        tokens,
        nominal_k=2,
        validated_unpadded=True,
        scope="phase2_support",
        fit_scope="full_support",
    )
    second = build_marc_ot_support_features(
        model,
        iq,
        labels,
        tokens,
        nominal_k=2,
        validated_unpadded=True,
        scope="phase2_support",
        fit_scope="full_support",
    )
    order = torch.tensor([2, 0, 3, 1])
    permuted = build_marc_ot_support_features(
        model,
        iq[order],
        labels[order],
        tuple(tokens[index] for index in order.tolist()),
        nominal_k=2,
        validated_unpadded=True,
        scope="phase2_support",
        fit_scope="full_support",
    )

    assert torch.equal(first.rows, second.rows)
    assert torch.equal(permuted.rows, first.rows[order])
    assert permuted.physical_tokens == tuple(tokens[index] for index in order.tolist())


@pytest.mark.parametrize("nominal_k", (1, 2, 5, 10, 20))
def test_builder_supports_frozen_k_registry_without_adding_view_rows(nominal_k: int) -> None:
    built = _batch(
        _tone(frequency=4.0 / 256.0, rows=2 * nominal_k), nominal_k=nominal_k
    )

    assert built.rows.shape == (2 * nominal_k, MARC_OT_SUPPORT_ROW_DIM)
    assert len(built.physical_tokens) == 2 * nominal_k
    k_slice = MARC_OT_SUPPORT_LAYOUT["k_and_mask"]
    assert torch.equal(
        built.rows[:, k_slice],
        torch.tensor([[float(nominal_k), float(nominal_k), 1.0]]).repeat(
            2 * nominal_k, 1
        ),
    )
    assert built.audit["deterministic_view"]["adds_physical_rows"] is False
    assert built.audit["deterministic_view"]["adds_effective_k"] is False


def test_builder_tracks_effective_k_and_requires_explicit_mask_for_padding() -> None:
    iq = _tone(frequency=4.0 / 256.0, rows=4)
    labels = torch.tensor([0, 0, 1, 1])
    tokens = ("a0", "a1", "b0", "b1")
    with pytest.raises(ValueError, match="effective_mask"):
        build_marc_ot_support_features(
            _ReviewedAuxModel(),
            iq,
            labels,
            tokens,
            nominal_k=2,
            scope="phase2_support",
            fit_scope="full_support",
        )
    built = build_marc_ot_support_features(
        _ReviewedAuxModel(),
        iq,
        labels,
        tokens,
        nominal_k=2,
        effective_mask=torch.tensor([1, 0, 0, 1]),
        scope="phase2_support",
        fit_scope="full_support",
    )

    assert built.effective_mask.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert built.rows[:, MARC_OT_SUPPORT_LAYOUT["k_and_mask"]].tolist() == [
        [2.0, 1.0, 1.0],
        [2.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 1.0, 1.0],
    ]


def test_builder_records_formal_nominal_k_for_fit_only_crossfit_subset() -> None:
    iq = _tone(frequency=4.0 / 256.0, rows=16)
    labels = torch.tensor([0] * 8 + [1] * 8)
    tokens = tuple(f"fit-{index}" for index in range(16))

    built = build_marc_ot_support_features(
        _ReviewedAuxModel(),
        iq,
        labels,
        tokens,
        nominal_k=10,
        effective_mask=torch.ones(16),
        scope="phase2_support",
        fit_scope="crossfit",
    )

    assert built.rows[:, MARC_OT_SUPPORT_LAYOUT["k_and_mask"]].tolist() == [
        [10.0, 8.0, 1.0]
    ] * 16


def test_builder_rejects_k8_as_damaged_k10_full_support() -> None:
    iq = _tone(frequency=4.0 / 256.0, rows=16)
    labels = torch.tensor([0] * 8 + [1] * 8)
    tokens = tuple(f"full-{index}" for index in range(16))

    with pytest.raises(ValueError, match="full.*K mismatch"):
        build_marc_ot_support_features(
            _ReviewedAuxModel(),
            iq,
            labels,
            tokens,
            nominal_k=10,
            effective_mask=torch.ones(16),
            scope="phase2_support",
            fit_scope="full_support",
        )


def test_builder_restores_mixed_parent_and_child_module_modes_exactly() -> None:
    model = _ReviewedAuxModel()
    model.batch_norm = nn.BatchNorm1d(2)
    model.dropout = nn.Dropout()
    model.train()
    model.batch_norm.eval()
    before = tuple(module.training for module in model.modules())

    build_marc_ot_support_features(
        model,
        _tone(frequency=4.0 / 256.0),
        torch.tensor([0]),
        ("mixed-mode",),
        nominal_k=1,
        validated_unpadded=True,
        scope="phase2_support",
        fit_scope="full_support",
    )

    assert tuple(module.training for module in model.modules()) == before


def test_builder_rejects_zero_dc_removed_rms_and_uses_true_unit_rms_view() -> None:
    constant = torch.tensor([[[1.0] * 256, [0.25] * 256]])
    with pytest.raises(ValueError, match="DC-removed complex RMS"):
        build_marc_ot_support_features(
            _ReviewedAuxModel(),
            constant,
            torch.tensor([0]),
            ("constant",),
            nominal_k=1,
            validated_unpadded=True,
            scope="phase2_support",
            fit_scope="full_support",
        )

    class CapturingModel(_ReviewedAuxModel):
        def __init__(self) -> None:
            super().__init__()
            self.observed: list[torch.Tensor] = []

        def forward(self, values: torch.Tensor, return_aux: bool = True):
            self.observed.append(values.detach().clone())
            return super().forward(values, return_aux=return_aux)

    model = CapturingModel()
    build_marc_ot_support_features(
        model,
        _tone(frequency=4.0 / 256.0),
        torch.tensor([0]),
        ("normal",),
        nominal_k=1,
        validated_unpadded=True,
        scope="phase2_support",
        fit_scope="full_support",
    )
    view_rms = model.observed[1].square().sum(dim=1).mean(dim=1).sqrt()

    assert view_rms.tolist() == pytest.approx([1.0], abs=1.0e-6)


def test_phase1_builder_audit_names_source_scope_without_false_zero_source_claim() -> None:
    built = build_marc_ot_support_features(
        _ReviewedAuxModel(),
        _tone(frequency=4.0 / 256.0),
        torch.tensor([0]),
        ("source-row",),
        nominal_k=1,
        validated_unpadded=True,
        scope="phase1_source",
        fit_scope="full_episode",
    )

    assert built.audit["input_scope"] == "phase1_source"
    assert "source_iq_rows_used" not in built.audit


def test_known_tone_and_chirp_have_expected_phase_and_psd_proxies() -> None:
    tone = _tone(frequency=32.0 / 256.0)
    built = _batch(tone, nominal_k=1)
    phase = built.rows[0, MARC_OT_SUPPORT_LAYOUT["phase_clock_proxies"]]
    psd = built.rows[0, MARC_OT_SUPPORT_LAYOUT["normalized_log_psd_16"]]

    phase_values = phase.detach()
    assert float(phase_values[0]) == pytest.approx(2.0 * math.pi * 32.0 / 256.0, abs=2e-5)
    assert float(phase_values[1]) == pytest.approx(0.0, abs=2e-5)
    assert float(phase_values[2]) == pytest.approx(0.0, abs=2e-5)
    assert float(phase_values[3]) == pytest.approx(0.0, abs=2e-5)
    assert int(psd.argmax()) == 2
    assert float(psd.detach().mean()) == pytest.approx(0.0, abs=2e-6)

    sample = torch.arange(256, dtype=torch.float32)
    phase_curve = 0.02 * sample + 0.0008 * sample.square()
    chirp = torch.stack((torch.cos(phase_curve), torch.sin(phase_curve))).unsqueeze(0)
    chirp_built = _batch(chirp, nominal_k=1)
    chirp_proxy = chirp_built.rows[0, MARC_OT_SUPPORT_LAYOUT["phase_clock_proxies"]]
    assert float(chirp_proxy[2].detach()) > 0.0
    assert float(chirp_proxy[3].detach()) > 0.0

    assert built.audit["cfo"]["status"] == "PROXY_ONLY"
    assert built.audit["sfo"]["status"] == "PROXY_ONLY"
    assert built.audit["cfo"]["absolute_physical_units_available"] is False
    assert built.audit["sfo"]["physical_sfo_available"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("duplicate_token", "unique"),
        ("class_k", "class K"),
        ("nonfinite_iq", "finite"),
        ("z_id_geometry", "geometry"),
        ("t_emb_nonfinite", "finite"),
        ("missing_aux_id", "aux_id"),
    ),
)
def test_builder_fails_closed_on_token_k_input_or_aux_drift(mutation: str, match: str) -> None:
    iq = _tone(frequency=3.0 / 256.0, rows=4)
    labels = torch.tensor([0, 0, 1, 1])
    tokens = ("a0", "a1", "b0", "b1")
    model = _ReviewedAuxModel(bad_key=mutation if "_" in mutation else None)
    if mutation == "duplicate_token":
        tokens = ("a0", "a0", "b0", "b1")
    elif mutation == "class_k":
        labels = torch.tensor([0, 0, 0, 1])
    elif mutation == "nonfinite_iq":
        iq = iq.clone()
        iq[0, 0, 0] = float("nan")

    with pytest.raises((ValueError, RuntimeError), match=match):
        build_marc_ot_support_features(
            model,
            iq,
            labels,
            tokens,
            nominal_k=2,
            validated_unpadded=True,
            scope="phase2_support",
            fit_scope="full_support",
        )


def test_builder_keeps_gradient_through_reviewed_model_features() -> None:
    model = _ReviewedAuxModel()
    built = build_marc_ot_support_features(
        model,
        _tone(frequency=3.0 / 256.0),
        torch.tensor([0]),
        ("one",),
        nominal_k=1,
        validated_unpadded=True,
        scope="phase2_support",
        fit_scope="full_support",
    )

    built.rows[:, :640].sum().backward()

    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
    assert torch.count_nonzero(model.scale.grad)
