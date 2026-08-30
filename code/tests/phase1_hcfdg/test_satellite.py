from dataclasses import FrozenInstanceError

import pytest
import torch
from sat_channel import SatSimConfig, apply_sat_gnd_channel_batch

from cvsrffi.phase1_hcfdg.satellite import (
    ChannelFactors,
    SingleViewBatch,
    build_single_view_batch,
)


def _factor_metadata(batch_size: int) -> dict[str, torch.Tensor]:
    values = torch.arange(batch_size, dtype=torch.float32)
    return {
        "cfo": values + 1.0,
        "phase_noise": values + 11.0,
        "snr": values + 21.0,
        "multipath": values + 31.0,
        "elevation": values + 41.0,
    }


class _FakeAugmentor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, str, torch.Generator]] = []

    def __call__(
        self,
        batch: torch.Tensor,
        *,
        scenario: str,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.calls.append((batch.detach().clone(), scenario, generator))
        return batch + 1000.0, _factor_metadata(batch.shape[0])

    def concat(self, *_args, **_kwargs):
        raise AssertionError("single-view construction must not use a concat helper")


def _expected_mask(batch_size: int, seed: int, p_sat: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = int(batch_size * p_sat + 0.5)
    mask = torch.zeros(batch_size, dtype=torch.bool)
    mask[torch.randperm(batch_size, generator=generator)[:count]] = True
    return mask


def test_single_view_batch_keeps_one_position_per_sample_and_emits_factor_labels():
    x = torch.arange(8 * 2 * 4, dtype=torch.float32).reshape(8, 2, 4)
    seed = 123
    augmentor = _FakeAugmentor()

    result = build_single_view_batch(
        x,
        augmentor,
        torch.Generator(device="cpu").manual_seed(seed),
        p_sat=0.30,
    )

    expected_mask = _expected_mask(len(x), seed, 0.30)
    expected_factors = torch.zeros((len(x), 5), dtype=x.dtype)
    expected_factors[expected_mask] = torch.stack(
        tuple(_factor_metadata(int(expected_mask.sum()))[name] for name in (
            "cfo",
            "phase_noise",
            "snr",
            "multipath",
            "elevation",
        )),
        dim=1,
    )

    assert isinstance(result, SingleViewBatch)
    assert result.iq.shape == x.shape
    assert result.satellite_mask.shape == (x.shape[0],)
    assert result.channel_factors.shape == (x.shape[0], 5)
    assert torch.equal(result.satellite_mask, expected_mask)
    assert torch.equal(result.channel_labels, expected_mask.to(dtype=torch.long))
    assert torch.equal(result.channel_factors, expected_factors)
    assert torch.equal(result.iq[~expected_mask], x[~expected_mask])
    assert torch.equal(result.iq[expected_mask], x[expected_mask] + 1000.0)
    assert len(augmentor.calls) == 1
    assert augmentor.calls[0][0].shape[0] == int(expected_mask.sum())
    assert augmentor.calls[0][1] == "mixed_orbit"


def test_single_view_batch_mask_and_outputs_are_reproducible_for_same_generator_seed():
    x = torch.randn(12, 2, 8)

    first = build_single_view_batch(
        x,
        _FakeAugmentor(),
        torch.Generator(device="cpu").manual_seed(392001),
    )
    second = build_single_view_batch(
        x,
        _FakeAugmentor(),
        torch.Generator(device="cpu").manual_seed(392001),
    )

    assert torch.equal(first.satellite_mask, second.satellite_mask)
    assert torch.equal(first.iq, second.iq)
    assert torch.equal(first.channel_labels, second.channel_labels)
    assert torch.equal(first.channel_factors, second.channel_factors)


def test_single_view_batch_uses_exact_nearest_integer_satellite_count():
    x = torch.zeros(96, 2, 4)
    result = build_single_view_batch(
        x,
        _FakeAugmentor(),
        torch.Generator(device="cpu").manual_seed(392002),
        p_sat=0.30,
    )

    assert result.satellite_mask.sum().item() == 29


def test_empirical_satellite_fraction_tracks_point_three():
    x = torch.zeros(20, 2, 4)
    augmentor = _FakeAugmentor()
    generator = torch.Generator(device="cpu").manual_seed(7)

    count = sum(
        build_single_view_batch(x, augmentor, generator, 0.30)
        .satellite_mask.sum()
        .item()
        for _ in range(200)
    )

    assert abs(count / (200 * len(x)) - 0.30) < 0.02


def test_selected_satellite_rows_reject_missing_all_channel_factors():
    def tensor_only_augmentor(batch, *, scenario, generator):
        return batch + 1.0

    with pytest.raises(ValueError, match="channel factors.*required"):
        build_single_view_batch(
            torch.zeros(4, 2, 8),
            tensor_only_augmentor,
            torch.Generator(device="cpu").manual_seed(11),
            p_sat=1.0,
        )


@pytest.mark.parametrize(
    "missing_name",
    ["cfo", "phase_noise", "snr", "multipath", "elevation"],
)
def test_selected_satellite_rows_reject_any_one_missing_channel_factor(missing_name: str):
    def incomplete_augmentor(batch, *, scenario, generator):
        factors = _factor_metadata(batch.shape[0])
        del factors[missing_name]
        return batch + 1.0, factors

    with pytest.raises(ValueError, match=missing_name):
        build_single_view_batch(
            torch.zeros(4, 2, 8),
            incomplete_augmentor,
            torch.Generator(device="cpu").manual_seed(13),
            p_sat=1.0,
        )


def test_zero_satellite_probability_keeps_clean_rows_and_skips_augmentor():
    x = torch.ones(4, 2, 8)
    augmentor = _FakeAugmentor()

    result = build_single_view_batch(
        x,
        augmentor,
        torch.Generator(device="cpu").manual_seed(9),
        p_sat=0.0,
    )

    assert torch.equal(result.iq, x)
    assert not result.satellite_mask.any()
    assert torch.equal(result.channel_labels, torch.zeros(4, dtype=torch.long))
    assert torch.equal(result.channel_factors, torch.zeros(4, 5))
    assert augmentor.calls == []


@pytest.mark.parametrize("p_sat", [-0.01, 1.01, float("nan"), float("inf")])
def test_satellite_probability_must_be_inclusive_unit_interval(p_sat: float):
    with pytest.raises(ValueError, match="p_sat"):
        build_single_view_batch(
            torch.zeros(2, 2, 4),
            _FakeAugmentor(),
            torch.Generator(device="cpu").manual_seed(3),
            p_sat=p_sat,
        )


def test_channel_factor_and_single_view_schemas_are_frozen():
    factors = ChannelFactors(
        cfo=torch.zeros(1),
        phase_noise=torch.zeros(1),
        snr=torch.zeros(1),
        multipath=torch.zeros(1),
        elevation=torch.zeros(1),
    )
    batch = SingleViewBatch(
        iq=torch.zeros(1, 2, 4),
        satellite_mask=torch.zeros(1, dtype=torch.bool),
        channel_labels=torch.zeros(1, dtype=torch.long),
        channel_factors=torch.zeros(1, 5),
    )

    with pytest.raises(FrozenInstanceError):
        factors.cfo = torch.ones(1)
    with pytest.raises(FrozenInstanceError):
        batch.iq = torch.ones(1, 2, 4)


def test_real_satellite_metadata_exposes_all_hcfdg_physical_factors():
    x = torch.randn(4, 2, 64)
    cfg = SatSimConfig(
        enable_multipath=True,
        num_taps=(2, 3),
        phase_noise_inc_std=(1e-4, 2e-4),
    )

    _, metadata, _ = apply_sat_gnd_channel_batch(
        x,
        cfg,
        gen=torch.Generator().manual_seed(91),
        return_meta=True,
    )

    assert metadata is not None
    assert metadata["phase_noise_std"].shape == (4,)
    assert metadata["num_taps"].shape == (4,)
