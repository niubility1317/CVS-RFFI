from __future__ import annotations

from dataclasses import fields, replace
import math

import pytest
import torch

from cvsrffi import stage2_d127_phase1_assets as assets
from cvsrffi import stage2_d127_da_candidates as core


def _receiver_means(dimension: int) -> torch.Tensor:
    values = torch.zeros(3, 2, dimension, dtype=torch.float32)
    values[0, 0, :2] = torch.tensor([2.0, 0.5])
    values[0, 1, :2] = torch.tensor([1.3, -0.4])
    values[1, 0, :2] = torch.tensor([-0.8, 1.4])
    values[1, 1, :2] = torch.tensor([-1.1, 0.6])
    values[2, 0, :2] = torch.tensor([0.2, -1.7])
    values[2, 1, :2] = torch.tensor([0.7, -1.1])
    return values


def _fsrg_episode(receiver: int, k_shot: int, dimension: int = 4) -> assets.FSRGEpisode:
    generator = torch.Generator().manual_seed(1000 + receiver * 31 + k_shot)
    classes = torch.tensor([3, 9], dtype=torch.int64)
    labels = classes.repeat_interleave(k_shot)
    support = torch.randn(2 * k_shot, dimension, generator=generator, dtype=torch.float32)
    support[:, 0] += torch.where(labels == 3, 0.8, -0.6)
    support[:, 1] += torch.where(labels == 3, -0.3, 0.7)
    query_labels = torch.tensor([3, 3, 9, 9], dtype=torch.int64)
    query = torch.randn(4, dimension, generator=generator, dtype=torch.float32)
    query[:, 0] += torch.where(query_labels == 3, 0.75, -0.55)
    query[:, 1] += torch.where(query_labels == 3, -0.25, 0.65)
    prefix = f"r{receiver}-k{k_shot}"
    return assets.FSRGEpisode(
        episode_id=prefix,
        receiver_id=f"rx-{receiver}",
        k_shot=k_shot,
        support_taps=support,
        support_labels=labels,
        query_taps=query,
        query_labels=query_labels,
        support_physical_ids=tuple(f"{prefix}-s{i}" for i in range(2 * k_shot)),
        query_physical_ids=tuple(f"{prefix}-q{i}" for i in range(4)),
    )


def _fsrg_episodes() -> assets.FrozenFSRGEpisodes:
    return assets.FrozenFSRGEpisodes(
        tuple(_fsrg_episode(receiver, k) for receiver in range(3) for k in (1, 5))
    )


def _fsrg_callbacks() -> assets.FSRGLossCallbacks:
    def support_per_sample(
        episode: assets.FSRGEpisode, rows: torch.Tensor
    ) -> torch.Tensor:
        labels = episode.support_labels
        value = rows.mean(dim=2) if rows.ndim == 3 else rows
        target = torch.where(labels == 3, 0.4, -0.5).to(dtype=rows.dtype)
        return (value[:, 0] + 0.35 * value[:, 1] - target).square()

    def outer_per_sample(
        episode: assets.FSRGEpisode,
        support: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        support_labels = episode.support_labels
        query_labels = episode.query_labels
        support_value = support.mean(dim=2) if support.ndim == 3 else support
        query_value = query.mean(dim=2) if query.ndim == 3 else query
        class_means = torch.stack(
            [support_value[support_labels == label].mean(dim=0) for label in (3, 9)]
        )
        expected = torch.where(query_labels == 3, class_means[0, 0], class_means[1, 0])
        return (query_value[:, 0] + 0.2 * query_value[:, 1] - expected).square()

    return assets.FSRGLossCallbacks(
        support_per_sample=support_per_sample,
        outer_query_per_sample=outer_per_sample,
    )


def _rdha_episode(receiver: int, k_shot: int) -> assets.RDHAEpisode:
    generator = torch.Generator().manual_seed(4000 + receiver * 31 + k_shot)
    labels = torch.tensor([4, 13], dtype=torch.int64).repeat_interleave(k_shot)
    support = 0.35 * torch.randn(
        2 * k_shot, core.JOINT_PROJ_INPUT_DIM, generator=generator, dtype=torch.float32
    )
    class_scale = 1.0 + 0.08 * receiver + 0.01 * k_shot
    support[:, 0] += class_scale * torch.where(labels == 4, 0.50, -0.35)
    support[:, 1] += (2.0 - class_scale) * torch.where(labels == 4, -0.25, 0.40)
    support[:, 0] += 0.04 * receiver + 0.005 * k_shot
    support[:, 1] += -0.03 * receiver + 0.004 * k_shot
    query_labels = torch.tensor([4, 4, 13, 13], dtype=torch.int64)
    query = 0.35 * torch.randn(
        4, core.JOINT_PROJ_INPUT_DIM, generator=generator, dtype=torch.float32
    )
    query[:, 0] += class_scale * torch.where(query_labels == 4, 0.45, -0.32)
    query[:, 1] += (2.0 - class_scale) * torch.where(query_labels == 4, -0.22, 0.37)
    query[:, 0] += 0.04 * receiver + 0.005 * k_shot
    query[:, 1] += -0.03 * receiver + 0.004 * k_shot
    prefix = f"c-r{receiver}-k{k_shot}"
    return assets.RDHAEpisode(
        episode_id=prefix,
        receiver_id=f"rx-{receiver}",
        k_shot=k_shot,
        support_hidden=support,
        support_labels=labels,
        query_hidden=query,
        query_labels=query_labels,
        support_physical_ids=tuple(f"{prefix}-s{i}" for i in range(2 * k_shot)),
        query_physical_ids=tuple(f"{prefix}-q{i}" for i in range(4)),
    )


def _rdha_episodes() -> assets.FrozenRDHAEpisodes:
    return assets.FrozenRDHAEpisodes(
        tuple(_rdha_episode(receiver, k) for receiver in range(3) for k in (1, 5))
    )


def _rdha_outer_loss(
    episode: assets.RDHAEpisode,
    support: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    support_labels = episode.support_labels
    query_labels = episode.query_labels
    means = torch.stack([support[support_labels == label, 0].mean() for label in (4, 13)])
    expected = torch.where(query_labels == 4, means[0], means[1])
    return (query[:, 0] + 0.15 * query[:, 1] - expected).square()


def test_canonical_receiver_mean_svd_is_deterministic_sign_fixed_and_rank_two() -> None:
    first = assets.canonical_receiver_mean_svd(_receiver_means(4), dimension=4)
    second = assets.canonical_receiver_mean_svd(_receiver_means(4), dimension=4)
    assert torch.equal(first.U, second.U)
    assert torch.equal(first.V, second.V)
    assert first.U.dtype == torch.float32 and first.V.dtype == torch.float32
    for row in first.V:
        assert row[torch.argmax(torch.abs(row))].item() > 0.0
    assert first.singular_values[0] >= first.singular_values[1] > 0.0

    translated = _receiver_means(4)[torch.tensor([2, 0, 1])][:, torch.tensor([1, 0])]
    translated = translated + torch.tensor([7.0, -4.0, 1.5, 0.25])
    rerooted = assets.canonical_receiver_mean_svd(translated, dimension=4)
    assert torch.allclose(first.U, rerooted.U, atol=2.0e-6, rtol=0.0)
    assert torch.allclose(first.V, rerooted.V, atol=2.0e-6, rtol=0.0)

    rank_one = _receiver_means(4)
    rank_one[..., 1] = 2.0 * rank_one[..., 0]
    rank_one[..., 2:] = 0.0
    with pytest.raises(assets.D127Phase1AssetError, match="rank below two"):
        assets.canonical_receiver_mean_svd(rank_one, dimension=4)


@pytest.mark.parametrize("near_scale", [1.0, 1.0 + 1.0e-9])
def test_canonical_receiver_mean_svd_fails_closed_on_repeated_or_near_root(
    near_scale: float,
) -> None:
    root = torch.tensor(
        [
            [1.0, 0.0],
            [-0.5, math.sqrt(3.0) / 2.0 * near_scale],
            [-0.5, -math.sqrt(3.0) / 2.0 * near_scale],
        ],
        dtype=torch.float64,
    )
    means = torch.zeros(3, 2, 4, dtype=torch.float64)
    means[:, :, :2] = root[:, None, :]
    means[:, 1, 2] = 0.7
    with pytest.raises(assets.D127Phase1AssetError, match="non-unique top-two"):
        assets.canonical_receiver_mean_svd(means, dimension=4)


def test_frozen_phase1_episodes_enforce_support_query_physical_disjointness() -> None:
    episode = _fsrg_episode(0, 1)
    with pytest.raises(assets.D127Phase1AssetError, match="physically disjoint"):
        assets.FSRGEpisode(
            episode_id=episode.episode_id,
            receiver_id=episode.receiver_id,
            k_shot=episode.k_shot,
            support_taps=episode.support_taps,
            support_labels=episode.support_labels,
            query_taps=episode.query_taps,
            query_labels=episode.query_labels,
            support_physical_ids=episode.support_physical_ids,
            query_physical_ids=(episode.support_physical_ids[0], "new-q1", "new-q2", "new-q3"),
        )

    episodes = list(_fsrg_episodes().episodes)
    episodes[1] = replace(
        episodes[1],
        query_physical_ids=(
            episodes[0].support_physical_ids[0],
            *episodes[1].query_physical_ids[1:],
        ),
    )
    with pytest.raises(assets.D127Phase1AssetError, match="globally disjoint"):
        assets.FrozenFSRGEpisodes(tuple(episodes))


def test_fsrg_phase1_training_is_deterministic_nonzero_and_budget_locked() -> None:
    episodes = _fsrg_episodes()
    initialization = assets.canonical_receiver_mean_svd(_receiver_means(4), dimension=4)
    callbacks = _fsrg_callbacks()
    first = assets.train_fsrg_phase1_asset(
        candidate_id=core.CANDIDATE_A,
        episodes=episodes,
        initialization=initialization,
        callbacks=callbacks,
    )
    second = assets.train_fsrg_phase1_asset(
        candidate_id=core.CANDIDATE_A,
        episodes=episodes,
        initialization=initialization,
        callbacks=callbacks,
    )
    assert first.receipt.max_iter == assets.LBFGS_MAX_ITER == 128
    assert first.receipt.line_search_fn == "strong_wolfe"
    assert first.receipt.initialization_count == 1
    assert first.receipt.closure_calls >= 1
    assert first.receipt.internal_iterations >= 1
    assert first.receipt.initial_gradient_norm > 0.0
    assert all(value > 0.0 for value in first.receipt.initial_parameter_gradient_norms)
    assert torch.equal(first.asset.U, second.asset.U)
    assert torch.equal(first.asset.V, second.asset.V)
    assert torch.all(first.statistics.d_f_diag > 0.0)
    assert first.statistics.rho > 0.0

    quantized = assets.quantize_fsrg_asset(first.asset)
    assert quantized.numeric_payload_bytes == 4 * first.asset.dimension + 14
    assert quantized.persistent_fp32_sidecar is False
    assets.assert_no_persistent_fp32_sidecar(quantized)
    decoded = quantized.decode()
    assert decoded.U.dtype == torch.float32 and decoded.U.requires_grad is False
    assert decoded.V.dtype == torch.float32 and decoded.V.requires_grad is False
    assert decoded.d_f_diag.dtype == torch.float32


def test_rdah_phase1_training_qb_and_1328_byte_state() -> None:
    episodes = _rdha_episodes()
    initialization = assets.canonical_receiver_mean_svd(
        _receiver_means(core.JOINT_PROJ_INPUT_DIM), dimension=core.JOINT_PROJ_INPUT_DIM
    )
    result = assets.train_rdah_phase1_asset(
        episodes=episodes,
        initialization=initialization,
        outer_query_per_sample=_rdha_outer_loss,
    )
    assert result.receipt.max_iter == 128
    assert result.receipt.line_search_fn == "strong_wolfe"
    assert result.receipt.initial_gradient_norm > 0.0
    assert all(value > 0.0 for value in result.receipt.initial_parameter_gradient_norms)
    assert result.asset.Q.shape == (2, 5) and result.asset.b.shape == (2,)
    assert result.statistics.a_max > 0.0
    quantized = assets.quantize_rdah_asset(result.asset)
    assert quantized.numeric_payload_bytes == 1328
    assert quantized.persistent_fp32_sidecar is False
    assets.assert_no_persistent_fp32_sidecar(quantized)
    decoded = quantized.decode()
    assert decoded.U.dtype == torch.float32 and decoded.Q.dtype == torch.float32
    assert decoded.mean_p1.dtype == torch.float32 and decoded.std_p1.dtype == torch.float32
    fixture = episodes.episodes[0]

    def rdha_forward(asset: core.FSRGAsset | core.RDHAAsset) -> torch.Tensor:
        assert isinstance(asset, core.RDHAAsset)
        state = core.fit_rdah_support_state(
            fixture.support_hidden, fixture.support_labels, asset
        )
        return core.adapt_rdah_query(fixture.query_hidden, asset, state)[:, :2]

    parity = assets.phase1_fixture_parity_receipt(
        fixture_id="phase1-fixed-rdha-fixture",
        float_asset=result.asset,
        quantized_asset=quantized,
        forward=rdha_forward,
    )
    assert parity.argmax_equal
    assert math.isclose(
        quantized.numeric_payload_bytes,
        2 * 320 + 4 + 2 * 320 + 4 + 2 * 5 + 4 + 2 + 2 + 2 * 5 + 2 * 5 + 2,
    )


def test_quantized_typed_state_has_no_tensor_sidecar_and_receives_phase1_parity() -> None:
    U = torch.tensor(
        [[0.72, -0.11], [-0.34, 0.85], [0.19, 0.26], [-0.44, -0.18]], dtype=torch.float32
    )
    V = torch.tensor(
        [[0.31, -0.22, 0.54, 0.12], [-0.17, 0.41, -0.28, 0.63]], dtype=torch.float32
    )
    float_asset = core.FSRGAsset(
        candidate_id=core.CANDIDATE_B,
        tap_name=core.TAP_B,
        U=U,
        V=V,
        d_f_diag=torch.tensor([0.4, 1.2], dtype=torch.float32),
        rho=0.2,
    )
    quantized = assets.quantize_fsrg_asset(float_asset)
    assets.assert_no_persistent_fp32_sidecar(quantized)

    def contains_tensor(value: object) -> bool:
        if torch.is_tensor(value):
            return True
        if hasattr(value, "__dataclass_fields__"):
            return any(contains_tensor(getattr(value, item.name)) for item in fields(value))
        if isinstance(value, tuple):
            return any(contains_tensor(item) for item in value)
        return False

    assert contains_tensor(quantized) is False
    fixture = torch.tensor(
        [[1.2, -0.3, 0.4, -0.1], [-0.8, 1.1, -0.2, 0.7], [0.1, 0.2, -0.5, 0.6]],
        dtype=torch.float32,
    )

    def forward(asset: core.FSRGAsset | core.RDHAAsset) -> torch.Tensor:
        assert isinstance(asset, core.FSRGAsset)
        return fixture @ asset.U + 0.1 * (fixture @ asset.V.transpose(0, 1))

    receipt = assets.phase1_fixture_parity_receipt(
        fixture_id="phase1-fixed-fsrg-fixture",
        float_asset=float_asset,
        quantized_asset=quantized,
        forward=forward,
    )
    assert receipt.output_shape == (3, 2)
    assert receipt.element_count == 6
    assert receipt.max_abs_error >= 0.0
    assert receipt.argmax_equal
    assert receipt.argmax_agreement == 1.0


def test_positive_fp16_statistics_fail_on_underflow_and_preserve_subnormal() -> None:
    with pytest.raises(assets.D127Phase1AssetError, match="not representable"):
        assets.FP16Buffer.from_tensor(
            torch.tensor([1.0e-12], dtype=torch.float32),
            name="underflow-statistic",
            require_positive=True,
        )

    smallest_subnormal = float(
        torch.nextafter(
            torch.tensor(0.0, dtype=torch.float16),
            torch.tensor(1.0, dtype=torch.float16),
        ).item()
    )
    assert 0.0 < smallest_subnormal < float(torch.finfo(torch.float16).tiny)
    encoded = assets.FP16Buffer.from_tensor(
        torch.tensor([smallest_subnormal], dtype=torch.float32),
        name="subnormal-statistic",
        require_positive=True,
    )
    decoded = encoded.decode()
    assert decoded.dtype == torch.float32
    assert decoded.item() == smallest_subnormal


def test_quantized_from_tensor_and_decode_survive_disabled_numpy_torch_abi_bridge(monkeypatch) -> None:
    def _blocked(*_args, **_kwargs):
        raise TypeError("simulated NumPy2/Torch2.1 ABI mismatch")

    monkeypatch.setattr(torch, "from_numpy", _blocked)
    monkeypatch.setattr(torch, "as_tensor", _blocked)
    monkeypatch.setattr(torch.Tensor, "numpy", _blocked)
    vector = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
    matrix = torch.tensor([[0.25, -0.5], [0.75, 0.125]], dtype=torch.float32)
    fp16 = assets.FP16Buffer.from_tensor(vector, name="compat-fp16")
    decoded_fp16 = fp16.decode()
    decoded_matrix = assets.SymmetricInt8Matrix.from_tensor(
        matrix, group_axis="column", name="compat-matrix"
    ).decode()
    decoded_vector = assets.SymmetricInt8Vector.from_tensor(
        vector, name="compat-vector"
    ).decode()
    assert decoded_fp16.dtype == decoded_matrix.dtype == decoded_vector.dtype == torch.float32
    assert decoded_fp16.shape == vector.shape
    assert decoded_matrix.shape == matrix.shape
    assert decoded_vector.shape == vector.shape
    assert not decoded_fp16.requires_grad
    assert torch.allclose(decoded_fp16, vector, atol=5.0e-4, rtol=0.0)
    assert torch.allclose(decoded_matrix, matrix, atol=5.0e-3, rtol=0.0)
    assert torch.allclose(decoded_vector, vector, atol=5.0e-3, rtol=0.0)
