from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cvsrffi import stage2_d127_da_candidates as da
from cvsrffi import stage2_d127_checkpoint_hooks as hooks
from cvsrffi import stage2_d127_phase1_assets as phase1_assets
from cvsrffi import stage2_zid_student_t_qknn as qknn


CHECKPOINT_PATH = Path(
    "E:/type10-7/automation_reports/CV-SincNet/"
    "qknnv42_strict_dual125_20260714_183556/artifacts/"
    "best_joint_safe_ssdg.pth"
)


def _phase1_qknn_locks() -> dict[int, qknn.Phase1ZIDStudentTLock]:
    return {
        k_shot: qknn.Phase1ZIDStudentTLock(
            active_k=k_shot,
            student_nu=3.0,
            kernel_effective_dim=160,
            kernel_volume_gamma=1.0,
            shared_h0=0.2,
            scale_prior_strength=2.0,
            scale_min_ratio=0.5,
            scale_max_ratio=2.0,
            temperature=0.85,
            phase1_lodo_receipt_sha256="1" * 64,
            quantization_margin_audit_sha256="2" * 64,
        )
        for k_shot in (1, 5)
    }


class _TinyT2(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(value))


class _TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.joint_proj = nn.Sequential(
            nn.Linear(320, 160), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )


class _TinyIdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_fuse = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=1, bias=False),
            nn.GroupNorm(1, 4),
            nn.ReLU(inplace=True),
        )
        self.time_down = nn.Identity()
        self.t1 = nn.Identity()
        self.t2 = _TinyT2(4)
        self.t3 = nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.hidden = nn.Linear(4, 320)
        self.cls_head = _TinyHead()
        self.calls = 0

    def forward(
        self,
        rows: torch.Tensor,
        y: torch.Tensor | None = None,
        return_aux: bool = True,
        domain_labels: torch.Tensor | None = None,
    ):
        del y, domain_labels
        self.calls += 1
        value = self.time_fuse(rows)
        value = self.time_down(value)
        value = self.t1(value)
        value = self.t2(value)
        value = self.t3(value)
        hidden = self.hidden(self.pool(value).squeeze(-1))
        z_id = self.cls_head.joint_proj(hidden)
        logits = torch.stack((z_id[:, 0], z_id[:, 1]), dim=1)
        if not return_aux:
            return logits
        return {"feat_joint": z_id, "feat_imp": z_id, "logits": logits}


class _TinyDualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _TinyIdentityBackbone()


def _fsrg_asset(candidate_id: str, dimension: int, device: torch.device) -> da.FSRGAsset:
    coordinate = torch.linspace(-1.0, 1.0, dimension, device=device)
    U = torch.stack((0.4 * coordinate, 0.15 + 0.25 * coordinate.square()), dim=1)
    V = torch.stack((0.5 * coordinate, -0.35 * coordinate + 0.1), dim=0)
    tap = da.TAP_A if candidate_id == da.CANDIDATE_A else da.TAP_B
    return da.FSRGAsset(
        candidate_id=candidate_id,
        tap_name=tap,
        U=U.float(),
        V=V.float(),
        d_f_diag=torch.tensor([0.7, 1.3], device=device, dtype=torch.float32),
        rho=0.18,
    )


def _rdha_asset(device: torch.device) -> da.RDHAAsset:
    coordinate = torch.linspace(-1.0, 1.0, 320, device=device)
    U = torch.stack((0.18 * coordinate, 0.12 + 0.08 * coordinate.square()), dim=1)
    V = torch.stack((0.25 * coordinate, -0.2 * coordinate + 0.05), dim=0)
    Q = torch.tensor(
        [[0.45, -0.10, 0.05, 0.00, 0.02], [-0.08, 0.35, 0.01, -0.04, 0.03]],
        dtype=torch.float32,
        device=device,
    )
    return da.RDHAAsset(
        U=U.float(),
        V=V.float(),
        Q=Q,
        b=torch.tensor([0.18, -0.14], dtype=torch.float32, device=device),
        mean_p1=torch.zeros(5, dtype=torch.float32, device=device),
        std_p1=torch.ones(5, dtype=torch.float32, device=device),
        a_max=0.16,
    )


def _asset_for_model(model: nn.Module, candidate_id: str) -> da.FSRGAsset | da.RDHAAsset:
    device = next(model.parameters()).device
    if candidate_id == da.CANDIDATE_A:
        return _fsrg_asset(
            candidate_id, int(model.id_backbone.time_fuse[1].num_channels), device
        )
    if candidate_id == da.CANDIDATE_B:
        return _fsrg_asset(candidate_id, int(model.id_backbone.t2.norm.num_channels), device)
    return _rdha_asset(device)


def _phase1_receiver_means(dimension: int) -> torch.Tensor:
    """Small source-only fixture with a uniquely oriented rank-two basis."""

    values = torch.zeros(3, 2, dimension, dtype=torch.float32)
    values[0, 0, :2] = torch.tensor([2.0, 0.5])
    values[0, 1, :2] = torch.tensor([1.3, -0.4])
    values[1, 0, :2] = torch.tensor([-0.8, 1.4])
    values[1, 1, :2] = torch.tensor([-1.1, 0.6])
    values[2, 0, :2] = torch.tensor([0.2, -1.7])
    values[2, 1, :2] = torch.tensor([0.7, -1.1])
    return values


def _phase1_fsrg_episode(
    receiver: int, k_shot: int, dimension: int
) -> phase1_assets.FSRGEpisode:
    """One synthetic, disjoint Phase1 source episode in the real 3D tap layout."""

    generator = torch.Generator().manual_seed(127700 + receiver * 31 + k_shot)
    labels = torch.tensor([3, 9], dtype=torch.int64).repeat_interleave(k_shot)
    support = 0.12 * torch.randn(
        2 * k_shot, dimension, 3, generator=generator, dtype=torch.float32
    )
    support[:, 0, :] += torch.where(labels == 3, 0.85, -0.65).reshape(-1, 1)
    support[:, 1, :] += torch.where(labels == 3, -0.30, 0.70).reshape(-1, 1)
    query_labels = torch.tensor([3, 3, 9, 9], dtype=torch.int64)
    query = 0.12 * torch.randn(4, dimension, 3, generator=generator, dtype=torch.float32)
    query[:, 0, :] += torch.where(query_labels == 3, 0.80, -0.60).reshape(-1, 1)
    query[:, 1, :] += torch.where(query_labels == 3, -0.25, 0.65).reshape(-1, 1)
    prefix = f"phase1-fsrg-r{receiver}-k{k_shot}"
    return phase1_assets.FSRGEpisode(
        episode_id=prefix,
        receiver_id=f"phase1-rx-{receiver}",
        k_shot=k_shot,
        support_taps=support,
        support_labels=labels,
        query_taps=query,
        query_labels=query_labels,
        support_physical_ids=tuple(f"{prefix}-s{index}" for index in range(2 * k_shot)),
        query_physical_ids=tuple(f"{prefix}-q{index}" for index in range(4)),
    )


def _phase1_fsrg_callbacks() -> phase1_assets.FSRGLossCallbacks:
    def support_per_sample(
        episode: phase1_assets.FSRGEpisode, rows: torch.Tensor
    ) -> torch.Tensor:
        labels = episode.support_labels
        value = rows.mean(dim=2)
        target = torch.where(labels == 3, 0.40, -0.50).to(dtype=rows.dtype)
        return (value[:, 0] + 0.35 * value[:, 1] - target).square()

    def outer_query_per_sample(
        episode: phase1_assets.FSRGEpisode,
        support: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        support_labels = episode.support_labels
        query_labels = episode.query_labels
        support_value = support.mean(dim=2)
        query_value = query.mean(dim=2)
        means = torch.stack(
            [support_value[support_labels == label].mean(dim=0) for label in (3, 9)]
        )
        expected = torch.where(query_labels == 3, means[0, 0], means[1, 0])
        return (query_value[:, 0] + 0.20 * query_value[:, 1] - expected).square()

    return phase1_assets.FSRGLossCallbacks(
        support_per_sample=support_per_sample,
        outer_query_per_sample=outer_query_per_sample,
    )


def _phase1_fsrg_episodes(dimension: int) -> phase1_assets.FrozenFSRGEpisodes:
    return phase1_assets.FrozenFSRGEpisodes(
        tuple(
            _phase1_fsrg_episode(receiver, k_shot, dimension)
            for receiver in range(3)
            for k_shot in (1, 5)
        )
    )


def _phase1_rdah_episode(receiver: int, k_shot: int) -> phase1_assets.RDHAEpisode:
    generator = torch.Generator().manual_seed(4000 + receiver * 31 + k_shot)
    labels = torch.tensor([4, 13], dtype=torch.int64).repeat_interleave(k_shot)
    support = 0.35 * torch.randn(
        2 * k_shot, 320, generator=generator, dtype=torch.float32
    )
    class_scale = 1.0 + 0.08 * receiver + 0.01 * k_shot
    support[:, 0] += class_scale * torch.where(labels == 4, 0.50, -0.35)
    support[:, 1] += (2.0 - class_scale) * torch.where(labels == 4, -0.25, 0.40)
    support[:, 0] += 0.04 * receiver + 0.005 * k_shot
    support[:, 1] += -0.03 * receiver + 0.004 * k_shot
    query_labels = torch.tensor([4, 4, 13, 13], dtype=torch.int64)
    query = 0.35 * torch.randn(4, 320, generator=generator, dtype=torch.float32)
    query[:, 0] += class_scale * torch.where(query_labels == 4, 0.45, -0.32)
    query[:, 1] += (2.0 - class_scale) * torch.where(query_labels == 4, -0.22, 0.37)
    query[:, 0] += 0.04 * receiver + 0.005 * k_shot
    query[:, 1] += -0.03 * receiver + 0.004 * k_shot
    prefix = f"phase1-rdha-r{receiver}-k{k_shot}"
    return phase1_assets.RDHAEpisode(
        episode_id=prefix,
        receiver_id=f"phase1-rx-{receiver}",
        k_shot=k_shot,
        support_hidden=support,
        support_labels=labels,
        query_hidden=query,
        query_labels=query_labels,
        support_physical_ids=tuple(f"{prefix}-s{index}" for index in range(2 * k_shot)),
        query_physical_ids=tuple(f"{prefix}-q{index}" for index in range(4)),
    )


def _phase1_rdah_episodes() -> phase1_assets.FrozenRDHAEpisodes:
    return phase1_assets.FrozenRDHAEpisodes(
        tuple(
            _phase1_rdah_episode(receiver, k_shot)
            for receiver in range(3)
            for k_shot in (1, 5)
        )
    )


def _phase1_rdah_outer_loss(
    episode: phase1_assets.RDHAEpisode,
    support: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    support_labels = episode.support_labels
    query_labels = episode.query_labels
    means = torch.stack(
        [support[support_labels == label, 0].mean() for label in (4, 13)]
    )
    expected = torch.where(query_labels == 4, means[0], means[1])
    return (query[:, 0] + 0.15 * query[:, 1] - expected).square()


def _phase1_quantized_asset_for_model(
    model: nn.Module, candidate_id: str
) -> phase1_assets.QuantizedFSRGAsset | phase1_assets.QuantizedRDHAAsset:
    """Exercise the required Phase1-train -> INT8 -> decode deployment route."""

    if candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B):
        dimension = (
            int(model.id_backbone.time_fuse[1].num_channels)
            if candidate_id == da.CANDIDATE_A
            else int(model.id_backbone.t2.norm.num_channels)
        )
        trained = phase1_assets.train_fsrg_phase1_asset(
            candidate_id=candidate_id,
            episodes=_phase1_fsrg_episodes(dimension),
            initialization=phase1_assets.canonical_receiver_mean_svd(
                _phase1_receiver_means(dimension), dimension=dimension
            ),
            callbacks=_phase1_fsrg_callbacks(),
        )
        quantized: phase1_assets.QuantizedFSRGAsset | phase1_assets.QuantizedRDHAAsset = (
            phase1_assets.quantize_fsrg_asset(trained.asset)
        )
    else:
        trained = phase1_assets.train_rdah_phase1_asset(
            episodes=_phase1_rdah_episodes(),
            initialization=phase1_assets.canonical_receiver_mean_svd(
                _phase1_receiver_means(320), dimension=320
            ),
            outer_query_per_sample=_phase1_rdah_outer_loss,
        )
        quantized = phase1_assets.quantize_rdah_asset(trained.asset)
    assert trained.receipt.closure_calls >= 1
    assert trained.receipt.initial_gradient_norm > 0.0
    phase1_assets.assert_no_persistent_fp32_sidecar(quantized)
    return quantized


def _assert_query_isolation(
    model: nn.Module,
    query_iq: torch.Tensor,
    result: hooks.D127CheckpointMaterialization,
    asset: da.FSRGAsset | da.RDHAAsset,
) -> None:
    state_before = result.state.a.detach().clone()
    with torch.enable_grad():
        query = hooks.materialize_d127_query(
            model, query_iq, asset=asset, state=result.state
        )
    assert not query.flags.writeable
    np.testing.assert_array_equal(query, result.adapted_cache.query_zid)
    assert torch.equal(state_before, result.state.a)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert result.state.receipt.query_rows_used_for_fit == 0
    assert result.state.receipt.query_state_updates == 0
    assert result.state.receipt.query_gradient_calls == 0


def _real_fsrg_bridge_episode(
    model: nn.Module, *, candidate_id: str = da.CANDIDATE_A
) -> tuple[hooks.D127Phase1CheckpointBridge, phase1_assets.FSRGEpisode]:
    torch.manual_seed(127101)
    episode_id = f"bridge-{candidate_id}-k1"
    raw = hooks.D127Phase1EpisodeIQ(
        episode_id=episode_id,
        support_iq=torch.randn(2, 2, 256, dtype=torch.float32),
        query_iq=torch.randn(4, 2, 256, dtype=torch.float32),
    )
    bridge = hooks.D127Phase1CheckpointBridge(
        model,
        candidate_id=candidate_id,
        episode_iq_by_id={episode_id: raw},
    )
    support = bridge.capture_raw(episode_id, split="support")
    query = bridge.capture_raw(episode_id, split="query")
    labels = torch.tensor([3, 9], dtype=torch.int64)
    query_labels = torch.tensor([3, 9, 3, 9], dtype=torch.int64)
    return bridge, phase1_assets.FSRGEpisode(
        episode_id=episode_id,
        receiver_id="phase1-bridge-rx",
        k_shot=1,
        support_taps=support.tap.detach().clone(),
        support_labels=labels,
        query_taps=query.tap.detach().clone(),
        query_labels=query_labels,
        support_physical_ids=(f"{episode_id}-s0", f"{episode_id}-s1"),
        query_physical_ids=tuple(f"{episode_id}-q{index}" for index in range(4)),
    )


def _real_rdah_bridge_episode(
    model: nn.Module,
) -> tuple[hooks.D127Phase1CheckpointBridge, phase1_assets.RDHAEpisode]:
    torch.manual_seed(127102)
    episode_id = "bridge-rdha-k1"
    raw = hooks.D127Phase1EpisodeIQ(
        episode_id=episode_id,
        support_iq=torch.randn(2, 2, 256, dtype=torch.float32),
        query_iq=torch.randn(4, 2, 256, dtype=torch.float32),
    )
    bridge = hooks.D127Phase1CheckpointBridge(
        model,
        candidate_id=da.CANDIDATE_C,
        episode_iq_by_id={episode_id: raw},
    )
    support = bridge.capture_raw(episode_id, split="support")
    query = bridge.capture_raw(episode_id, split="query")
    labels = torch.tensor([4, 13], dtype=torch.int64)
    query_labels = torch.tensor([4, 13, 4, 13], dtype=torch.int64)
    return bridge, phase1_assets.RDHAEpisode(
        episode_id=episode_id,
        receiver_id="phase1-bridge-rx",
        k_shot=1,
        support_hidden=support.tap.detach().clone(),
        support_labels=labels,
        query_hidden=query.tap.detach().clone(),
        query_labels=query_labels,
        support_physical_ids=(f"{episode_id}-s0", f"{episode_id}-s1"),
        query_physical_ids=tuple(f"{episode_id}-q{index}" for index in range(4)),
    )


def test_synthetic_three_taps_materialize_once_and_query_api_is_closed() -> None:
    torch.manual_seed(127003)
    model = hooks.freeze_d127_checkpoint_model(_TinyDualModel())
    support = torch.randn(4, 2, 256, dtype=torch.float32)
    query = torch.randn(3, 2, 256, dtype=torch.float32)
    labels = torch.tensor([31, 31, 5, 5], dtype=torch.long)

    for candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C):
        asset = _asset_for_model(model, candidate_id)
        result = hooks.materialize_d127_candidate(
            model, support, labels, query, asset=asset
        )
        assert result.candidate_id == candidate_id
        assert result.base_cache.support_zid.shape == (4, 160)
        assert result.base_cache.query_zid.shape == (3, 160)
        assert not result.base_cache.support_zid.flags.writeable
        assert not result.adapted_cache.query_zid.flags.writeable
        assert result.hook_receipt.same_forward_tap_and_final_pre_relu
        assert result.hook_receipt.same_model_downstream
        assert result.hook_receipt.query_rows_used_for_fit == 0
        assert result.hook_receipt.query_state_updates == 0
        assert result.hook_receipt.query_gradient_calls == 0
        assert result.hook_receipt.total_id_backbone_forwards == (
            5 if candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B) else 4
        )
        if candidate_id == da.CANDIDATE_C:
            assert result.hook_receipt.tap_shape == (4, 320)
            assert result.hook_receipt.tap_name == da.TAP_C
            assert result.state.receipt.phase2_backward_calls == 0
        else:
            assert result.hook_receipt.tap_shape == (4, 4, 256)
            assert result.hook_receipt.tap_name == (
                da.TAP_A if candidate_id == da.CANDIDATE_A else da.TAP_B
            )
            assert result.state.receipt.phase2_backward_calls == 1
        assert np.max(
            np.abs(result.base_cache.query_zid - result.adapted_cache.query_zid)
        ) > 1.0e-7
        _assert_query_isolation(model, query, result, asset)

    assert hooks.query_api_forbidden_parameter_names() == ()
    query_parameter_names = set(inspect.signature(hooks.materialize_d127_query).parameters)
    assert not {
        "labels",
        "query_labels",
        "truth",
        "role",
        "update",
        "optimizer",
    }.intersection(query_parameter_names)


def test_phase1_bridge_uses_real_downstream_int8_qknn_and_keeps_outer_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Phase1 bridge must not substitute an approximate feature head."""

    def _blocked(*_args: object, **_kwargs: object) -> object:
        raise TypeError("simulated NumPy2/Torch2.1 ABI mismatch")

    monkeypatch.setattr(torch, "from_numpy", _blocked)
    monkeypatch.setattr(torch, "as_tensor", _blocked)

    torch.manual_seed(127103)
    model = hooks.freeze_d127_checkpoint_model(_TinyDualModel())
    locks = _phase1_qknn_locks()
    bridge, episode = _real_fsrg_bridge_episode(model)
    captured = bridge.capture_episode(episode, split="support")
    assert captured.tap.shape == episode.support_taps.shape
    assert torch.equal(captured.tap, episode.support_taps)
    assert captured.pre_relu.shape == (2, 160)
    assert captured.z160.shape == (2, 160)
    with pytest.raises(hooks.D127CheckpointHookError, match="split"):
        bridge.capture_raw(episode.episode_id, split="outer")  # type: ignore[arg-type]
    with pytest.raises(hooks.D127CheckpointHookError, match="episode ID"):
        bridge.capture_episode(
            phase1_assets.FSRGEpisode(
                episode_id="wrong-episode-id",
                receiver_id=episode.receiver_id,
                k_shot=episode.k_shot,
                support_taps=episode.support_taps,
                support_labels=episode.support_labels,
                query_taps=episode.query_taps,
                query_labels=episode.query_labels,
                support_physical_ids=episode.support_physical_ids,
                query_physical_ids=episode.query_physical_ids,
            ),
            split="support",
        )

    callbacks = bridge.fsrg_loss_callbacks(qknn_locks=locks)
    support_replacement = (
        episode.support_taps.detach().clone() + 0.015
    ).requires_grad_(True)
    support_forward = bridge.forward_with_replacement(
        episode, split="support", replacement=support_replacement
    )
    assert support_forward.pre_relu.requires_grad
    assert support_forward.z160.requires_grad
    pre_gradient = torch.autograd.grad(
        support_forward.pre_relu.square().sum(), support_replacement, retain_graph=True
    )[0]
    z_direction = torch.linspace(
        -0.5, 0.5, 160, dtype=torch.float32
    ).reshape(1, -1)
    z_gradient = torch.autograd.grad(
        (support_forward.z160 * z_direction).sum(),
        support_replacement,
        retain_graph=True,
    )[0]
    assert torch.any(torch.abs(pre_gradient) > 0.0)
    assert torch.any(torch.abs(z_gradient) > 0.0)
    support_loss = callbacks.support_per_sample(episode, support_replacement)
    support_gradient = torch.autograd.grad(support_loss.mean(), support_replacement)[0]
    assert support_loss.shape == (2,)
    assert torch.any(torch.abs(support_gradient) > 0.0)

    asset = _fsrg_asset(da.CANDIDATE_A, episode.dimension, torch.device("cpu"))
    asset.U.requires_grad_(True)
    asset.V.requires_grad_(True)
    coefficient = torch.tensor([0.07, -0.05], dtype=torch.float32)

    def fsrg_outer(rows: torch.Tensor) -> torch.Tensor:
        response = torch.tanh(torch.einsum("rd,ndt->nrt", asset.V, rows))
        return rows + torch.einsum(
            "dr,nrt->ndt", asset.U, response * coefficient.reshape(1, 2, 1)
        )

    adapted_support = fsrg_outer(episode.support_taps)
    adapted_query = fsrg_outer(episode.query_taps)
    support_forward = bridge.forward_with_replacement(
        episode, split="support", replacement=adapted_support
    )
    query_forward = bridge.forward_with_replacement(
        episode, split="query", replacement=adapted_query
    )
    bank = bridge.build_deployment_qknn_bank(
        episode, support_zid=support_forward.z160, qknn_locks=locks
    )
    torch_logits = bridge.deployment_qknn_logits(
        episode,
        support_zid=support_forward.z160,
        query_zid=query_forward.z160,
        qknn_locks=locks,
    )
    expected = qknn.score_zid_student_t_logits(
        bank,
        np.asarray(query_forward.z160.detach().cpu().tolist(), dtype=np.float32),
        metric=qknn.identity_shared_psd_metric(config=locks[1]),
    )
    np.testing.assert_allclose(
        torch_logits.detach().cpu().numpy(), expected.astype(np.float64), rtol=0.0, atol=3.0e-6
    )
    support_bank_gradient, query_gradient = torch.autograd.grad(
        torch_logits.sum(),
        (support_forward.z160, query_forward.z160),
        allow_unused=True,
        retain_graph=True,
    )
    assert support_bank_gradient is None
    assert query_gradient is not None and torch.any(torch.abs(query_gradient) > 0.0)

    swapped = replace(
        episode,
        support_labels=torch.tensor([9, 3], dtype=torch.int64),
        query_labels=torch.tensor([9, 3, 9, 3], dtype=torch.int64),
    )
    swapped_logits = bridge.deployment_qknn_logits(
        swapped,
        support_zid=support_forward.z160,
        query_zid=query_forward.z160,
        qknn_locks=locks,
    )
    np.testing.assert_allclose(
        swapped_logits.detach().cpu().numpy(),
        torch_logits.detach().cpu().numpy()[:, [1, 0]],
        rtol=0.0,
        atol=3.0e-6,
    )

    outer_loss = callbacks.outer_query_per_sample(
        episode, adapted_support, adapted_query
    )
    grad_u, grad_v = torch.autograd.grad(outer_loss.mean(), (asset.U, asset.V))
    assert torch.any(torch.abs(grad_u) > 0.0)
    assert torch.any(torch.abs(grad_v) > 0.0)

    b_bridge, b_episode = _real_fsrg_bridge_episode(
        model, candidate_id=da.CANDIDATE_B
    )
    b_asset = _fsrg_asset(da.CANDIDATE_B, b_episode.dimension, torch.device("cpu"))
    b_asset.U.requires_grad_(True)
    b_asset.V.requires_grad_(True)
    b_callbacks = b_bridge.fsrg_loss_callbacks(qknn_locks=locks)
    b_state = da.fit_fsrg_support_state(
        b_episode.support_taps,
        b_episode.support_labels,
        b_asset,
        lambda adapted: b_callbacks.support_per_sample(b_episode, adapted),
    )
    b_outer_loss = b_callbacks.outer_query_per_sample(
        b_episode,
        da.apply_fsrg_outer(b_episode.support_taps, b_asset, b_state),
        da.apply_fsrg_outer(b_episode.query_taps, b_asset, b_state),
    )
    b_grad_u, b_grad_v = torch.autograd.grad(
        b_outer_loss.mean(), (b_asset.U, b_asset.V)
    )
    assert torch.any(torch.abs(b_grad_u) > 0.0)
    assert torch.any(torch.abs(b_grad_v) > 0.0)

    c_bridge, c_episode = _real_rdah_bridge_episode(model)
    c_asset = _rdha_asset(torch.device("cpu"))
    for parameter in (c_asset.U, c_asset.V, c_asset.Q, c_asset.b):
        parameter.requires_grad_(True)
    c_outer = da.apply_rdah_outer(
        c_episode.support_hidden,
        c_episode.support_labels,
        c_episode.query_hidden,
        c_asset,
    )
    c_loss = c_bridge.rdha_outer_callback(qknn_locks=locks)(
        c_episode, c_outer.adapted_support, c_outer.adapted_query
    )
    gradients = torch.autograd.grad(
        c_loss.mean(), (c_asset.U, c_asset.V, c_asset.Q, c_asset.b)
    )
    assert all(torch.any(torch.abs(gradient) > 0.0) for gradient in gradients)


def test_frozen_audit_replacement_is_strictly_separate_and_same_downstream() -> None:
    """A/C exercise the real hook layout without fabricating a caller graph."""

    torch.manual_seed(127128)
    model = hooks.freeze_d127_checkpoint_model(_TinyDualModel())
    locks = _phase1_qknn_locks()
    cases: tuple[
        tuple[
            hooks.D127Phase1CheckpointBridge,
            phase1_assets.FSRGEpisode | phase1_assets.RDHAEpisode,
            da.FSRGAsset | da.RDHAAsset,
        ],
        ...,
    ] = (
        (*_real_fsrg_bridge_episode(model), _fsrg_asset(da.CANDIDATE_A, 4, torch.device("cpu"))),
        (*_real_rdah_bridge_episode(model), _rdha_asset(torch.device("cpu"))),
    )

    for bridge, episode, asset in cases:
        captured = bridge.capture_episode(episode, split="query")
        frozen_replacement = captured.tap.detach().clone() + 0.0125
        with pytest.raises(
            hooks.D127CheckpointHookError, match="differentiable caller graph"
        ):
            bridge.forward_with_replacement(
                episode, split="query", replacement=frozen_replacement
            )
        live_replacement = frozen_replacement.detach().clone().requires_grad_(True)
        live = bridge.forward_with_replacement(
            episode, split="query", replacement=live_replacement
        )
        frozen = bridge.forward_with_frozen_audit_replacement(
            episode, split="query", replacement=frozen_replacement
        )
        assert live.pre_relu.requires_grad and live.z_id.requires_grad
        assert not frozen.pre_relu.requires_grad and not frozen.z_id.requires_grad
        torch.testing.assert_close(live.tap.detach(), frozen.tap, rtol=0.0, atol=0.0)
        torch.testing.assert_close(live.hidden.detach(), frozen.hidden, rtol=0.0, atol=0.0)
        torch.testing.assert_close(live.pre_relu.detach(), frozen.pre_relu, rtol=0.0, atol=0.0)
        torch.testing.assert_close(live.z_id.detach(), frozen.z_id, rtol=0.0, atol=0.0)

        if isinstance(episode, phase1_assets.FSRGEpisode):
            assert isinstance(asset, da.FSRGAsset)
            callbacks = bridge.fsrg_loss_callbacks(qknn_locks=locks, frozen_audit=True)
            state = da.fit_fsrg_support_state(
                episode.support_taps,
                episode.support_labels,
                asset,
                lambda adapted: callbacks.support_per_sample(episode, adapted),
            )
            support = da.adapt_fsrg_support(episode.support_taps, asset, state)
            query = da.adapt_fsrg_query(episode.query_taps, asset, state)
            losses = callbacks.outer_query_per_sample(episode, support, query)
        else:
            assert isinstance(asset, da.RDHAAsset)
            callback = bridge.rdha_outer_callback(qknn_locks=locks, frozen_audit=True)
            state = da.fit_rdah_support_state(
                episode.support_hidden, episode.support_labels, asset
            )
            support = da.adapt_rdah_support(episode.support_hidden, asset, state)
            query = da.adapt_rdah_query(episode.query_hidden, asset, state)
            losses = callback(episode, support, query)
        assert not losses.requires_grad
        assert bool(torch.isfinite(losses).all().item())

    a_bridge, a_episode = _real_fsrg_bridge_episode(model)
    a_captured = a_bridge.capture_episode(a_episode, split="query")
    with pytest.raises(hooks.D127CheckpointHookError, match="finite float32"):
        a_bridge.forward_with_frozen_audit_replacement(
            a_episode,
            split="query",
            replacement=torch.full_like(a_captured.tap, float("nan")),
        )


def test_signed_totalization_uses_pre_relu_only_for_relu_zero_rows() -> None:
    pre_relu = torch.zeros(2, 160, dtype=torch.float32)
    pre_relu[0, 0] = -2.0
    pre_relu[0, 1] = 3.0
    pre_relu[1, 2] = -4.0
    signed = hooks._unit_rows_from_pre_relu(pre_relu)
    assert signed[0, 0] == pytest.approx(0.0)
    assert signed[0, 1] > 0.0
    assert signed[1, 2] < 0.0
    assert torch.allclose(torch.linalg.vector_norm(signed, dim=1), torch.ones(2))


def test_hook_exception_removes_temporary_hooks_before_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed hooked forward must not leak an intervention into the retry."""

    torch.manual_seed(127041)
    model = hooks.freeze_d127_checkpoint_model(_TinyDualModel())
    support = torch.randn(4, 2, 256, dtype=torch.float32)
    query = torch.randn(2, 2, 256, dtype=torch.float32)
    labels = torch.tensor([31, 31, 5, 5], dtype=torch.long)
    asset = _asset_for_model(model, da.CANDIDATE_A)

    def injected_forward_failure(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected D127 hook-forward failure")

    with monkeypatch.context() as patch:
        patch.setattr(model.id_backbone, "forward", injected_forward_failure)
        with pytest.raises(RuntimeError, match="injected D127 hook-forward failure"):
            hooks.materialize_d127_candidate(model, support, labels, query, asset=asset)

    assert not model.id_backbone.time_fuse[1]._forward_hooks
    assert not model.id_backbone.cls_head.joint_proj[0]._forward_hooks
    assert not model.id_backbone.cls_head.joint_proj[0]._forward_pre_hooks
    recovered = hooks.materialize_d127_candidate(model, support, labels, query, asset=asset)
    assert recovered.hook_receipt.same_forward_tap_and_final_pre_relu


def test_real_checkpoint_three_taps_strict_rebuild_and_no_query_smoke() -> None:
    assert CHECKPOINT_PATH.is_file(), CHECKPOINT_PATH
    model, checkpoint_receipt = hooks.load_d127_frozen_checkpoint(
        CHECKPOINT_PATH, device="cpu"
    )
    assert checkpoint_receipt["model_reconstruction"]["checkpoint_load_strict"] is True
    assert checkpoint_receipt["model_reconstruction"]["input_len"] == 256
    assert checkpoint_receipt["model_reconstruction"]["num_domains_from_state"] == 14
    assert checkpoint_receipt["num_domains"] == 14
    assert checkpoint_receipt["all_checkpoint_parameters_frozen"] is True

    torch.manual_seed(127062)
    support = torch.randn(4, 2, 256, dtype=torch.float32)
    query = torch.randn(2, 2, 256, dtype=torch.float32)
    labels = torch.tensor([7, 7, 19, 19], dtype=torch.long)
    query_parameter_names = set(inspect.signature(hooks.materialize_d127_query).parameters)
    assert hooks.query_api_forbidden_parameter_names() == ()
    assert not {
        "labels",
        "query_labels",
        "truth",
        "role",
        "quota",
        "update",
        "optimizer",
    }.intersection(query_parameter_names)
    for candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C):
        quantized = _phase1_quantized_asset_for_model(model, candidate_id)
        phase1_assets.assert_no_persistent_fp32_sidecar(quantized)
        asset = quantized.decode(device=next(model.parameters()).device)
        assert asset.candidate_id == candidate_id
        assert all(parameter.requires_grad is False for parameter in model.parameters())
        result = hooks.materialize_d127_candidate(
            model, support, labels, query, asset=asset
        )
        assert result.base_cache.support_zid.shape == (len(support), 160)
        assert result.adapted_cache.support_zid.shape == (len(support), 160)
        assert result.base_cache.query_zid.shape == (len(query), 160)
        assert result.adapted_cache.query_zid.shape == (len(query), 160)
        assert result.hook_receipt.tap_shape[0] == len(support)
        assert result.hook_receipt.tap_shape[1] == (
            320 if candidate_id == da.CANDIDATE_C else asset.dimension
        )
        assert len(result.hook_receipt.tap_shape) == (
            2 if candidate_id == da.CANDIDATE_C else 3
        )
        assert result.hook_receipt.tap_name == {
            da.CANDIDATE_A: da.TAP_A,
            da.CANDIDATE_B: da.TAP_B,
            da.CANDIDATE_C: da.TAP_C,
        }[candidate_id]
        assert result.hook_receipt.query_rows_used_for_fit == 0
        assert result.hook_receipt.query_state_updates == 0
        assert result.hook_receipt.query_gradient_calls == 0
        assert np.max(
            np.abs(result.base_cache.query_zid - result.adapted_cache.query_zid)
        ) > 1.0e-8
        _assert_query_isolation(model, query, result, asset)
