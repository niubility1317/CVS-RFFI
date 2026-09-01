from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _ref(index: int, tx: int, rx: int, role: str = "L_s"):
    from cvsrffi.meta_episodes import MetaSampleRef

    return MetaSampleRef(
        dataset_index=index,
        tx_i=tx,
        rx_i=rx,
        day_i=1,
        eq_i=0,
        capture_block_i=1,
        physical_sample_id=f"p-{index}",
        role=role,
        view="leo_clear_weak",
    )


def _batch():
    from cvsrffi.meta_episodes import EpisodeKind, MetaEpisode
    from cvsrffi.meta_trainer import MetaEpisodeBatch

    support = (_ref(0, 0, 1), _ref(1, 0, 1), _ref(2, 1, 2), _ref(3, 1, 2))
    query_adapt = (_ref(4, 0, 1), _ref(5, 1, 1), _ref(6, 0, 2), _ref(7, 1, 2))
    query_guard = (_ref(8, 2, 1), _ref(9, 2, 2))
    episode = MetaEpisode(
        kind=EpisodeKind.RX_HOLDOUT,
        support=support,
        query_adapt=query_adapt,
        query_guard=query_guard,
        adapt_class_ids=frozenset({0, 1}),
        guard_class_ids=frozenset({2}),
        k_shot=2,
        seed=17,
    )
    support_core = torch.tensor([[1.0, 0.2], [0.8, -0.1], [-0.3, 1.0], [0.1, 0.9]])
    query_core = torch.tensor(
        [[0.9, 0.0], [0.0, 0.9], [1.1, 0.2], [-0.2, 1.1], [-0.8, -0.7], [-0.6, -1.0]]
    )
    waveform = torch.linspace(1.0, 0.5, 16)
    support_x = support_core.unsqueeze(-1) * waveform
    query_x = query_core.unsqueeze(-1) * waveform
    return MetaEpisodeBatch(
        episode=episode,
        support_x=support_x,
        support_y=torch.tensor([0, 0, 1, 1]),
        query_x=query_x,
        query_y=torch.tensor([0, 1, 0, 1, 2, 2]),
        adapt_mask=torch.tensor([True, True, True, True, False, False]),
        guard_mask=torch.tensor([False, False, False, False, True, True]),
        frozen_prototypes=torch.zeros(3, 2),
    )


def _trainable_bank():
    from cvsrffi.meta_weight_bank import (
        BlockSpec,
        DeltaBankEntry,
        DeltaTaskKey,
        WEIGHT_DELTA_BANK_SCHEMA,
        WeightDeltaBank,
    )

    weight_basis = torch.nn.Parameter(
        torch.tensor([[0.6], [-0.4], [0.2], [-0.5], [0.7], [-0.3]], dtype=torch.float32)
    )
    bias_basis = torch.nn.Parameter(torch.tensor([[0.3], [-0.2], [0.5]], dtype=torch.float32))
    bank = WeightDeltaBank(
        schema=WEIGHT_DELTA_BANK_SCHEMA,
        base_checkpoint_id="base-meta",
        task_keys=(DeltaTaskKey("rx-1", "day-1", "leo_clear_weak", 2),),
        entries=(
            DeltaBankEntry(
                spec=BlockSpec(
                    "t3",
                    ("id_backbone.t3.weight",),
                    ((2, 3),),
                    ("torch.float32",),
                ),
                basis=weight_basis,
                task_coefficients=torch.ones(1, 1),
                effective_rank=1,
                relative_error=0.0,
            ),
            DeltaBankEntry(
                spec=BlockSpec(
                    "fusion",
                    ("id_backbone.fusion.bias",),
                    ((3,),),
                    ("torch.float32",),
                ),
                basis=bias_basis,
                task_coefficients=torch.ones(1, 1),
                effective_rank=1,
                relative_error=0.0,
            ),
        ),
    )
    return bank, (weight_basis, bias_basis)


def _encoder():
    from cvsrffi.meta_support_set_encoder import SupportSetEncoder

    torch.manual_seed(31)
    return SupportSetEncoder(
        feature_dim=685,
        coefficient_dim=2,
        block_count=2,
        hidden_dim=5,
        lr_min=0.01,
        lr_max=0.08,
    )


def _base_state():
    return {
        "id_backbone.t3.weight": torch.zeros(2, 3),
        "id_backbone.fusion.bias": torch.zeros(3),
    }


def _optimizer(encoder, basis_tensors, *, omit_last_basis=False, extra=None):
    parameters = list(encoder.parameters()) + list(
        basis_tensors[:-1] if omit_last_basis else basis_tensors
    )
    if extra is not None:
        parameters.append(extra)
    return torch.optim.SGD(parameters, lr=0.01)


def _functional_forward(calls: list[int]):
    def forward(state, x):
        calls.append(x.data_ptr())
        core = x[:, :, 0]
        return core @ state["id_backbone.t3.weight"] + state["id_backbone.fusion.bias"]

    return forward


class _SupportFeatureModel(nn.Module):
    def forward(self, values, return_aux=True):
        assert return_aux is True
        core = values.float().mean(dim=-1)
        weights = torch.linspace(0.25, 1.25, 160, device=values.device).view(1, -1)
        z_id = (core[:, :1] + 1.5) * weights
        t_emb = (core[:, 1:] + 2.0) * weights
        f_emb = (core.mean(dim=1, keepdim=True) + 2.5) * weights.flip(1)
        return {
            "z_id": z_id,
            "aux_id": {"t_emb": t_emb, "f_emb": f_emb},
            "z_dom": torch.full_like(z_id, float("nan")),
        }


def test_meta_bank_step_keeps_query_out_of_inner_loop_and_backpropagates_all_outer_paths() -> None:
    """Query leakage or detached encoder/gate/LR/basis paths breaks this contract."""
    from cvsrffi.meta_bank_trainer import MetaBankTrainerConfig, run_meta_bank_step

    batch = _batch()
    bank, basis_tensors = _trainable_bank()
    encoder = _encoder()
    calls: list[int] = []
    config = MetaBankTrainerConfig(
        source_receiver_ids=(1, 2),
        inner_steps=1,
        receiver_cvar_fraction=0.5,
        receiver_cvar_weight=0.4,
        worst_class_guard_weight=0.7,
    )
    optimizer = _optimizer(encoder, basis_tensors)

    result = run_meta_bank_step(
        _functional_forward(calls),
        base_state=_base_state(),
        base_checkpoint_id="base-meta",
        bank=bank,
        support_encoder=encoder,
        support_feature_model=_SupportFeatureModel(),
        batch=batch,
        config=config,
        optimizer=optimizer,
    )

    assert calls == [batch.support_x.data_ptr(), batch.query_x.data_ptr()]
    assert result.fast_state.parameters["id_backbone.t3.weight"].shape == (2, 3)
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.count_nonzero(parameter.grad))
    final_head = encoder.rho[-1]
    assert isinstance(final_head, torch.nn.Linear)
    q_end = encoder.coefficient_dim
    u_end = q_end + 1
    gate_end = u_end + encoder.block_count
    assert bool(torch.count_nonzero(final_head.bias.grad[:q_end]))
    assert bool(torch.count_nonzero(final_head.bias.grad[u_end:gate_end]))
    assert bool(torch.count_nonzero(final_head.bias.grad[gate_end:]))
    for basis in basis_tensors:
        assert basis.grad is not None
        assert bool(torch.isfinite(basis.grad).all())
        assert bool(torch.count_nonzero(basis.grad))


def test_meta_bank_step_outer_objective_matches_mean_receiver_cvar_and_worst_guard_class() -> None:
    """Dropping or mis-grouping any of the three outer terms changes these literals."""
    from cvsrffi.meta_bank_trainer import MetaBankTrainerConfig, run_meta_bank_step

    batch = _batch()
    bank, basis_tensors = _trainable_bank()
    encoder = _encoder()
    calls: list[int] = []
    config = MetaBankTrainerConfig(
        source_receiver_ids=(1, 2),
        inner_steps=1,
        receiver_cvar_fraction=0.5,
        receiver_cvar_weight=0.4,
        worst_class_guard_weight=0.7,
    )
    forward = _functional_forward(calls)
    optimizer = _optimizer(encoder, basis_tensors)
    result = run_meta_bank_step(
        forward,
        base_state=_base_state(),
        base_checkpoint_id="base-meta",
        bank=bank,
        support_encoder=encoder,
        support_feature_model=_SupportFeatureModel(),
        batch=batch,
        config=config,
        optimizer=optimizer,
    )
    logits = forward(result.fast_state.parameters, batch.query_x)
    row_losses = F.cross_entropy(logits, batch.query_y, reduction="none")
    adapt_mean = row_losses[:4].mean()
    receiver_means = torch.stack((row_losses[:2].mean(), row_losses[2:4].mean()))
    receiver_cvar = receiver_means.max()
    guard_worst = row_losses[4:].mean()
    expected_total = adapt_mean + 0.4 * receiver_cvar + 0.7 * guard_worst

    assert result.query_adapt_mean.item() == pytest.approx(adapt_mean.item(), rel=1e-6)
    assert result.receiver_cvar.item() == pytest.approx(receiver_cvar.item(), rel=1e-6)
    assert result.worst_class_guard.item() == pytest.approx(guard_worst.item(), rel=1e-6)
    assert result.loss.item() == pytest.approx(expected_total.item(), rel=1e-6)


def test_meta_bank_step_reuses_source_episode_role_checks() -> None:
    """A target or Phase2-marked role must be rejected by the existing source validator."""
    from cvsrffi.meta_bank_trainer import MetaBankTrainerConfig, run_meta_bank_step

    batch = _batch()
    bad_support = (replace(batch.episode.support[0], role="target_support"), *batch.episode.support[1:])
    bad_batch = replace(batch, episode=replace(batch.episode, support=bad_support))
    bank, basis_tensors = _trainable_bank()
    encoder = _encoder()

    with pytest.raises(ValueError, match="forbidden target/query field"):
        run_meta_bank_step(
            _functional_forward([]),
            base_state=_base_state(),
            base_checkpoint_id="base-meta",
            bank=bank,
            support_encoder=encoder,
            support_feature_model=_SupportFeatureModel(),
            batch=bad_batch,
            config=MetaBankTrainerConfig(source_receiver_ids=(1, 2), inner_steps=1),
            optimizer=_optimizer(encoder, basis_tensors),
        )


def test_meta_bank_step_rejects_optimizer_missing_basis_after_clearing_stale_grad() -> None:
    """A stale omitted-basis gradient must not satisfy the current-episode proof."""
    from cvsrffi.meta_bank_trainer import MetaBankTrainerConfig, run_meta_bank_step

    batch = _batch()
    bank, basis_tensors = _trainable_bank()
    encoder = _encoder()
    omitted_basis = basis_tensors[-1]
    omitted_basis.grad = torch.full_like(omitted_basis, 123.0)
    calls: list[int] = []

    with pytest.raises(ValueError, match="optimizer"):
        run_meta_bank_step(
            _functional_forward(calls),
            base_state=_base_state(),
            base_checkpoint_id="base-meta",
            bank=bank,
            support_encoder=encoder,
            support_feature_model=_SupportFeatureModel(),
            batch=batch,
            config=MetaBankTrainerConfig(source_receiver_ids=(1, 2), inner_steps=1),
            optimizer=_optimizer(encoder, basis_tensors, omit_last_basis=True),
        )

    assert omitted_basis.grad is None
    assert calls == []


def test_meta_bank_step_rejects_optimizer_with_unrelated_parameter_before_forward() -> None:
    """An optimizer may contain only the encoder and bank-basis parameters."""
    from cvsrffi.meta_bank_trainer import MetaBankTrainerConfig, run_meta_bank_step

    batch = _batch()
    bank, basis_tensors = _trainable_bank()
    encoder = _encoder()
    unrelated = torch.nn.Parameter(torch.tensor(1.0))
    calls: list[int] = []

    with pytest.raises(ValueError, match="optimizer"):
        run_meta_bank_step(
            _functional_forward(calls),
            base_state=_base_state(),
            base_checkpoint_id="base-meta",
            bank=bank,
            support_encoder=encoder,
            support_feature_model=_SupportFeatureModel(),
            batch=batch,
            config=MetaBankTrainerConfig(source_receiver_ids=(1, 2), inner_steps=1),
            optimizer=_optimizer(encoder, basis_tensors, extra=unrelated),
        )

    assert calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inner_steps": 0},
        {"receiver_cvar_weight": 0.0},
        {"worst_class_guard_weight": 0.0},
    ],
)
def test_meta_bank_config_cannot_disable_required_inner_or_outer_paths(kwargs) -> None:
    """Zero steps or zero weights would silently remove a required gradient path."""
    from cvsrffi.meta_bank_trainer import MetaBankTrainerConfig

    with pytest.raises(ValueError):
        MetaBankTrainerConfig(source_receiver_ids=(1, 2), **kwargs)


def test_meta_bank_step_uses_canonical_builder_on_support_only(monkeypatch) -> None:
    import cvsrffi.meta_bank_trainer as subject
    from cvsrffi.marc_ot_support_features import build_marc_ot_support_features

    batch = _batch()
    bank, basis_tensors = _trainable_bank()
    encoder = _encoder()
    observed: list[tuple[int, tuple[str, ...]]] = []
    observed_scope: list[tuple[str, str, bool, object]] = []

    def spy_builder(model, iq, labels, tokens, **kwargs):
        observed.append((iq.data_ptr(), tuple(tokens)))
        observed_scope.append(
            (
                str(kwargs["scope"]),
                str(kwargs["fit_scope"]),
                bool(kwargs["validated_unpadded"]),
                kwargs.get("effective_mask"),
            )
        )
        return build_marc_ot_support_features(model, iq, labels, tokens, **kwargs)

    monkeypatch.setattr(subject, "build_marc_ot_support_features", spy_builder)
    subject.run_meta_bank_step(
        _functional_forward([]),
        base_state=_base_state(),
        base_checkpoint_id="base-meta",
        bank=bank,
        support_encoder=encoder,
        support_feature_model=_SupportFeatureModel(),
        batch=batch,
        config=subject.MetaBankTrainerConfig(source_receiver_ids=(1, 2), inner_steps=1),
        optimizer=_optimizer(encoder, basis_tensors),
    )

    assert observed == [
        (
            batch.support_x.data_ptr(),
            tuple(row.physical_sample_id for row in batch.episode.support),
        )
    ]
    assert observed_scope == [("phase1_source", "full_episode", True, None)]
    query_tokens = {row.physical_sample_id for row in batch.episode.query_adapt + batch.episode.query_guard}
    assert query_tokens.isdisjoint(observed[0][1])


def test_meta_bank_step_rejects_forged_phase1_k_metadata() -> None:
    from cvsrffi.meta_bank_trainer import (
        MetaBankTrainerConfig,
        run_meta_bank_step,
    )

    batch = _batch()
    forged = replace(batch, episode=replace(batch.episode, k_shot=5))
    bank, basis_tensors = _trainable_bank()
    encoder = _encoder()

    with pytest.raises(ValueError, match="full.*K mismatch"):
        run_meta_bank_step(
            _functional_forward([]),
            base_state=_base_state(),
            base_checkpoint_id="base-meta",
            bank=bank,
            support_encoder=encoder,
            support_feature_model=_SupportFeatureModel(),
            batch=forged,
            config=MetaBankTrainerConfig(source_receiver_ids=(1, 2), inner_steps=1),
            optimizer=_optimizer(encoder, basis_tensors),
        )
