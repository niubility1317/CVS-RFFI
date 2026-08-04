from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from cvsrffi import stage2_next_r1_assets as assets
from cvsrffi import stage2_next_r1_fabr as fabr
from cvsrffi import stage2_next_r1_real as real
from cvsrffi import stage2_next_r1_runtime as runtime
from cvsrffi import stage2_next_r1_tsl as tsl


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays():
    receivers = tuple(f"rx{i}" for i in range(7))
    classes = tuple(f"tx{i}" for i in range(6))
    rows = [(receiver, label, j) for receiver in receivers for label in classes for j in range(14)]
    iq = np.zeros((588, 2, 8), dtype=np.float32)
    for i, (receiver, label, j) in enumerate(rows):
        iq[i, 0] = np.float32((int(receiver[2:]) + 1) / 10)
        iq[i, 1] = np.float32((int(label[2:]) + 1) / 10 + j / 1000)
    strings = {
        "receiver_ids": np.asarray([item[0] for item in rows], dtype="<U4"),
        "day_ids": np.asarray(["d0"] * 588, dtype="<U2"),
        "physical_ids": np.asarray([f"p-{r}-{c}-{j:02d}" for r, c, j in rows], dtype="<U20"),
    }
    selected = {
        "received_iq": iq,
        **strings,
        "scenario_names": np.asarray(["leo_a_weak"] * 588, dtype="<U10"),
        "observation_ids": np.asarray([f"o-{i:03d}" for i in range(588)], dtype="<U8"),
    }
    labels = {
        "z_dom": np.zeros((588, 1), dtype=np.float32),
        "pre_relu": np.zeros((588, 1), dtype=np.float32),
        "receiver_ids": strings["receiver_ids"],
        "day_ids": strings["day_ids"],
        "tx_labels": np.asarray([item[1] for item in rows], dtype="<U4"),
        "physical_ids": strings["physical_ids"],
    }
    return selected, labels


def _write_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    selected, labels = _arrays()
    selected_path = tmp_path / "selected.npz"
    labels_path = tmp_path / "labels.npz"
    np.savez(selected_path, **selected)
    np.savez(labels_path, **labels)
    receipt = {
        "schema": real.SELECTED_RECEIPT_SCHEMA,
        "archive_sha256": _sha(selected_path),
        "row_count": 588,
        "contains_only_selected_ls_rows": True,
        "source_pool_labels_persisted": False,
        "clean_iq_access": False,
        "target_access": False,
        "formal_query_access": False,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return selected_path, labels_path, receipt_path


def _rows(tmp_path) -> real.NextR1RealRows:
    selected, labels, receipt = _write_inputs(tmp_path)
    return real.load_next_r1_real_rows(
        selected_iq_archive=selected,
        selected_iq_archive_sha256=_sha(selected),
        selected_iq_receipt=receipt,
        selected_iq_receipt_sha256=_sha(receipt),
        ls_label_join_archive=labels,
        ls_label_join_archive_sha256=_sha(labels),
    )


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_proj = nn.Sequential(nn.Linear(320, 160), nn.ReLU())
        self.out = nn.Linear(160, 6)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = nn.Module(); self.t1.norm = nn.LayerNorm(72)
        self.t2 = nn.Module(); self.t2.norm = nn.LayerNorm(96)
        self.t3 = nn.Module(); self.t3.norm = nn.LayerNorm(96)
        self.cls_head = _Head()

    def forward(self, x, y=None, return_aux=True, domain_labels=None):
        scalar = x.mean(dim=(1, 2), keepdim=False)[:, None]
        a = scalar * self.t1.norm.weight + self.t1.norm.bias
        b = scalar * self.t2.norm.weight + self.t2.norm.bias
        c = scalar * self.t3.norm.weight + self.t3.norm.bias
        hidden = torch.cat((a, b, c, torch.zeros((len(x), 56), device=x.device)), dim=1)
        feat = self.cls_head.joint_proj(hidden)
        return {"logits": self.cls_head.out(feat), "feat_joint": feat}


class _Model(nn.Module):
    def __init__(self):
        super().__init__(); self.id_backbone = _Backbone()


def test_pinned_no_pickle_join_and_label_mismatch_fail_closed(tmp_path) -> None:
    rows = _rows(tmp_path)
    assert rows.received_iq.shape == (588, 2, 8)
    assert len(rows.receiver_registry) == 7 and len(rows.class_registry) == 6
    assert rows.receipt["historical_features_consumed"] is False

    selected, labels, receipt = _write_inputs(tmp_path / "second")
    with np.load(labels, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["physical_ids"] = values["physical_ids"].copy()
    values["physical_ids"][0] = "wrong"
    np.savez(labels, **values)
    with pytest.raises(real.NextR1RealError, match="do not join exactly"):
        real.load_next_r1_real_rows(
            selected_iq_archive=selected, selected_iq_archive_sha256=_sha(selected),
            selected_iq_receipt=receipt, selected_iq_receipt_sha256=_sha(receipt),
            ls_label_join_archive=labels, ls_label_join_archive_sha256=_sha(labels),
        )


def test_real_shaped_tap_four_gradient_blocks_and_exact_restore(tmp_path) -> None:
    rows = _rows(tmp_path)
    bridge = real.NextR1RealModelBridge(_Model(), rows, "1" * 64, "cpu")
    logits, z160 = bridge.forward_indices((0, 1))
    assert logits.shape == (2, 6) and z160.shape == (2, 160)
    before = {
        name: value.detach().clone()
        for name, value in bridge.model.state_dict().items()
    }
    basis = np.zeros((160, 2), dtype=np.float64); basis[0, 0] = 1.0
    bridge.forward_indices((0, 1), block_id="joint_proj_bias", basis=basis, coefficient=np.asarray([0.01, 0], dtype=np.float32))
    assert all(
        torch.equal(value, bridge.model.state_dict()[name])
        for name, value in before.items()
    )
    blocks = bridge.gradient_blocks((0, 1), microbatch_size=2)
    assert tuple(block.block_id for block in blocks) == fabr.BLOCK_TIE_ORDER
    assert tuple(block.gradients.shape for block in blocks) == tuple((2, fabr.BLOCK_DIMENSIONS[key]) for key in fabr.BLOCK_TIE_ORDER)
    assert all(not parameter.requires_grad for parameter in bridge.model.parameters())


def test_iq_tensor_bridge_does_not_use_numpy_c_api_from_numpy(tmp_path, monkeypatch) -> None:
    rows = _rows(tmp_path)
    bridge = real.NextR1RealModelBridge(_Model(), rows, "1" * 64, "cpu")
    monkeypatch.setattr(
        torch, "from_numpy", lambda _value: (_ for _ in ()).throw(RuntimeError("forbidden"))
    )
    tensor = bridge._indices_tensor((0, 1))
    assert tensor.dtype == torch.float32 and tuple(tensor.shape) == (2, 2, 8)
    assert np.array_equal(tensor.numpy(), rows.received_iq[[0, 1]])


def test_held_pair_creates_420_rows_thirty_cells_and_complete_loo(tmp_path) -> None:
    rows = _rows(tmp_path)
    bridge = real.NextR1RealModelBridge(_Model(), rows, "1" * 64, "cpu")
    indices = real._fit_indices(rows, "rx6", "tx5")
    cells, bindings = real._cells_and_loo(bridge, indices, "rx6", "tx5")
    assert len(indices) == 420 and len(cells) == 30 and len(bindings) == 30
    assert all(len(cell.physical_ids) == 14 for cell in cells)
    assert len({pid for binding in bindings for pid in binding.fold.validation_physical_ids}) == 420


def test_typed_external_smoke_requires_pinned_checkpoint_and_real_forward(tmp_path) -> None:
    basis = np.zeros((144, 2), dtype=np.int8); basis[0, 0] = basis[1, 1] = 64
    asset = fabr.FABRAsset(
        checkpoint_sha256="1" * 64, phase1_seal_sha256="2" * 64,
        phase1_selection_sha256="3" * 64, block_id="t1_norm_affine",
        basis_qint8=basis, basis_scale_fp16=np.asarray([1 / 64, 1 / 64], dtype=np.float16),
        fisher_k_fp16=np.eye(2, dtype=np.float16),
        forward_jitter_tolerance_fp16=np.asarray([0], dtype=np.float16),
    )
    prior = tsl.TSLPhase1Prior(
        q_logv0=np.zeros(160, dtype=np.int8), scale_logv0=np.float16(.01),
        offset_logv0=np.float16(0), nu0=np.float16(2), rho_h=np.float16(1),
        checkpoint_sha256="1" * 64, cell_physical_id_root_sha256="4" * 64,
        representation_rule_sha256=fabr.REPRESENTATION_RULE_SHA256,
    )
    receipt = {
        "schema": assets.BUNDLE_SCHEMA, "selection_sha256": "3" * 64,
        "fold_seal_sha256": "5" * 64, "checkpoint_sha256": "1" * 64,
        "representation_rule_sha256": fabr.REPRESENTATION_RULE_SHA256,
        "row_phase1_seal_sha256": "2" * 64,
        "phase1_cell_physical_id_root_sha256": "4" * 64,
        "phase1_receiver_registry_sha256": "6" * 64,
        "phase1_class_registry_sha256": "7" * 64,
        "selected_block_id": "t1_norm_affine", "tsl_prior_sha256": prior.prior_sha256,
    }
    bundle = assets.NextR1Phase1AssetBundle(asset, prior, "5" * 64, "6" * 64, "7" * 64, receipt)
    rows = _rows(tmp_path / "rows")
    bridge = real.NextR1RealModelBridge(_Model(), rows, "1" * 64, "cpu")
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"real-smoke-checkpoint")
    checkpoint_sha = _sha(checkpoint)
    bridge.checkpoint_sha256 = checkpoint_sha
    asset = fabr.FABRAsset(
        checkpoint_sha256=checkpoint_sha, phase1_seal_sha256="2" * 64,
        phase1_selection_sha256="3" * 64, block_id="t1_norm_affine",
        basis_qint8=basis, basis_scale_fp16=np.asarray([1 / 64, 1 / 64], dtype=np.float16),
        fisher_k_fp16=np.eye(2, dtype=np.float16),
        forward_jitter_tolerance_fp16=np.asarray([0], dtype=np.float16),
    )
    prior = tsl.TSLPhase1Prior(
        q_logv0=np.zeros(160, dtype=np.int8), scale_logv0=np.float16(.01),
        offset_logv0=np.float16(0), nu0=np.float16(2), rho_h=np.float16(1),
        checkpoint_sha256=checkpoint_sha, cell_physical_id_root_sha256="4" * 64,
        representation_rule_sha256=fabr.REPRESENTATION_RULE_SHA256,
    )
    receipt["checkpoint_sha256"] = checkpoint_sha
    receipt["tsl_prior_sha256"] = prior.prior_sha256
    bundle = assets.NextR1Phase1AssetBundle(asset, prior, "5" * 64, "6" * 64, "7" * 64, receipt)
    smoke = real.verified_checkpoint_smoke(
        bridge, bundle, checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha, smoke_indices=(0, 1),
    )
    assert type(smoke) is runtime.NextR1VerifiedCheckpointSmoke
    assert smoke.verification_mode == "verified_external_receipt"

    with pytest.raises(real.NextR1RealError, match="SHA256 drift"):
        real.verified_checkpoint_smoke(
            bridge, bundle, checkpoint_path=checkpoint,
            checkpoint_sha256="f" * 64, smoke_indices=(0, 1),
        )


def test_no_truth_smoke_head_is_class_permutation_equivariant_and_keeps_exact_ties() -> None:
    classes = tuple(f"tx{index}" for index in range(6))
    support = np.zeros((6, 160), dtype=np.float32)
    support[np.arange(6), np.arange(6)] = 1.0
    query = np.zeros((1, 160), dtype=np.float32)
    query[0, :2] = np.float32(1.0 / np.sqrt(2.0))
    support_cache = runtime.NextR1FeatureCache(support, tuple(f"s{index}" for index in range(6)))
    query_cache = runtime.NextR1FeatureCache(query, ("q0",))
    context = runtime.NextR1ArmContext("R0", support_cache, query_cache, classes, classes)
    logits = real.no_truth_head(context)
    assert logits[0, 0] == logits[0, 1]
    with pytest.raises(tsl.TSLTieUnresolvedError):
        tsl.require_unique_float32_top(logits)

    reversed_classes = tuple(reversed(classes))
    permuted = runtime.NextR1ArmContext(
        "R0", support_cache, query_cache, classes, reversed_classes
    )
    assert np.array_equal(real.no_truth_head(permuted), logits[:, ::-1])


def test_phase1_directional_validation_rejects_exact_tie_before_class_order() -> None:
    logits = np.asarray([[2.0, 2.0, 0.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    with pytest.raises(real.NextR1RealError, match="unresolved exact tie"):
        real._strict_phase1_predictions(logits, name="baseline")
    with pytest.raises(real.NextR1RealError, match="unresolved exact tie"):
        real._strict_phase1_predictions(logits[:, ::-1], name="baseline")


def _normalized_random(rows: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(size=(rows, 160)).astype(np.float32)
    return np.ascontiguousarray(value / np.linalg.norm(value, axis=1, keepdims=True))


def test_frozen_qknn_callback_uses_the_matching_phase1_k_lock() -> None:
    classes = tuple(f"tx{index}" for index in range(6))
    support = runtime.NextR1FeatureCache(
        _normalized_random(30, seed=11), tuple(f"s{index}" for index in range(30))
    )
    query = runtime.NextR1FeatureCache(
        _normalized_random(7, seed=12), tuple(f"q{index}" for index in range(7))
    )
    labels = tuple(label for label in classes for _ in range(5))
    context = runtime.NextR1ArmContext("R0", support, query, labels, classes)
    logits = real.frozen_qknn_head(context)
    assert logits.dtype == np.float32 and logits.shape == (7, 6)
    assert np.isfinite(logits).all()


def test_frozen_historical_d92_full160_callback_is_k5_only() -> None:
    classes = tuple(f"tx{index}" for index in range(6))
    support = runtime.NextR1FeatureCache(
        _normalized_random(30, seed=21), tuple(f"s{index}" for index in range(30))
    )
    query = runtime.NextR1FeatureCache(
        _normalized_random(7, seed=22), tuple(f"q{index}" for index in range(7))
    )
    labels = tuple(label for label in classes for _ in range(5))
    context = runtime.NextR1ArmContext("R1", support, query, labels, classes)
    logits = real.frozen_d92_full160_head(context)
    assert logits.dtype == np.float32 and logits.shape == (7, 6)
    assert np.isfinite(logits).all()

    k1_support = runtime.NextR1FeatureCache(
        support.z160[::5], tuple(f"k1-{index}" for index in range(6))
    )
    k1_context = runtime.NextR1ArmContext("R1", k1_support, query, classes, classes)
    with pytest.raises(real.NextR1RealError, match="frozen to K5"):
        real.frozen_d92_full160_head(k1_context)
