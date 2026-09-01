from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cvsrffi.marc_ot_phase1_entry import (
    build_marc_ot_functional_forward,
    run_marc_ot_phase1_bundle,
    validate_marc_ot_phase1_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_payload() -> dict:
    return {
        "schema": "cvs.phase1.marc_ot.bundle.v1",
        "run_id": "marc_ot_phase1_bundle_20260901_r1",
        "seed": 392002,
        "base_checkpoint": "runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth",
        "base_checkpoint_id": "ADV3B02_CORE90_SOFT_E200",
        "wisig_pkl": "Dataset_WigSig/ManySig.pkl",
        "source_receiver_ids": [0, 1, 2, 3, 4, 5, 6],
        "source_days": [0, 1],
        "source_roles": {"L_s": 0.07, "U_s": 0.63, "V": 0.30},
        "wisig": {
            "equalized": 1,
            "out_len": 256,
            "crop_mode": "center",
            "normalize": True,
            "domain": "rx_day",
            "max_samples_per_combo": 0,
        },
        "model": {
            "builder": "dual",
            "num_classes": 6,
            "num_domains": 14,
            "model_size": "M",
            "dataset": "wisig",
            "input_len": 256,
            "sample_rate_hz": 25000000.0,
            "id_feature_key": "feat_joint",
            "dom_feature_key": "feat_imp",
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "domain_enhancer_strength": 0.35,
        },
        "k_choices": [1, 2, 5, 10, 20],
        "training_k": [10],
        "query_per_class": 2,
        "schedule_seed": 713104,
        "expert": {
            "steps": 25,
            "lr": 3e-5,
            "max_rank": 16,
            "trainable_prefixes": [
                "id_backbone.t1.",
                "id_backbone.t2.",
                "id_backbone.t3.",
                "id_backbone.f1.",
                "id_backbone.f2.",
                "id_backbone.f3.",
                "id_backbone.time_projection.",
                "id_backbone.time_proj.",
                "id_backbone.frequency_projection.",
                "id_backbone.f_proj.",
                "id_backbone.freq_stats_proj.",
                "id_backbone.fusion.",
                "id_backbone.time_fuse.",
                "id_backbone.freq_gate.",
                "id_backbone.identity_mapping.",
                "identity_mapping.",
            ],
        },
        "encoder": {"hidden_dim": 64, "lr_min": 1e-4, "lr_max": 1e-2},
        "trainer": {
            "inner_steps": 3,
            "receiver_cvar_fraction": 0.5,
            "receiver_cvar_weight": 1.0,
            "worst_class_guard_weight": 1.0,
        },
        "meta_outer_lr": 1e-4,
        "outer_cycles": 10,
    }


def test_config_rejects_split_validation_roles_wrong_ratios_and_training_k() -> None:
    payload = _valid_payload()
    payload["source_roles"] = {
        "L_s": 0.07,
        "U_s": 0.63,
        "V_cal": 0.15,
        "V_select": 0.15,
    }
    with pytest.raises(ValueError, match="source_roles|V_cal|V_select"):
        validate_marc_ot_phase1_config(payload)

    payload = _valid_payload()
    payload["source_roles"] = {"L_s": 0.08, "U_s": 0.62, "V": 0.30}
    with pytest.raises(ValueError, match="0.07|source_roles"):
        validate_marc_ot_phase1_config(payload)

    payload = _valid_payload()
    payload["training_k"] = [3]
    with pytest.raises(ValueError, match="training_k"):
        validate_marc_ot_phase1_config(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("expert", "lr"), math.nan),
        (("expert", "steps"), 0),
        (("encoder", "lr_max"), math.inf),
        (("trainer", "inner_steps"), 11),
        (("meta_outer_lr",), -1e-4),
        (("outer_cycles",), True),
    ),
)
def test_config_rejects_nonfinite_or_illegal_hyperparameters(path, value) -> None:
    payload = _valid_payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((TypeError, ValueError)):
        validate_marc_ot_phase1_config(payload)


def test_formal_config_is_exact_and_valid() -> None:
    path = ROOT / "configs" / "marc_ot_phase1_bundle_20260901.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert validate_marc_ot_phase1_config(payload) == payload
    assert payload["run_id"] == "marc_ot_phase1_bundle_20260901_r2"
    assert payload["base_checkpoint"] == (
        "/home/szu2070436088/2510044040/CV-SincNet/runs/"
        "phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/"
        "best_joint_safe_ssdg.pth"
    )
    assert payload["wisig_pkl"] == (
        "/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl"
    )


class _ToyIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t1 = nn.Linear(2, 160, bias=False)
        self.register_buffer("scale", torch.tensor([1.5]))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _ToyIdentity()
        self.head = nn.Linear(160, 6, bias=False)

    def forward(self, values, return_aux=False):
        pooled = values.float().mean(dim=-1) if values.ndim == 3 else values.float()
        z_id = self.id_backbone.t1(pooled) * self.id_backbone.scale
        logits = self.head(z_id)
        if return_aux:
            return {
                "tx_logits": logits,
                "z_id": z_id,
                "aux_id": {"t_emb": z_id, "f_emb": z_id * 0.75},
            }
        return logits


def test_functional_forward_overrides_only_canonical_fast_parameters() -> None:
    model = _ToyModel()
    with torch.no_grad():
        model.id_backbone.t1.weight.fill_(0.25)
        model.head.weight.fill_(0.5)
        model.id_backbone.scale.fill_(2.0)
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    forward = build_marc_ot_functional_forward(model, base_state)

    with torch.no_grad():
        model.head.weight.zero_()
        model.id_backbone.scale.zero_()
    fast_weight = torch.full_like(base_state["id_backbone.t1.weight"], 0.75, requires_grad=True)
    values = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    logits = forward({"id_backbone.t1.weight": fast_weight}, values)
    expected_z = torch.full((1, 160), 4.5)
    expected = expected_z @ base_state["head.weight"].T
    assert torch.allclose(logits, expected)
    gradient = torch.autograd.grad(logits.sum(), fast_weight)[0]
    assert torch.count_nonzero(gradient)

    with pytest.raises(ValueError, match="canonical|fast"):
        forward({"head.weight": base_state["head.weight"].clone().requires_grad_(True)}, values)


class _RoleDataset:
    def __init__(self, rows) -> None:
        self.index = list(rows)
        self.capture_block_size = 8

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index[int(index)]
        base = float(row.tx_i + 1) + 0.01 * float(row.sig_i + 1)
        i = torch.linspace(0.0, 1.0, 16) + base
        q = torch.linspace(1.0, 0.0, 16) + 0.5 * base
        return torch.stack((i, q)), int(row.tx_i), 0, {
            "rx_i": int(row.rx_i),
            "day_i": int(row.day_i),
            "eq_i": int(row.eq_i),
            "capture_block_i": int(row.sig_i) // 8,
            "physical_sample_id": (
                f"tx{row.tx_i}|rx{row.rx_i}|day{row.day_i}|eq{row.eq_i}|sig{row.sig_i}"
            ),
        }


def _role_datasets():
    from dataset_wisig import WiSigIndex

    labeled = [
        WiSigIndex(tx, rx, day, 0, sig)
        for tx in range(6)
        for rx in range(7)
        for day in range(2)
        for sig in range(32)
    ]
    unlabeled = [
        WiSigIndex(tx, rx, day, 0, 100)
        for tx in range(6)
        for rx in range(7)
        for day in range(2)
    ]
    validation = [
        WiSigIndex(tx, rx, day, 0, 200)
        for tx in range(6)
        for rx in range(7)
        for day in range(2)
    ]
    return {"L_s": _RoleDataset(labeled), "U_s": _RoleDataset(unlabeled), "V": _RoleDataset(validation)}


def test_injected_entry_runs_real_schedule_and_enforces_source_boundaries(tmp_path: Path) -> None:
    from cvsrffi.marc_ot_source_experts import build_source_expert_bank
    from cvsrffi.meta_episodes import (
        audit_marc_ot_episode_coverage,
        sample_marc_ot_coverage_schedule,
    )

    payload = _valid_payload()
    model = _ToyModel()
    role_datasets = _role_datasets()
    captured = {}

    def source_expert_boundary(current_model, task_batches, config):
        captured["task_batches"] = task_batches
        captured["expert_config"] = config
        return build_source_expert_bank(
            current_model,
            task_batches,
            replace(config, steps=1, max_rank=min(config.max_rank, 4)),
        )

    def bank_training_boundary(**kwargs):
        scheduled = sample_marc_ot_coverage_schedule(
            kwargs["sampler"], seed=kwargs["schedule_seed"]
        )
        audit_marc_ot_episode_coverage(
            scheduled,
            source_receiver_ids=tuple(payload["source_receiver_ids"]),
            require_complete=True,
        )
        selected = tuple(kwargs["training_episode_selector"](scheduled))
        assert len(scheduled) == 55
        assert len(selected) == 11
        assert {episode.k_shot for episode in selected} == {10}
        mask_builder = kwargs["episode_coefficient_mask"]
        assert callable(mask_builder)
        coefficient_dim = sum(entry.effective_rank for entry in kwargs["bank"].entries)
        for episode in selected:
            pseudo_target = str((episode.query_adapt + episode.query_guard)[0].rx_i)
            mask = mask_builder(episode)
            assert tuple(mask.shape) == (coefficient_dim,)
            offset = 0
            for entry in kwargs["bank"].entries:
                segment = mask[offset : offset + entry.effective_rank]
                expected = torch.tensor(
                    [0.0 if key.receiver == pseudo_target else 1.0 for key in kwargs["bank"].task_keys]
                )
                assert torch.equal(segment.cpu(), expected)
                offset += entry.effective_rank
        required = {
            id(parameter)
            for parameter in kwargs["support_encoder"].parameters()
        } | {id(entry.basis) for entry in kwargs["bank"].entries}
        observed = {
            id(parameter)
            for group in kwargs["optimizer"].param_groups
            for parameter in group["params"]
        }
        assert observed == required
        assert not any(entry.task_coefficients.requires_grad for entry in kwargs["bank"].entries)
        assert all(id(parameter) not in observed for parameter in kwargs["support_feature_model"].parameters())
        bundle_path = Path(kwargs["bundle_path"])
        torch.save({"strict_readback_marker": True}, bundle_path)
        assert torch.load(bundle_path, weights_only=True)["strict_readback_marker"] is True
        return SimpleNamespace(
            bundle_path=bundle_path,
            software_coverage={"software_supported_k": (1, 2, 5, 10, 20), "episode_count": 55},
            training_coverage={
                "k_shot": (10,),
                "trained_episode_count": 11,
                "optimizer_step_count": 110,
                "outer_cycles": 10,
            },
            loaded_bundle=SimpleNamespace(bank=kwargs["bank"]),
            pilot_executed=False,
        )

    output_root = tmp_path / "bundle-output"
    summary = run_marc_ot_phase1_bundle(
        SimpleNamespace(config=payload, output_root=output_root, device="cpu"),
        {"tx_list": list(range(6)), "rx_list": list(range(7))},
        injected_context={
            "model": model,
            "role_datasets": role_datasets,
            "build_source_expert_bank": source_expert_boundary,
            "run_bank_training": bank_training_boundary,
        },
    )

    assert summary["software_supported_k"] == [1, 2, 5, 10, 20]
    assert summary["training_coverage_k"] == [10]
    assert summary["trained_episode_count"] == 11
    assert summary["optimizer_step_count"] == 110
    assert summary["outer_cycles"] == 10
    assert summary["task_count"] == len(captured["task_batches"])
    assert summary["bank"]["block_count"] > 0
    assert summary["bank"]["effective_ranks"]
    assert summary["bank"]["coordinate_system"] == "EXACT_TASK_RECEIVER_COLUMNS"
    assert summary["loro_memory_exclusion"] == "M_MINUS_D_FOR_EVERY_TRAINING_EPISODE"
    assert summary["source_role_sizes"] == {"L_s": 2688, "U_s": 84, "V": 84}
    assert summary["pilot_executed"] is False
    assert summary["performance_claim"] == "NOT_EVALUATED"
    assert "accuracy" not in json.dumps(summary).lower()
    assert (output_root / "config_snapshot.json").is_file()
    assert (output_root / "summary.json").is_file()
    assert (output_root / "marc_ot_weight_bundle.pt").is_file()

    for key, task_batch in captured["task_batches"].items():
        refs = task_batch["refs"]
        assert refs
        assert {ref.role for ref in refs} == {"L_s"}
        assert {ref.rx_i for ref in refs} == {int(key.receiver)}
        assert {ref.day_i for ref in refs} == {int(key.day)}
        assert {ref.capture_block_i for ref in refs} == {int(key.capture_block)}
        assert {ref.view for ref in refs} == {key.scene}
        assert len({ref.physical_sample_id for ref in refs}) == len(refs)
        assert set(task_batch["physical_ids"]).isdisjoint(task_batch["excluded_outer_query_ids"])

    before = (output_root / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="output root"):
        run_marc_ot_phase1_bundle(
            SimpleNamespace(config=payload, output_root=output_root, device="cpu"),
            {},
            injected_context={"model": model, "role_datasets": role_datasets},
        )
    assert (output_root / "summary.json").read_bytes() == before


def test_failure_writes_failed_summary_and_reraises(tmp_path: Path) -> None:
    payload = _valid_payload()
    output_root = tmp_path / "failed-output"

    def fail_source_expert(*_args, **_kwargs):
        raise RuntimeError("injected deterministic failure")

    with pytest.raises(RuntimeError, match="deterministic failure"):
        run_marc_ot_phase1_bundle(
            SimpleNamespace(config=payload, output_root=output_root, device="cpu"),
            {"tx_list": list(range(6)), "rx_list": list(range(7))},
            injected_context={
                "model": _ToyModel(),
                "role_datasets": _role_datasets(),
                "build_source_expert_bank": fail_source_expert,
            },
        )
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["error_type"] == "RuntimeError"
    assert "deterministic failure" in summary["error"]
