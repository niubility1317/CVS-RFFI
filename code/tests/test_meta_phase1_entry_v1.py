import json
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import numpy as np
import torch
from torch import nn

from cvsrffi.meta_phase1_entry import (
    _build_refs,
    _build_source_batches,
    _candidate_from_curves,
    _compute_frozen_class_prototypes,
    _evaluate_final_checkpoint_scenarios,
    _episode_batch,
    _sample_episode,
    _source_role_manifest,
    _training_batches_for_step,
    _training_kind_schedule,
    parse_args_for_test,
    run_meta_phase1,
    validate_meta_phase1_config,
)
from cvsrffi.meta_adapter import ResidualMetaAdapter
from cvsrffi.meta_episodes import EpisodeKind, MetaEpisode, MetaSampleRef
from cvsrffi.meta_trainer import AdaptationCurve, AdaptationCurveRow, MetaEpisodeBatch, select_source_checkpoint


def valid_config():
    return {
        "schema": "cvs.phase1.meta_adapter.tri_r4.v1",
        "run_id": "phase1_test_r1",
        "seed": 392002,
        "base_checkpoint": "runs/base/best.pth",
        "wisig_pkl": "Dataset_WigSig/ManySig.pkl",
        "source_receiver_ids": [0, 1, 2, 3, 4, 5, 6],
        "source_split": "tx_rx_day_1_7_2",
        "source_days": [0, 1],
        "clean_test_days": [2, 3],
        "source_roles": {"L_s": 0.07, "U_s": 0.63, "V_cal": 0.15, "V_select": 0.15},
        "adapter": {
            "rank": 4,
            "sites": ["time", "freq", "fusion"],
            "inner_steps": 3,
            "deployment_max_steps": 5,
            "source_diagnostic_max_steps": 10,
        },
        "episode_weights": {
            "Q_SAME_DOMAIN": 0.40,
            "Q_RX_HOLDOUT": 0.20,
            "Q_DAY_CHANNEL_HOLDOUT": 0.15,
            "Q_CLEAN_TO_LEO": 0.15,
            "Q_LEO_CROSS": 0.10,
        },
        "k_choices": [1, 2, 5, 10],
        "meta_batch_size": 4,
        "phase1c_backbone_lr_ratio": 0.05,
        "evaluate_steps": [0, 1, 3, 5, 10],
        "wisig_equalized": 1,
        "wisig_out_len": 256,
        "wisig_domain": "rx_day",
        "wisig_max_day123_per_combo": 0,
        "meta_train_steps": 1,
        "meta_eval_episodes": 2,
        "meta_query_per_class": 2,
        "candidate_plan": [
            {"candidate_id": "P0", "training_mode": "frozen_base", "learn_step_sizes": False},
            {"candidate_id": "P1", "training_mode": "random_adapter", "learn_step_sizes": False},
            {"candidate_id": "P2", "training_mode": "supervised_adapter", "learn_step_sizes": False},
            {"candidate_id": "P3", "training_mode": "fomaml_fixed_lr", "learn_step_sizes": False},
            {"candidate_id": "P4", "training_mode": "fomaml_meta_sgd", "learn_step_sizes": True},
        ],
        "model": {
            "builder": "single",
            "num_classes": 3,
            "dataset": "wisig",
            "input_len": 3,
            "model_size": "S",
            "model_variant": "base",
        },
    }


class _ToyLegacyModel(nn.Module):
    def __init__(self, class_count=3):
        super().__init__()
        self.t_proj = nn.Linear(3, 4)
        self.f_proj = nn.Linear(3, 4)
        self.fuse = nn.Linear(8, 4)
        self.cls_head = nn.Linear(4, class_count)


class _ToyMetaModel(_ToyLegacyModel):
    def __init__(self, class_count=3):
        super().__init__(class_count)
        self.meta_adapter_time = ResidualMetaAdapter(4, rank=4)
        self.meta_adapter_freq = ResidualMetaAdapter(4, rank=4)
        self.meta_adapter_fusion = ResidualMetaAdapter(4, rank=4)

    def forward(self, x, y=None, return_aux=True):
        del y
        t = self.meta_adapter_time(self.t_proj(x))
        f = self.meta_adapter_freq(self.f_proj(x))
        z = self.meta_adapter_fusion(self.fuse(torch.cat((t, f), dim=1)))
        result = {"logits": self.cls_head(z), "feat_cls": z}
        return result if return_aux else {"logits": result["logits"]}


class _ToyIQMetaModel(nn.Module):
    def __init__(self, class_count=4):
        super().__init__()
        self.proj = nn.Linear(2 * 64, 4)
        self.meta_adapter_time = ResidualMetaAdapter(4, rank=4)
        self.meta_adapter_freq = ResidualMetaAdapter(4, rank=4)
        self.meta_adapter_fusion = ResidualMetaAdapter(4, rank=4)
        self.cls_head = nn.Linear(4, class_count)

    def forward(self, x, y=None, return_aux=True):
        del y
        z = self.proj(x.flatten(start_dim=1))
        z = self.meta_adapter_time(z)
        z = self.meta_adapter_freq(z)
        z = self.meta_adapter_fusion(z)
        result = {"logits": self.cls_head(z), "feat_cls": z}
        return result if return_aux else {"logits": result["logits"]}


class _ViewDataset:
    def __init__(self):
        generator = torch.Generator().manual_seed(1234)
        self.x = torch.randn(2, 64, generator=generator)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        index = int(index)
        return (
            self.x.clone(),
            index % 4,
            0,
            {
                "rx_i": 0,
                "day_i": 0,
                "eq_i": 0,
                "capture_block_i": index,
                "physical_sample_id": f"view-physical-{index}",
            },
        )


def _view_ref(index, tx_i, view, role="L_s"):
    return MetaSampleRef(
        dataset_index=index,
        tx_i=tx_i,
        rx_i=0,
        day_i=0,
        eq_i=0,
        capture_block_i=index,
        physical_sample_id=f"view-physical-{index}",
        role=role,
        view=view,
    )


def _leo_view_episode():
    return MetaEpisode(
        kind=EpisodeKind.LEO_CROSS,
        support=(
            _view_ref(0, 0, "clean"),
            _view_ref(1, 1, "leo_clear_weak"),
            _view_ref(2, 2, "leo_low_elev_weak"),
        ),
        query_adapt=(_view_ref(3, 0, "leo_rain_weak"),),
        query_guard=(_view_ref(4, 3, "clean"),),
        adapt_class_ids=frozenset({0, 1, 2}),
        guard_class_ids=frozenset({3}),
        k_shot=1,
        seed=17,
    )


def _curve_row(
    role,
    episode_index,
    step,
    mean_accuracy,
    *,
    clean_step0_accuracy=0.8,
    floor_accuracy=0.7,
    guard_floor_accuracy=0.7,
):
    return AdaptationCurveRow(
        episode_index=episode_index,
        role=role,
        step=step,
        episode_kind=EpisodeKind.LEO_CROSS.value,
        k_shot=1,
        mean_accuracy=mean_accuracy,
        floor_accuracy=floor_accuracy,
        per_class_accuracy=((0, mean_accuracy),),
        adapt_accuracy=mean_accuracy,
        guard_accuracy=mean_accuracy,
        clean_step0_accuracy=clean_step0_accuracy,
        adaptation_delta_pp=0.0,
        held_receiver=(0,),
        held_day=(0,),
        held_channel=(episode_index,),
        leo_scenarios=(role,) if role.startswith("leo_") else (),
        y_adapt_count=1,
        y_guard_count=1,
        adapter_norm=0.0,
        module_step_sizes=(),
        parameter_ratio=0.0,
        state_size_bytes=0,
        latency_ms=1.0,
        guard_floor_accuracy=guard_floor_accuracy,
    )


def _paired_curves(v_cal_delta, v_select_delta):
    baseline = []
    final = []
    for role, episode_index, delta in (
        ("V_cal", 0, v_cal_delta),
        ("V_select", 1, v_select_delta),
    ):
        baseline.extend(
            (
                _curve_row(role, episode_index, 0, 0.8),
                _curve_row(role, episode_index, 3, 0.5),
            )
        )
        final.extend(
            (
                _curve_row(role, episode_index, 0, 0.8),
                _curve_row(role, episode_index, 3, 0.5 + delta),
            )
        )
    return (
        AdaptationCurve(steps=(0, 1, 3, 5, 10), rows=tuple(baseline)),
        AdaptationCurve(steps=(0, 1, 3, 5, 10), rows=tuple(final)),
    )


def _toy_batch(role="L_s", rx_i=0):
    refs = tuple(
        MetaSampleRef(
            dataset_index=i,
            tx_i=i,
            rx_i=rx_i,
            day_i=0,
            eq_i=0,
            capture_block_i=0,
            physical_sample_id=f"{role}-support-{i}",
            role=role,
            view="clean",
        )
        for i in range(2)
    )
    query_adapt = (
        MetaSampleRef(10, 0, rx_i, 1, 0, 1, f"{role}-query-0", role, "clean"),
        MetaSampleRef(11, 1, rx_i, 1, 0, 1, f"{role}-query-1", role, "clean"),
    )
    query_guard = (
        MetaSampleRef(12, 2, rx_i, 1, 0, 1, f"{role}-query-2", role, "clean"),
    )
    episode = MetaEpisode(
        kind=EpisodeKind.SAME_DOMAIN,
        support=refs,
        query_adapt=query_adapt,
        query_guard=query_guard,
        adapt_class_ids=frozenset({0, 1}),
        guard_class_ids=frozenset({2}),
        k_shot=1,
        seed=17,
    )
    return MetaEpisodeBatch(
        episode=episode,
        support_x=torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]),
        support_y=torch.tensor([0, 1], dtype=torch.long),
        query_x=torch.tensor([[0.8, 0.1, 0.2], [0.2, 0.9, -0.1], [-0.4, 0.2, 0.8]]),
        query_y=torch.tensor([0, 1, 2], dtype=torch.long),
        adapt_mask=torch.tensor([True, True, False]),
        guard_mask=torch.tensor([False, False, True]),
        frozen_prototypes=torch.zeros(3, 4),
    )


def _toy_args(config_path, output_root, base_path, wisig_path, batch_factory=None):
    args = parse_args_for_test(["--use_cvs_meta_adapter"])
    args.meta_config = str(config_path)
    args.meta_output_root = str(output_root)
    args.wisig_train_rxs = "0,1,2,3,4,5,6"
    args.wisig_train_days = "0,1"
    args.wisig_out_len = 256
    args.wisig_equalized = 1
    args.wisig_domain = "rx_day"
    args.base_checkpoint = str(base_path)
    args.wisig_pkl = str(wisig_path)
    if batch_factory is not None:
        args.meta_model_factory = lambda config, ds_w, device: _ToyMetaModel(3).to(device)
        args.meta_episode_batch_factory = batch_factory
    return args


def test_meta_adapter_cli_defaults_are_v1_locked():
    args = parse_args_for_test(["--use_cvs_meta_adapter"])
    assert args.use_cvs_meta_adapter is True
    assert args.meta_adapter_rank == 4
    assert args.meta_adapter_sites == "time,freq,fusion"
    assert args.meta_inner_steps == 3
    assert args.meta_inner_max_steps == 5


def test_phase1_config_allows_fusion_only_small_layer_profile():
    config = valid_config()
    config["schema"] = "cvs.phase1.meta_adapter.r4.v1"
    config["adapter"]["sites"] = ["fusion"]

    validated = validate_meta_phase1_config(config)

    assert validated["schema"] == "cvs.phase1.meta_adapter.r4.v1"
    assert validated["adapter"]["sites"] == ["fusion"]


def test_phase1_config_allows_registered_fusion_only_rank8_profile():
    config = valid_config()
    config["schema"] = "cvs.phase1.meta_adapter.r4.v1"
    config["adapter"]["sites"] = ["fusion"]
    config["adapter"]["rank"] = 8

    validated = validate_meta_phase1_config(config)

    assert validated["adapter"]["rank"] == 8
    assert validated["adapter"]["sites"] == ["fusion"]


def test_phase1_config_rejects_rank8_outside_fusion_only_profile():
    config = valid_config()
    config["schema"] = "cvs.phase1.meta_adapter.r4.v1"
    config["adapter"]["rank"] = 8

    with pytest.raises(ValueError, match="registered rank/site profile"):
        validate_meta_phase1_config(config)


def test_legacy_tri_schema_rejects_fusion_only_profile():
    config = valid_config()
    config["adapter"]["sites"] = ["fusion"]

    with pytest.raises(ValueError, match="tri_r4 schema"):
        validate_meta_phase1_config(config)


@pytest.mark.parametrize(
    "sites",
    [[], ["time"], ["freq"], ["time", "fusion"], ["fusion", "time", "freq"]],
)
def test_phase1_config_rejects_unregistered_adapter_site_profiles(sites):
    config = valid_config()
    config["adapter"]["sites"] = sites

    with pytest.raises(ValueError, match="registered profiles"):
        validate_meta_phase1_config(config)


def test_phase1_entry_rejects_noncanonical_source_ratios():
    config = valid_config()
    config["source_roles"]["L_s"] = 0.10
    with pytest.raises(ValueError, match=r"0\.07"):
        validate_meta_phase1_config(config)


def test_phase1_config_requires_explicit_source_receiver_ids():
    config = valid_config()
    del config["source_receiver_ids"]
    with pytest.raises(ValueError, match="source_receiver_ids"):
        validate_meta_phase1_config(config)


def test_training_kind_schedule_preserves_all_declared_query_domains_and_weights():
    config = validate_meta_phase1_config(valid_config())

    schedule = _training_kind_schedule(config["episode_weights"], seed=config["seed"])

    counts = {kind: schedule.count(kind) for kind in EpisodeKind}
    assert len(schedule) == 20
    assert counts == {
        EpisodeKind.SAME_DOMAIN: 8,
        EpisodeKind.RX_HOLDOUT: 4,
        EpisodeKind.DAY_CHANNEL_HOLDOUT: 3,
        EpisodeKind.CLEAN_TO_LEO: 3,
        EpisodeKind.LEO_CROSS: 2,
    }


def test_training_batch_rotation_consumes_entire_weighted_pool_before_reuse():
    pool = list(range(20))

    first_cycle = [
        item
        for step in range(5)
        for item in _training_batches_for_step(pool, train_step=step, meta_batch_size=4)
    ]

    assert sorted(first_cycle) == pool
    assert _training_batches_for_step(pool, train_step=5, meta_batch_size=4) == pool[:4]


def test_sample_episode_can_require_a_specific_cross_domain_kind():
    desired = types.SimpleNamespace(
        kind=EpisodeKind.LEO_CROSS,
        guard_class_ids=frozenset({5}),
        query_guard=(object(),),
        query_adapt=(),
    )
    wrong = types.SimpleNamespace(
        kind=EpisodeKind.SAME_DOMAIN,
        guard_class_ids=frozenset({5}),
        query_guard=(object(),),
        query_adapt=(),
    )

    class FakeSampler:
        def __init__(self):
            self.seeds = []

        def sample(self, seed):
            self.seeds.append(seed)
            return desired if len(self.seeds) == 3 else wrong

    sampler = FakeSampler()
    episode = _sample_episode(
        sampler,
        seed=100,
        required_kind=EpisodeKind.LEO_CROSS,
    )

    assert episode is desired
    assert sampler.seeds == [100, 101, 102]


def test_real_source_batch_builder_materializes_weighted_cross_domain_pool(monkeypatch):
    import cvsrffi.meta_episodes as episode_module
    import cvsrffi.meta_phase1_entry as entry_module

    kinds = tuple(EpisodeKind)

    class FakeSampler:
        def __init__(self, refs, config):
            del refs, config

        def sample(self, seed):
            row = types.SimpleNamespace(view="clean", role="L_s")
            return types.SimpleNamespace(
                kind=kinds[int(seed) % len(kinds)],
                guard_class_ids=frozenset({5}),
                query_guard=(row,),
                query_adapt=(row,),
                support=(row,),
            )

    monkeypatch.setattr(episode_module, "HierarchicalMetaEpisodeSampler", FakeSampler)
    monkeypatch.setattr(
        entry_module,
        "_build_refs",
        lambda dataset, role: ([types.SimpleNamespace(role=role)], dataset),
    )
    monkeypatch.setattr(
        entry_module,
        "_episode_batch",
        lambda episode, dataset, **kwargs: episode,
    )
    config = validate_meta_phase1_config(valid_config())

    train_pool, eval_batches = _build_source_batches(
        {"L_s": object(), "V_cal": object(), "V_select": object()},
        config,
        _ToyMetaModel(3),
        torch.device("cpu"),
    )

    counts = {kind: sum(batch.kind == kind for batch in train_pool) for kind in kinds}
    assert len(train_pool) == 20
    assert counts == {
        EpisodeKind.SAME_DOMAIN: 8,
        EpisodeKind.RX_HOLDOUT: 4,
        EpisodeKind.DAY_CHANNEL_HOLDOUT: 3,
        EpisodeKind.CLEAN_TO_LEO: 3,
        EpisodeKind.LEO_CROSS: 2,
    }
    assert len(eval_batches) == config["meta_eval_episodes"]


def test_source_manifest_builds_declared_clean_test_disjoint_from_all_selection_roles(
    monkeypatch
):
    class FakeCompactDataset:
        def __init__(self, ds_w, *, day_keep, rx_keep, **kwargs):
            del ds_w, kwargs
            self.rows = []
            for day in day_keep:
                for rx in rx_keep:
                    for index in range(5):
                        self.rows.append((int(day), int(rx), index))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            day, rx, sample = self.rows[int(index)]
            return (
                torch.ones(2, 64) * (sample + 1),
                sample % 3,
                0,
                {
                    "rx_i": rx,
                    "day_i": day,
                    "eq_i": 0,
                    "capture_block_i": sample,
                    "physical_sample_id": f"day{day}-rx{rx}-sample{sample}",
                },
            )

    class FakeSubsetDataset:
        def __init__(self, base, indices, split_source):
            del split_source
            self.base = base
            self.indices = list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            return self.base[self.indices[int(index)]]

    dataset_module = types.ModuleType("dataset_wisig")
    dataset_module.WiSigCompactDataset = FakeCompactDataset
    dataset_module.WiSigSubsetDataset = FakeSubsetDataset
    split_module = types.ModuleType("SSDG.train_ssdg")

    def split_roles(base, **kwargs):
        del kwargs
        indices = list(range(len(base)))
        return indices[0::4], indices[1::4], indices[2::4], indices[3::4]

    split_module.split_tx_rx_day_1_7_2_roles = split_roles
    monkeypatch.setitem(sys.modules, "dataset_wisig", dataset_module)
    monkeypatch.setitem(sys.modules, "SSDG.train_ssdg", split_module)
    config = validate_meta_phase1_config(valid_config())
    args = _toy_args("unused.json", "unused", "base", "wisig")
    args.wisig_test_days = "2,3"
    manifest = _source_role_manifest(
        {"data": object(), "rx_list": list(range(12))}, config, args
    )
    role_ids = {
        item[-1]["physical_sample_id"]
        for dataset in manifest["role_datasets"].values()
        for item in (dataset[index] for index in range(len(dataset)))
    }
    test_ids = {
        manifest["clean_test_dataset"][index][-1]["physical_sample_id"]
        for index in range(len(manifest["clean_test_dataset"]))
    }
    assert manifest["clean_test_days"] == (2, 3)
    assert manifest["clean_test_physical_disjoint"] is True
    assert role_ids.isdisjoint(test_ids)


def test_source_manifest_checks_physical_ids_without_decoding_iq(monkeypatch):
    class FakeCompactDataset:
        def __init__(self, ds_w, *, day_keep, rx_keep, **kwargs):
            del ds_w, kwargs
            self.index = [
                types.SimpleNamespace(
                    tx_i=sample % 3,
                    rx_i=int(rx),
                    day_i=int(day),
                    eq_i=0,
                    sig_i=sample,
                )
                for day in day_keep
                for rx in rx_keep
                for sample in range(5)
            ]

        def __len__(self):
            return len(self.index)

        def __getitem__(self, index):
            raise AssertionError(f"IQ decoding is forbidden during manifest scan: {index}")

    class FakeSubsetDataset:
        def __init__(self, base, indices, split_source):
            del split_source
            self.base = base
            self.selected = np.asarray(indices, dtype=np.int64)
            self.index = [base.index[int(index)] for index in self.selected.tolist()]

        def __len__(self):
            return len(self.index)

        def __getitem__(self, index):
            raise AssertionError(f"IQ decoding is forbidden during manifest scan: {index}")

    def physical_id(item):
        return (
            f"tx{item.tx_i}|rx{item.rx_i}|day{item.day_i}|"
            f"eq{item.eq_i}|sig{item.sig_i}"
        )

    dataset_module = types.ModuleType("dataset_wisig")
    dataset_module.WiSigCompactDataset = FakeCompactDataset
    dataset_module.WiSigSubsetDataset = FakeSubsetDataset
    dataset_module.wisig_physical_sample_id = physical_id
    split_module = types.ModuleType("SSDG.train_ssdg")

    def split_roles(base, **kwargs):
        del kwargs
        indices = list(range(len(base)))
        return indices[0::4], indices[1::4], indices[2::4], indices[3::4]

    split_module.split_tx_rx_day_1_7_2_roles = split_roles
    monkeypatch.setitem(sys.modules, "dataset_wisig", dataset_module)
    monkeypatch.setitem(sys.modules, "SSDG.train_ssdg", split_module)
    config = validate_meta_phase1_config(valid_config())
    args = _toy_args("unused.json", "unused", "base", "wisig")
    args.wisig_test_days = "2,3"

    manifest = _source_role_manifest(
        {"data": object(), "rx_list": list(range(12))}, config, args
    )

    assert manifest["clean_test_physical_disjoint"] is True
    assert manifest["clean_test_size"] > 0


def test_episode_refs_use_wisig_index_without_decoding_iq():
    class IndexOnlyDataset:
        capture_block_size = 4
        index = [
            types.SimpleNamespace(tx_i=0, rx_i=2, day_i=1, eq_i=0, sig_i=3),
            types.SimpleNamespace(tx_i=1, rx_i=4, day_i=0, eq_i=0, sig_i=9),
        ]

        def __len__(self):
            return len(self.index)

        def __getitem__(self, index):
            raise AssertionError(f"IQ decoding is forbidden during ref build: {index}")

    dataset = IndexOnlyDataset()
    refs, returned_dataset = _build_refs(dataset, "L_s")

    assert returned_dataset is dataset
    assert len(refs) == len(dataset) * 4
    assert {(ref.dataset_index, ref.tx_i) for ref in refs} == {(0, 0), (1, 1)}
    assert {ref.physical_sample_id for ref in refs} == {
        "tx0|rx2|day1|eq0|sig3",
        "tx1|rx4|day0|eq0|sig9",
    }
    assert {ref.capture_block_i for ref in refs if ref.dataset_index == 0} == {0}
    assert {ref.capture_block_i for ref in refs if ref.dataset_index == 1} == {2}


def test_final_scenario_evaluation_uses_declared_clean_test_not_v_select():
    dataset = _ViewDataset()
    result = _evaluate_final_checkpoint_scenarios(
        _ToyIQMetaModel(4),
        source_manifest={
            "available": True,
            "clean_test_dataset": dataset,
            "clean_test_days": (2, 3),
            "clean_test_physical_disjoint": True,
        },
        eval_batches=[_toy_batch("V_select")],
        device=torch.device("cpu"),
        seed=392002,
    )
    assert result["split"] == "declared_clean_test"
    assert result["test_days"] == [2, 3]
    assert result["physical_disjoint_from_phase1_roles"] is True
    assert result["evidence_origin"] == "declared_clean_test_source_iq"


def test_final_scenario_evaluation_streams_without_full_split_stack(monkeypatch):
    import cvsrffi.meta_phase1_entry as sut

    class LargeViewDataset(_ViewDataset):
        def __len__(self):
            return 129

    original_stack = sut.torch.stack

    def bounded_stack(values, *args, **kwargs):
        rows = tuple(values)
        assert len(rows) <= 128, "final evaluation must not stack the full split"
        return original_stack(rows, *args, **kwargs)

    monkeypatch.setattr(sut.torch, "stack", bounded_stack)
    monkeypatch.setattr(
        sut,
        "_materialize_ref_view",
        lambda x, ref, *, view_seed: x,
    )
    result = sut._evaluate_final_checkpoint_scenarios(
        _ToyIQMetaModel(4),
        source_manifest={
            "available": True,
            "clean_test_dataset": LargeViewDataset(),
            "clean_test_days": (2, 3),
            "clean_test_physical_disjoint": True,
        },
        eval_batches=[_toy_batch("V_select")],
        device=torch.device("cpu"),
        seed=392002,
    )

    assert set(result["scenarios"]) == {
        "clean",
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    }
    assert {row["count"] for row in result["scenarios"].values()} == {129}


def test_streamed_final_scenario_metrics_match_materialized_reference():
    import cvsrffi.meta_phase1_entry as sut

    dataset = _ViewDataset()
    model = _ToyIQMetaModel(4)
    model.eval()
    refs, _ = sut._build_refs(dataset, "declared_clean_test")
    expected = {}
    for view in sut._SOURCE_META_VIEWS:
        rows = [ref for ref in refs if ref.view == view]
        values = []
        labels = []
        for ref in rows:
            x, y, _metadata = sut._dataset_item(dataset, ref.dataset_index)
            values.append(sut._materialize_ref_view(x, ref, view_seed=392002))
            labels.append(y)
        expected[view] = sut._scenario_accuracy(
            model,
            torch.stack(values),
            torch.tensor(labels, dtype=torch.long),
        )

    actual = sut._evaluate_final_checkpoint_scenarios(
        model,
        source_manifest={
            "available": True,
            "clean_test_dataset": dataset,
            "clean_test_days": (2, 3),
            "clean_test_physical_disjoint": True,
        },
        eval_batches=[_toy_batch("V_select")],
        device=torch.device("cpu"),
        seed=392002,
    )

    assert actual["scenarios"] == expected


@pytest.mark.parametrize(
    "field",
    ("wisig_equalized", "wisig_out_len", "wisig_domain", "wisig_max_day123_per_combo"),
)
def test_phase1_config_requires_frozen_wisig_view_fields(field):
    config = valid_config()
    del config[field]
    with pytest.raises(ValueError, match=field):
        validate_meta_phase1_config(config)


def test_phase1_entry_rejects_wisig_cli_drift_before_input_loading(tmp_path):
    config = valid_config()
    config_path = tmp_path / "meta.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = _toy_args(config_path, tmp_path / "run-root", tmp_path / "base.pth", tmp_path / "ManySig.pkl")
    args.wisig_out_len = 512
    with pytest.raises(ValueError, match="wisig_out_len"):
        run_meta_phase1(args, {"rx_list": list(range(7)), "tx_list": ["a", "b", "c"]})


def test_phase1_config_rejects_target_receiver_fields():
    config = valid_config()
    config["target_receiver_ids"] = [7]
    with pytest.raises(ValueError, match="target receiver"):
        validate_meta_phase1_config(config)


def test_launcher_dry_run_does_not_create_output_root(tmp_path, capsys):
    config = valid_config()
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    base_path.write_bytes(b"checkpoint")
    wisig_path.write_bytes(b"wisig")
    config["base_checkpoint"] = str(base_path)
    config["wisig_pkl"] = str(wisig_path)
    config_path = tmp_path / "meta.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_root = tmp_path / "run-root"

    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py"
    spec = importlib.util.spec_from_file_location("meta_phase1_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(launcher)
    main = launcher.main

    main(["--config", str(config_path), "--output-root", str(output_root), "--dry-run"])
    captured = capsys.readouterr().out
    assert "phase1_test_r1" in captured
    assert str(output_root) in captured
    assert "--seed 392002" in captured
    assert all(token in captured for token in ("P1", "P2", "P3", "P4"))
    assert not output_root.exists()

    output_root.mkdir()
    with pytest.raises(FileExistsError):
        main(["--config", str(config_path), "--output-root", str(output_root), "--dry-run"])


def test_launcher_plan_has_unique_immutable_candidate_outputs(tmp_path):
    config = valid_config()
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    base_path.write_bytes(b"checkpoint")
    wisig_path.write_bytes(b"wisig")
    config["base_checkpoint"] = str(base_path)
    config["wisig_pkl"] = str(wisig_path)
    config_path = tmp_path / "meta.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py"
    spec = importlib.util.spec_from_file_location("meta_phase1_launcher_plan", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(launcher)
    plan = launcher.build_launch_plan(config_path, output_root=tmp_path / "matrix")
    candidates = plan["candidate_plans"]
    assert [item["candidate_id"] for item in candidates] == ["P1", "P2", "P3", "P4"]
    assert len({item["output_root"] for item in candidates}) == 4
    assert all(Path(item["output_root"]).parent == tmp_path / "matrix" for item in candidates)


def test_launcher_explicit_input_overrides_release_relative_paths(tmp_path, capsys):
    config = valid_config()
    config["base_checkpoint"] = "runs/base/best.pth"
    config["wisig_pkl"] = "Dataset_WigSig/ManySig.pkl"
    config_path = tmp_path / "release" / "configs" / "meta.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    external_checkpoint = tmp_path / "project-root" / "runs" / "base" / "best.pth"
    external_wisig = tmp_path / "project-root" / "Dataset_WigSig" / "ManySig.pkl"
    external_checkpoint.parent.mkdir(parents=True)
    external_wisig.parent.mkdir(parents=True)
    external_checkpoint.write_bytes(b"checkpoint")
    external_wisig.write_bytes(b"wisig")

    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py"
    spec = importlib.util.spec_from_file_location("meta_phase1_launcher_overrides", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(launcher)
    output_root = tmp_path / "matrix"
    assert launcher.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--base-checkpoint",
            str(external_checkpoint),
            "--wisig-pkl",
            str(external_wisig),
            "--dry-run",
        ]
    ) == 0

    captured = capsys.readouterr().out
    assert f"base_checkpoint={external_checkpoint.resolve()}" in captured
    assert f"wisig_pkl={external_wisig.resolve()}" in captured
    assert captured.count(str(external_checkpoint.resolve())) == 5
    assert captured.count(str(external_wisig.resolve())) == 5
    assert not output_root.exists()


def test_launcher_runs_all_candidates_and_selects_across_matrix_after_scientific_failure(
    tmp_path, monkeypatch
):
    config = valid_config()
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    base_path.write_bytes(b"checkpoint")
    wisig_path.write_bytes(b"wisig")
    config["base_checkpoint"] = str(base_path)
    config["wisig_pkl"] = str(wisig_path)
    config_path = tmp_path / "meta.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py"
    spec = importlib.util.spec_from_file_location("meta_phase1_launcher_matrix", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(launcher)
    launched = []

    def fake_run(command, **kwargs):
        del kwargs
        output = Path(command[command.index("--meta_output_root") + 1])
        candidate_id = output.name
        launched.append(candidate_id)
        output.mkdir(parents=True)
        eligible = candidate_id != "P2"
        payload = {
            "candidate_id": candidate_id,
            "scientific_verdict": (
                "SOURCE_SELECTION_ELIGIBLE" if eligible else "SCIENTIFIC_FAILURE_NO_PROMOTION"
            ),
            "candidate_result": None if not eligible else {
                "worst_a3_delta_pp": {"P1": 0.1, "P3": 0.5, "P4": 0.7}[candidate_id],
                "parameter_count": 10,
                "latency_ms": 1.0,
            },
        }
        (output / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    root = tmp_path / "matrix-run"
    assert launcher.main(["--config", str(config_path), "--output-root", str(root)]) == 0
    assert launched == ["P1", "P2", "P3", "P4"]
    matrix = json.loads((root / "candidate_matrix_summary.json").read_text(encoding="utf-8"))
    assert matrix["selected_candidate_id"] == "P4"
    assert matrix["scientific_verdict"] == "SOURCE_SELECTION_ELIGIBLE"


def test_launcher_dry_run_rejects_missing_inputs_before_run_root(tmp_path):
    config = valid_config()
    config_path = tmp_path / "meta.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_root = tmp_path / "run-root"
    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py"
    spec = importlib.util.spec_from_file_location("meta_phase1_launcher_missing", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(launcher)
    with pytest.raises(FileNotFoundError, match="base_checkpoint|wisig"):
        launcher.main(["--config", str(config_path), "--output-root", str(output_root), "--dry-run"])
    assert not output_root.exists()


def test_meta_phase1_non_dry_run_loads_checkpoint_trains_curves_selects_and_writes_artifacts(tmp_path):
    config = valid_config()
    config_path = tmp_path / "meta.json"
    config["base_checkpoint"] = "base.pth"
    config["wisig_pkl"] = "ManySig.pkl"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    wisig_path.write_bytes(b"fixture")
    torch.save({"model": _ToyLegacyModel(3).state_dict()}, base_path)

    def batches(config, ds_w, model):
        del config, ds_w, model
        return {"train": [_toy_batch("L_s")] * 4, "eval": [_toy_batch("V_cal"), _toy_batch("V_select")]}

    output_root = tmp_path / "run-root"
    args = _toy_args(config_path, output_root, base_path, wisig_path, batches)
    result = run_meta_phase1(args, {"rx_list": list(range(7)), "tx_list": ["a", "b", "c"]})
    assert result["status"] == "ARTIFACTS_COMPLETE"
    assert result["task7_outer_steps"] == 1
    for name in ("logs.jsonl", "metrics.csv", "selected_meta_bundle.pt", "run_summary.json", "config_snapshot.json"):
        artifact = output_root / name
        assert artifact.is_file() and artifact.stat().st_size > 0
    assert (output_root / "source_adaptation_curve.json").is_file()
    assert json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))["status"] == "ARTIFACTS_COMPLETE"


def test_meta_phase1_prototype_artifact_avoids_torch_numpy_abi_bridge(tmp_path, monkeypatch):
    config = valid_config()
    config_path = tmp_path / "meta.json"
    config["base_checkpoint"] = "base.pth"
    config["wisig_pkl"] = "ManySig.pkl"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    wisig_path.write_bytes(b"fixture")
    torch.save({"model": _ToyLegacyModel(3).state_dict()}, base_path)

    def batches(config, ds_w, model):
        del config, ds_w, model
        return {"train": [_toy_batch("L_s")] * 4, "eval": [_toy_batch("V_cal"), _toy_batch("V_select")]}

    def reject_incompatible_numpy_bridge(self):
        del self
        raise TypeError("simulated Torch/NumPy ndarray identity mismatch")

    monkeypatch.setattr(torch.Tensor, "numpy", reject_incompatible_numpy_bridge)
    output_root = tmp_path / "run-root"
    result = run_meta_phase1(
        _toy_args(config_path, output_root, base_path, wisig_path, batches),
        {"rx_list": list(range(7)), "tx_list": ["a", "b", "c"]},
    )

    assert result["status"] == "ARTIFACTS_COMPLETE"
    with np.load(output_root / "frozen_prototypes.npz", allow_pickle=False) as archive:
        assert type(archive["prototypes"]) is np.ndarray
        assert archive["prototypes"].dtype == np.float32
        assert archive["class_ids"].dtype == np.int64


def test_meta_phase1_uses_train_cli_inputs_after_release_relocation(tmp_path):
    config = valid_config()
    config["base_checkpoint"] = "runs/base/best.pth"
    config["wisig_pkl"] = "Dataset_WigSig/ManySig.pkl"
    config_path = tmp_path / "release" / "configs" / "meta.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base_path = tmp_path / "project-root" / "runs" / "base" / "best.pth"
    wisig_path = tmp_path / "project-root" / "Dataset_WigSig" / "ManySig.pkl"
    base_path.parent.mkdir(parents=True)
    wisig_path.parent.mkdir(parents=True)
    wisig_path.write_bytes(b"fixture")
    torch.save({"model": _ToyLegacyModel(3).state_dict()}, base_path)

    def batches(config, ds_w, model):
        del config, ds_w, model
        return {"train": [_toy_batch("L_s")] * 4, "eval": [_toy_batch("V_cal"), _toy_batch("V_select")]}

    output_root = tmp_path / "run-root"
    args = _toy_args(config_path, output_root, base_path, wisig_path, batches)
    args.init_checkpoint = str(base_path)
    result = run_meta_phase1(args, {"rx_list": list(range(7)), "tx_list": ["a", "b", "c"]})

    assert result["status"] == "ARTIFACTS_COMPLETE"
    assert (output_root / "selected_meta_bundle.pt").is_file()


def test_meta_phase1_training_exception_keeps_failed_diagnostics_without_completion(tmp_path):
    config = valid_config()
    config["base_checkpoint"] = "base.pth"
    config["wisig_pkl"] = "ManySig.pkl"
    config_path = tmp_path / "meta.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    wisig_path.write_bytes(b"fixture")
    torch.save({"model": _ToyLegacyModel(3).state_dict()}, base_path)

    def failing_batches(config, ds_w, model):
        del config, ds_w, model
        raise RuntimeError("controlled outer-step failure")

    output_root = tmp_path / "failed-root"
    args = _toy_args(config_path, output_root, base_path, wisig_path, failing_batches)
    with pytest.raises(RuntimeError, match="controlled outer-step failure"):
        run_meta_phase1(args, {"rx_list": list(range(7)), "tx_list": ["a", "b", "c"]})
    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert not (output_root / "COMPLETED").exists()
    assert not (output_root / "selected_meta_bundle.pt").exists()


def test_meta_phase1_same_config_replays_seeded_model_episode_and_selection(tmp_path):
    config = valid_config()
    config["base_checkpoint"] = "base.pth"
    config["wisig_pkl"] = "ManySig.pkl"
    config_path = tmp_path / "meta.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    wisig_path.write_bytes(b"fixture")
    torch.save({"model": _ToyLegacyModel(3).state_dict()}, base_path)

    def batches(config, ds_w, model):
        del config, ds_w, model
        return {"train": [_toy_batch("L_s")] * 4, "eval": [_toy_batch("V_cal"), _toy_batch("V_select")]}

    output_a = tmp_path / "run-a"
    output_b = tmp_path / "run-b"
    args_a = _toy_args(config_path, output_a, base_path, wisig_path, batches)
    args_b = _toy_args(config_path, output_b, base_path, wisig_path, batches)
    ds_w = {"rx_list": list(range(7)), "tx_list": ["a", "b", "c"]}
    result_a = run_meta_phase1(args_a, ds_w)
    result_b = run_meta_phase1(args_b, ds_w)

    selected_a = dict(result_a["selected_candidate"])
    selected_b = dict(result_b["selected_candidate"])
    selected_a.pop("latency_ms", None)
    selected_b.pop("latency_ms", None)
    assert selected_a == selected_b
    assert (output_a / "metrics.csv").read_text(encoding="utf-8") == (output_b / "metrics.csv").read_text(encoding="utf-8")
    assert (output_a / "logs.jsonl").read_text(encoding="utf-8") == (output_b / "logs.jsonl").read_text(encoding="utf-8")
    curve_a = json.loads((output_a / "source_adaptation_curve.json").read_text(encoding="utf-8"))
    curve_b = json.loads((output_b / "source_adaptation_curve.json").read_text(encoding="utf-8"))
    for curve in (curve_a, curve_b):
        for section in ("baseline", "final", "v_calibration", "v_select"):
            payload = curve[section]
            payloads = (
                (value for value in payload.values() if isinstance(value, dict) and "rows" in value)
                if section in ("v_calibration", "v_select")
                else (payload,)
            )
            for item in payloads:
                for row in item["rows"]:
                    row.pop("latency_ms", None)
        curve["selected_candidate"].pop("latency_ms", None)
    assert curve_a == curve_b
    bundle_a = torch.load(output_a / "selected_meta_bundle.pt", map_location="cpu", weights_only=True)
    bundle_b = torch.load(output_b / "selected_meta_bundle.pt", map_location="cpu", weights_only=True)
    assert bundle_a["model_state"].keys() == bundle_b["model_state"].keys()
    assert all(torch.equal(bundle_a["model_state"][key], bundle_b["model_state"][key]) for key in bundle_a["model_state"])


def test_episode_batch_materializes_core90_leo_views_with_deterministic_contract(monkeypatch):
    from cvsrffi import eval as eval_module

    original = eval_module.apply_sat_channel_for_scenario
    calls = []

    def spy(x, scenario, args, *, gen=None, return_meta=False):
        calls.append((x.detach().clone(), str(scenario), args, return_meta))
        return original(x, scenario, args, gen=gen, return_meta=return_meta)

    monkeypatch.setattr(eval_module, "apply_sat_channel_for_scenario", spy)
    dataset = _ViewDataset()
    episode = _leo_view_episode()
    model = _ToyIQMetaModel(4)
    batch_a = _episode_batch(
        episode,
        dataset,
        model=model,
        num_classes=4,
        device=torch.device("cpu"),
        view_seed=392002,
    )
    batch_b = _episode_batch(
        episode,
        dataset,
        model=model,
        num_classes=4,
        device=torch.device("cpu"),
        view_seed=392002,
    )

    assert {item[1] for item in calls} == {
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    }
    assert all(item[0].shape == (1, 2, 64) for item in calls)
    assert all(item[3] is True for item in calls)
    assert all(item[2].sat_fs_hz == 25e6 and item[2].sat_fc_hz == 2.462e9 for item in calls)
    assert batch_a.support_x.shape == (3, 2, 64)
    assert batch_a.query_x.shape == (2, 2, 64)
    assert batch_a.support_x.dtype is torch.float32
    assert batch_a.query_x.dtype is torch.float32
    assert torch.isfinite(batch_a.support_x).all()
    assert torch.isfinite(batch_a.query_x).all()
    assert torch.equal(batch_a.support_x, batch_b.support_x)
    assert torch.equal(batch_a.query_x, batch_b.query_x)
    assert torch.equal(batch_a.support_x[0], dataset[0][0])
    assert not torch.equal(batch_a.support_x[1], dataset[1][0])
    assert not torch.equal(batch_a.support_x[2], dataset[2][0])
    assert not torch.equal(batch_a.query_x[0], dataset[3][0])
    assert not torch.equal(batch_a.support_x[1], batch_a.support_x[2])
    assert not torch.equal(batch_a.support_x[1], batch_a.query_x[0])


def test_candidate_selection_uses_v_select_holdouts_only():
    model = _ToyMetaModel(3)
    baseline_a, final_a = _paired_curves(v_cal_delta=0.9, v_select_delta=0.1)
    baseline_b, final_b = _paired_curves(v_cal_delta=-0.9, v_select_delta=0.1)
    candidate_a = _candidate_from_curves(baseline_a, final_a, candidate_id="A", model=model)
    candidate_b = _candidate_from_curves(baseline_b, final_b, candidate_id="A", model=model)
    assert candidate_a.derived_worst_a3_delta_pp == candidate_b.derived_worst_a3_delta_pp
    assert candidate_a.source_holdouts == candidate_b.source_holdouts

    baseline_c, final_c = _paired_curves(v_cal_delta=0.9, v_select_delta=0.2)
    candidate_c = _candidate_from_curves(baseline_c, final_c, candidate_id="A", model=model)
    assert candidate_c.derived_worst_a3_delta_pp != candidate_a.derived_worst_a3_delta_pp

    baseline_d, final_d = _paired_curves(v_cal_delta=-0.9, v_select_delta=0.1)
    candidate_d = _candidate_from_curves(baseline_d, final_d, candidate_id="B", model=model)
    assert select_source_checkpoint([candidate_c, candidate_d]).candidate_id == "A"


def test_candidate_guard_gate_uses_y_guard_per_class_floor_not_all_query_floor():
    model = _ToyMetaModel(3)
    baseline = AdaptationCurve(
        steps=(0, 1, 3, 5, 10),
        rows=(
            _curve_row("V_select", 0, 0, 0.8, floor_accuracy=0.1, guard_floor_accuracy=0.4),
            _curve_row("V_select", 0, 3, 0.7, floor_accuracy=0.1, guard_floor_accuracy=0.4),
        ),
    )
    final = AdaptationCurve(
        steps=(0, 1, 3, 5, 10),
        rows=(
            _curve_row("V_select", 0, 0, 0.8, floor_accuracy=0.9, guard_floor_accuracy=0.3),
            _curve_row("V_select", 0, 3, 0.8, floor_accuracy=0.9, guard_floor_accuracy=0.3),
        ),
    )
    candidate = _candidate_from_curves(baseline, final, candidate_id="guard", model=model)
    assert candidate.guard_floor_delta_pp == pytest.approx(-10.0)
    with pytest.raises(ValueError, match="eligible"):
        select_source_checkpoint([candidate])


def test_candidate_rejects_missing_clean_or_guard_evidence():
    model = _ToyMetaModel(3)
    baseline = AdaptationCurve(
        steps=(0, 1, 3, 5, 10),
        rows=(_curve_row("V_select", 0, 3, 0.5),),
    )
    final = AdaptationCurve(
        steps=(0, 1, 3, 5, 10),
        rows=(_curve_row("V_select", 0, 3, 0.6),),
    )
    with pytest.raises(ValueError, match="clean.*guard|guard.*clean"):
        _candidate_from_curves(baseline, final, candidate_id="missing", model=model)


def test_phase1_prototypes_are_real_nonzero_source_class_means():
    model = _ToyMetaModel(3)
    prototypes = _compute_frozen_class_prototypes(model, [_toy_batch("L_s")], class_count=3)
    assert prototypes.shape == (3, 4)
    assert torch.isfinite(prototypes).all()
    assert torch.linalg.vector_norm(prototypes, dim=1).gt(0).all()


def test_phase1_prototypes_require_source_samples_for_every_registered_class():
    model = _ToyMetaModel(4)
    with pytest.raises(ValueError, match="every registered class|missing"):
        _compute_frozen_class_prototypes(model, [_toy_batch("L_s")], class_count=4)


@pytest.mark.parametrize(
    ("candidate_id", "training_mode", "outer_steps"),
    (
        ("P1", "random_adapter", 0),
        ("P2", "supervised_adapter", 1),
        ("P3", "fomaml_fixed_lr", 1),
        ("P4", "fomaml_meta_sgd", 1),
    ),
)
def test_phase1_entry_runs_each_preregistered_candidate_and_four_scenarios(
    tmp_path, candidate_id, training_mode, outer_steps
):
    config = valid_config()
    config["active_candidate_id"] = candidate_id
    config["base_checkpoint"] = "base.pth"
    config["wisig_pkl"] = "ManySig.pkl"
    config_path = tmp_path / f"{candidate_id}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    wisig_path.write_bytes(b"fixture")
    torch.save({"model": _ToyLegacyModel(3).state_dict()}, base_path)

    def batches(config, ds_w, model):
        del config, ds_w, model
        return {"train": [_toy_batch("L_s")] * 4, "eval": [_toy_batch("V_cal"), _toy_batch("V_select")]}

    output = tmp_path / f"out-{candidate_id}"
    result = run_meta_phase1(
        _toy_args(config_path, output, base_path, wisig_path, batches),
        {"rx_list": list(range(7)), "tx_list": ["a", "b", "c"]},
    )
    assert result["candidate_id"] == candidate_id
    assert result["training_mode"] == training_mode
    assert result["task7_outer_steps"] == outer_steps
    with np.load(output / "frozen_prototypes.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"prototypes", "class_ids"}
        assert np.linalg.norm(archive["prototypes"], axis=1).min() > 0
    evaluation = json.loads((output / "final_checkpoint_evaluation.json").read_text(encoding="utf-8"))
    assert set(evaluation["scenarios"]) == {
        "clean",
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    }
    assert all(evaluation["scenarios"][key]["count"] > 0 for key in evaluation["scenarios"])


def test_phase1_missing_selection_evidence_is_scientific_failure_not_technical_failure(
    tmp_path, monkeypatch
):
    import cvsrffi.meta_phase1_entry as entry

    config = valid_config()
    config["base_checkpoint"] = "base.pth"
    config["wisig_pkl"] = "ManySig.pkl"
    config_path = tmp_path / "missing-evidence.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base_path = tmp_path / "base.pth"
    wisig_path = tmp_path / "ManySig.pkl"
    wisig_path.write_bytes(b"fixture")
    torch.save({"model": _ToyLegacyModel(3).state_dict()}, base_path)

    def batches(config, ds_w, model):
        del config, ds_w, model
        return {"train": [_toy_batch("L_s")] * 4, "eval": [_toy_batch("V_cal"), _toy_batch("V_select")]}

    monkeypatch.setattr(
        entry,
        "_candidate_from_curves",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing clean and guard evidence")),
    )
    output = tmp_path / "scientific-failure"
    result = run_meta_phase1(
        _toy_args(config_path, output, base_path, wisig_path, batches),
        {"rx_list": list(range(7)), "tx_list": ["a", "b", "c"]},
    )
    assert result["status"] == "ARTIFACTS_COMPLETE"
    assert result["scientific_verdict"] == "SCIENTIFIC_FAILURE_NO_PROMOTION"
    assert result["candidate_result"] is None
