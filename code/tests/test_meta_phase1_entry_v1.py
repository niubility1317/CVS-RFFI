import json
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from cvsrffi.meta_phase1_entry import (
    _candidate_from_curves,
    _episode_batch,
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


def _curve_row(role, episode_index, step, mean_accuracy, *, clean_step0_accuracy=0.8, floor_accuracy=0.7):
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
    assert not output_root.exists()

    output_root.mkdir()
    with pytest.raises(FileExistsError):
        main(["--config", str(config_path), "--output-root", str(output_root), "--dry-run"])


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
