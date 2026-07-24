from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from scripts import run_grb_jp4_adv_drqknn_bcrr_125 as runner
from scripts import run_adv3b02_ts_drqknn_bcrr_125 as shared


def _args(tmp_path: Path) -> Namespace:
    values = {
        "cache_root": str(tmp_path / "cache"),
        "authority_root": str(tmp_path / "authority"),
        "run_root": str(tmp_path / "run"),
        "gpu_ids": "0,1,2,3,4,5,6,7",
        "phase1_checkpoint": "checkpoint.pth",
        "sealed_runtime": "runtime.pt",
        "package_method_lock": "package_lock.json",
        "grb_outer_bundle": str(tmp_path / "signed_grb_outer"),
        "grb_detached_seal": str(tmp_path / "signed_grb_outer.seal.json"),
        "grb_signature_envelope": str(tmp_path / "signed_grb_outer.signature.json"),
    }
    for name in (
        "detached_seal", "signature_envelope", "checkpoint_lineage", "runtime",
        "component_pre_sign_content_root", "class_handle_binding", "parity_receipt",
        "generation_lock", "method_lock", "generation_config", "generation_code",
        "outer_content_root",
    ):
        values["grb_expected_" + name + "_sha256"] = "a" * 64
    return Namespace(**values)


def test_grb_full125_jobs_are_the_same_frozen_125_surface(tmp_path: Path) -> None:
    jobs = runner.matrix_jobs(_args(tmp_path))
    assert len(jobs) == 125
    assert len({job["job_id"] for job in jobs}) == 125
    assert {(job["k_shot"], job["new_class_count"]) for job in jobs} == set(runner.SLICES)
    assert runner.MATRIX_COUNTS == {
        "jobs": 125,
        "scene_slices": 375,
        "score_rows": 1875,
        "arm_state_prediction_artifacts": 1250,
    }


def test_grb_runner_uses_only_the_shared_scheduler_hook(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run_matrix(args, *, row_runner):
        captured["args"] = args
        captured["runner"] = row_runner
        return {"ok": True}

    monkeypatch.setattr(shared, "run_matrix", fake_run_matrix)
    args = _args(tmp_path)
    assert runner.run_matrix(args) == {"ok": True}
    hook = captured["runner"]
    assert isinstance(hook, shared.MatrixRowRunner)
    assert hook.candidate == runner.CANDIDATE
    assert hook.counts == runner.MATRIX_COUNTS
    assert Path(hook.row_script).name == "run_grb_jp4_adv_drqknn_bcrr_125.py"
    forwarded = hook.extra_row_args(args, runner.matrix_jobs(args)[0])
    assert "--grb-outer-bundle" in forwarded
    assert "--grb-expected-method-lock-sha256" in forwarded
    assert len(forwarded) == 30


def test_grb_formal_bindings_are_forwarded_without_new_authority_input(tmp_path: Path) -> None:
    args = _args(tmp_path)
    kwargs = runner._bundle_kwargs(args)
    assert set(kwargs) == {
        "detached_seal_path", "expected_detached_seal_sha256",
        "signature_envelope_path", "expected_signature_envelope_sha256",
        "expected_checkpoint_lineage_sha256", "expected_runtime_sha256",
        "expected_component_pre_sign_content_root_sha256",
        "expected_class_handle_binding_sha256", "expected_parity_receipt_sha256",
        "expected_generation_lock_sha256", "expected_method_lock_sha256",
        "expected_generation_config_sha256", "expected_generation_code_sha256",
        "expected_outer_content_root_sha256",
    }
    assert kwargs["expected_outer_content_root_sha256"] == "a" * 64


def test_run_row_has_no_preloaded_unconsumed_formal_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    """Each reverified fake bundle must be consumed by exactly one formal fit."""
    from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as dssc

    args = _args(tmp_path)
    args.update = None
    args.output_root = str(tmp_path / "row")
    args.receiver = "20-1"
    args.seed = 713102
    args.k_shot = 10
    args.new_class_count = 5
    args.device = "cpu"
    old_registry = ("old",)
    registry = ("old", "new_0", "new_1", "new_2", "new_3", "new_4")
    loaded: list[object] = []
    fit_bundles: list[object] = []

    def load_bundle(_args):
        bundle = SimpleNamespace(
            component=SimpleNamespace(class_registry=old_registry), runtime=object()
        )
        loaded.append(bundle)
        return bundle

    def support(_payload, classes, _k):
        if tuple(classes) == old_registry:
            return np.zeros((1, 2, 4), np.float32), old_registry, ("old-support",)
        labels = tuple(label for label in registry for _ in range(_k))
        return (
            np.zeros((len(labels), 2, 4), np.float32),
            labels,
            tuple("%s-support-%d" % (label, index) for index, label in enumerate(labels)),
        )

    def query(payload):
        tag = str(payload["tag"])
        return np.zeros((1, 2, 4), np.float32), ("query-" + tag,)

    def fit(*, bundle, **_kwargs):
        fit_bundles.append(bundle)
        return SimpleNamespace(bundle=bundle)

    def predict(_state, *, query_iq, query_tokens):
        assert len(query_iq) == len(query_tokens) == 1
        return ({arm: np.asarray([arm]) for arm in runner.ARMS}, {"query_rows_used_for_fit": 0})

    monkeypatch.setattr(
        runner, "_authority_surfaces",
        lambda _a, _out: (
            {"truth_sidecar": "truth.json"},
            {"before": {scene: {"tag": scene + "-before"} for scene in runner.SCENES},
             "after": {scene: {"tag": scene + "-after"} for scene in runner.SCENES}},
            old_registry, registry,
        ),
    )
    monkeypatch.setattr(runner, "_load_grb_bundle", load_bundle)
    monkeypatch.setattr(dssc, "_support", support)
    monkeypatch.setattr(dssc, "_query", query)
    monkeypatch.setattr(runner, "fit_stage2_b_from_support_iq", fit)
    monkeypatch.setattr(runner, "append_formal_stage2_c", lambda state, **_kwargs: state)
    monkeypatch.setattr(runner, "_state_prediction", predict)
    monkeypatch.setattr(
        runner, "_publish_state_predictions",
        lambda _out, _state, _rows: {arm: "a" * 64 for arm in runner.ARMS},
    )
    monkeypatch.setattr(
        runner.shared, "_score_real_row",
        lambda *_args, **_kwargs: {arm: "b" * 64 for arm in runner.ARMS},
    )
    monkeypatch.setattr(runner.shared, "write_json_new", lambda *_args, **_kwargs: "c" * 64)
    monkeypatch.setattr(runner, "validate_row_artifacts", lambda *_args, **_kwargs: None)

    runner.run_row(args)
    assert len(loaded) == len(runner.SCENES) * 2
    assert fit_bundles == loaded
