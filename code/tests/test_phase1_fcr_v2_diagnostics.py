from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import train  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _v2_artifacts() -> dict[str, object]:
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    content = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    zf = torch.stack((labels.float(), 1.0 - labels.float()), dim=1)
    ztx = torch.stack((labels.float() + 0.25, 1.25 - labels.float()), dim=1)
    zch = torch.stack((domains.float(), 1.0 - domains.float()), dim=1)
    zrx = zch + 0.1
    zsync = zch + 0.2
    zgain = zch + 0.3
    zs = torch.stack((content.float(), 1.0 - content.float()), dim=1)
    eta_target = torch.tensor(
        [
            [0.10, 0.20, 0.30],
            [0.11, 0.19, 0.31],
            [0.40, 0.50, 0.60],
            [0.41, 0.49, 0.61],
            [0.10, 0.20, 0.30],
            [0.11, 0.19, 0.31],
            [0.40, 0.50, 0.60],
            [0.41, 0.49, 0.61],
        ],
        dtype=torch.float32,
    )
    eta_pred = eta_target + 0.01
    decode_full = torch.ones(8, 2, 16)
    decode_zero_nuisance = decode_full * 0.7
    decode_swap = decode_full.roll(1, dims=0) * 1.1
    return {
        "z_f_id": zf,
        "z_tx_state": ztx,
        "z_n": {
            "channel": zch,
            "receiver": zrx,
            "sync": zsync,
            "gain": zgain,
        },
        "z_s": zs,
        "tx_labels": labels,
        "domain_labels": domains,
        "content_labels": content,
        "probe_train_mask": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool),
        "probe_eval_mask": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool),
        "clean_z_f_id": zf.detach(),
        "leo_z_f_id": zf.detach() + 0.05,
        "drop_f_error_full": torch.tensor([0.2, 0.3], dtype=torch.float32),
        "drop_f_error_without": torch.tensor([0.7, 0.8], dtype=torch.float32),
        "gram": torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float32)),
        "fisher_coverage": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "eta_target": eta_target,
        "eta_pred": eta_pred,
        "decode_full": decode_full,
        "decode_zero_nuisance": decode_zero_nuisance,
        "decode_swap": decode_swap,
        "response_quality": {
            "energy_ratio": torch.full((8,), 0.25, dtype=torch.float32),
            "state_norm": torch.full((8,), 1.50, dtype=torch.float32),
        },
    }


def _import_v2_module():
    spec = importlib.util.find_spec("cvsrffi.phase1_fcr_v2_diagnostics")
    assert spec is not None, "Task 5 V2 diagnostics module is missing"
    return importlib.import_module("cvsrffi.phase1_fcr_v2_diagnostics")


def test_v2_diagnostics_emit_complete_schema_and_explicit_reasons(tmp_path: Path) -> None:
    module = _import_v2_module()
    metrics = module.collect_fcr_v2_diagnostics(
        _v2_artifacts(),
        resources={
            "train_time_s": 12.5,
            "peak_vram_mb": 321.0,
            "latency_ms": 2.25,
            "epoch_time_s": 0.75,
            "grad_total": 4.0,
            "grad_backbone": 2.0,
            "grad_aux": 1.0,
            "grad_domain": 0.5,
            "active_lambdas": ("self", "eta", "swap"),
            "capability_reasons": {"factor": "disabled_by_row"},
        },
        row_id="R8",
    )

    assert metrics["schema"] == "adv3b02_fcr_diagnostics:v2"
    assert metrics["row_id"] == "R8"
    assert metrics["pair_count"] == 8
    assert metrics["pair_coverage"] == 1.0
    assert metrics["eta_valid_coverage"] == 1.0
    assert metrics["eta_component_error"] == pytest.approx(0.01, rel=1e-5)
    assert metrics["decoder_nuisance_sensitivity"] > 0.0
    assert metrics["swap_output_delta"] > 0.0
    assert metrics["zf_tx_probe"] == 1.0
    assert metrics["z_tx_state_tx_probe"] == 1.0
    assert metrics["grad_backbone_to_total_ratio"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["grad_aux_to_backbone_ratio"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["grad_domain_to_total_ratio"] == pytest.approx(0.125, rel=1e-6)
    assert metrics["grad_clean_leo_cosine"] == "N/A"
    assert metrics["grad_clean_leo_cosine_reason"]
    assert metrics["activation_state"]["active_lambdas"] == ["self", "eta", "swap"]
    assert metrics["per_tx_source_metrics"]["0"]["count"] == 4
    assert metrics["per_tx_source_metrics"]["1"]["count"] == 4

    destination = tmp_path / "fcr_v2_diagnostics.json"
    module.write_fcr_v2_diagnostics(destination, "R8", _v2_artifacts(), resources={"train_time_s": 1.0})
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == "adv3b02_fcr_diagnostics:v2"
    assert payload["row_id"] == "R8"


def test_diagnostics_written_before_deferred_target_return(tmp_path: Path, monkeypatch) -> None:
    helper = getattr(train, "finalize_fcr_v2_diagnostics_before_return", None)
    assert callable(helper), "deferred finalization helper is missing"

    def identity_sat(x, scenario, args, **kwargs):
        del scenario, args, kwargs
        return x, None

    monkeypatch.setattr(train, "apply_sat_channel_for_scenario", identity_sat)
    monkeypatch.setattr(
        train,
        "evaluate_loader",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected target evaluation")),
    )

    model = build_dual_model(
        num_classes=2,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=True,
    )
    iq = torch.randn(8, 2, 64)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    args = SimpleNamespace(
        use_fcr=True,
        defer_target_evaluation=True,
        fcr_diagnostics_path=str(tmp_path / "diag.json"),
        fcr_diagnostics_max_batches=1,
        eval_sat_scenario_list=("leo_clear_weak",),
        sat_seed=2027,
        fcr_ablation_row="R8",
        run_name="formal_R8",
    )

    should_return = helper(
        args,
        model=model,
        source_loader=[(iq, labels, domains)],
        device=torch.device("cpu"),
        training_started_at=time.perf_counter() - 1.0,
        grad_stats={
            "grad_total": 4.0,
            "grad_backbone": 2.0,
            "grad_aux": 1.0,
            "grad_domain": 0.5,
        },
        epoch_time_s=0.75,
        capability_reasons={"factor": "disabled_by_row"},
        active_lambdas=("self", "eta", "swap"),
    )

    assert should_return is True
    payload = json.loads((tmp_path / "diag.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "adv3b02_fcr_diagnostics:v2"
    assert payload["eta_valid_coverage"] >= 0.99
