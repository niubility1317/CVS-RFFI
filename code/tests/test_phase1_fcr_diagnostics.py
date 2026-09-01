from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

REQUIRED = {
    "zf_tx_probe",
    "zf_domain_probe",
    "zn_domain_probe",
    "zn_tx_probe",
    "zs_content_probe",
    "clean_leo_zf_distance",
    "same_tx_zf_distance",
    "drop_f_residual_gap",
    "transplant_target_id",
    "transplant_preserve_s",
    "transplant_preserve_n",
    "gram_condition",
    "effective_rank",
    "fisher_coverage",
    "train_time_s",
    "peak_vram_mb",
    "latency_ms",
}


def _artifacts() -> dict[str, object]:
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    content = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    zf = torch.stack((labels.float(), 1.0 - labels.float()), dim=1).requires_grad_()
    zn = torch.stack((domains.float(), 1.0 - domains.float()), dim=1).requires_grad_()
    zs = torch.stack((content.float(), 1.0 - content.float()), dim=1).requires_grad_()
    return {
        "z_f_id": zf,
        "z_n": zn,
        "z_s": zs,
        "tx_labels": labels,
        "domain_labels": domains,
        "content_labels": content,
        "probe_train_mask": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool),
        "probe_eval_mask": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool),
        "clean_z_f_id": zf.detach(),
        "leo_z_f_id": zf.detach() + 0.25,
        "same_tx_zf_left": zf.detach()[:4],
        "same_tx_zf_right": zf.detach()[4:],
        "drop_f_error_full": torch.tensor([0.2, 0.3]),
        "drop_f_error_without": torch.tensor([0.8, 0.9]),
        "strict_transplant": {
            "target_id": torch.tensor(0.75),
            "preserve_s": torch.tensor(0.10),
            "preserve_n": torch.tensor(0.20),
        },
        "gram": torch.diag(torch.tensor([4.0, 1.0])),
        "fisher_coverage": torch.tensor([0.5, 1.0]),
    }


def _compute_fcr_diagnostics(*args, **kwargs):
    spec = importlib.util.find_spec("cvsrffi.phase1_fcr_diagnostics")
    assert spec is not None, "Task11 diagnostics module is missing"
    module = importlib.import_module("cvsrffi.phase1_fcr_diagnostics")
    return module.compute_fcr_diagnostics(*args, **kwargs)


def test_diagnostics_emit_required_detached_metrics_without_backward(tmp_path: Path) -> None:
    artifacts = _artifacts()
    metrics = _compute_fcr_diagnostics(
        artifacts,
        resources={"train_time_s": 12.5, "peak_vram_mb": 321.0, "latency_ms": 2.25},
        row_id="R6",
    )

    assert REQUIRED <= set(metrics)
    assert metrics["row_id"] == "R6"
    assert metrics["zf_tx_probe"] == 1.0
    assert metrics["zn_domain_probe"] == 1.0
    assert metrics["clean_leo_zf_distance"] > 0.0
    assert metrics["drop_f_residual_gap"] > 0.0
    assert metrics["transplant_target_id"] == 0.75
    assert all(not torch.is_tensor(value) for value in metrics.values())
    assert artifacts["z_f_id"].grad is None
    assert artifacts["z_n"].grad is None
    assert artifacts["z_s"].grad is None

    module = importlib.import_module("cvsrffi.phase1_fcr_diagnostics")
    destination = tmp_path / "nested" / "diagnostics.json"
    module.write_fcr_diagnostics_json(destination, metrics)
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_missing_strict_pair_metrics_are_na_with_explicit_reasons_not_zero() -> None:
    artifacts = _artifacts()
    for key in (
        "same_tx_zf_left",
        "same_tx_zf_right",
        "drop_f_error_full",
        "drop_f_error_without",
        "strict_transplant",
    ):
        artifacts.pop(key)
    metrics = _compute_fcr_diagnostics(artifacts, resources={}, row_id="R5")

    for key in (
        "same_tx_zf_distance",
        "drop_f_residual_gap",
        "transplant_target_id",
        "transplant_preserve_s",
        "transplant_preserve_n",
    ):
        assert metrics[key] == "N/A"
        assert isinstance(metrics[f"{key}_reason"], str)
        assert metrics[f"{key}_reason"].strip()


def test_training_external_collector_detaches_source_clean_leo_artifacts() -> None:
    from model_dual_cvsincnet import build_dual_model
    import train

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

    def legal_leo_transform(clean: torch.Tensor, _batch_index: int, _extra) -> torch.Tensor:
        return clean * 0.9

    artifacts, resources = train.collect_fcr_diagnostic_artifacts(
        model,
        [(iq, labels, domains)],
        torch.device("cpu"),
        leo_transform=legal_leo_transform,
        max_batches=1,
    )

    for key in ("z_f_id", "z_n", "z_s", "clean_z_f_id", "leo_z_f_id", "gram"):
        value = artifacts[key]
        if isinstance(value, dict):
            assert all(tensor.requires_grad is False for tensor in value.values())
        else:
            assert value.requires_grad is False
    assert artifacts["z_f_id"].shape[0] == 8
    assert artifacts["clean_z_f_id"].shape == artifacts["leo_z_f_id"].shape
    assert artifacts["decoder_mode"] == "full_physics"
    assert resources["latency_ms"] >= 0.0
