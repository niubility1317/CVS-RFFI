from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "code" / "scripts" / "probe_d80_ground_commonmode_covariance_denoiser.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("d80_probe_test_module", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_lock_is_covariance_not_anchor_or_row_residual():
    probe = _load_probe()
    assert probe.ARM == "ground_commonmode_covariance_denoiser"
    assert "mean(scale^2/12)" in probe.FORMULA
    assert "full/block" in probe.FORMULA
    assert "row residual" not in probe.FORMULA
    assert "anchor" not in probe.FORMULA


def test_probe_injects_before_d62_closure_and_restores_factories():
    source = PROBE.read_text(encoding="utf-8")
    patch_index = source.index("d43.build_structured_fit = structured_builder")
    build_index = source.index("base_fit, call_records = d62.build_d62_fit(d42)")
    restore_index = source.index("d43.build_structured_fit = original_builder")
    assert patch_index < build_index < restore_index
    assert "D80 D43 module alias identity drift" in source


def test_probe_has_protocol_resource_and_hash_closure():
    source = PROBE.read_text(encoding="utf-8")
    required = (
        "d80_query_extra_macs",
        "d80_optimizer_steps_extra",
        "d80_trainable_parameters_extra",
        "ground_component_entry_npz_sha256",
        "ground_component_exit_npz_sha256",
        "ground_component_entry_manifest_sha256",
        "ground_component_exit_manifest_sha256",
        "formal_candidate",
        "selected_only_full_k10_refit_allowed",
    )
    assert all(token in source for token in required)


def test_probe_does_not_claim_bundle_radius_or_count():
    source = PROBE.read_text(encoding="utf-8")
    assert '"ground_bundle_contains_sample_radius": False' in source
    assert '"ground_bundle_contains_sample_count": False' in source

