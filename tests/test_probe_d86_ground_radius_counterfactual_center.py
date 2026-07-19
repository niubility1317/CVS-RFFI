from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "code" / "scripts" / "probe_d86_ground_radius_counterfactual_center.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("d86_probe_test_module", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_lock_is_nonquadratic_parameter_free_and_support_only() -> None:
    probe = _load_probe()
    assert probe.ARM == "ground_radius_counterfactual_robust_center"
    assert "nearest-rival margin" in probe.FORMULA
    assert "sqrt(2*median_class_p90_radius)" in probe.FORMULA
    assert "Cauchy weight" in probe.FORMULA
    assert "query" not in probe.FORMULA


def test_resource_inventory_covers_every_d62_full_and_block_closure() -> None:
    probe = _load_probe()
    classes, shots, domains = 6, 8, 14
    outer = probe._transform_macs(classes, shots, domains)
    inner = probe._transform_macs(classes, shots - 1, domains)
    assert probe._d62_counterfactual_chain_macs(classes, shots, domains) == (
        4 * outer + 4 * shots * inner
    )


def test_synthetic_d62_stack_keeps_support_cardinality_and_affine_output() -> None:
    probe = _load_probe()
    rng = np.random.default_rng(8602)
    prototypes = rng.normal(size=(14, 6, 160))
    radius = rng.uniform(0.001, 0.02, size=(14, 6))
    templates, amplitude, template_audit = (
        probe.core.ground_radius_counterfactual_templates(prototypes, radius)
    )
    ground_audit = {
        "ground_component_input_count": 84,
        "ground_statistic_semantics": "test_v2_center_radius",
        "d84_template_sha256": template_audit["template_sha256"],
        "d84_weight_sha256": template_audit["weight_sha256"],
        "d84_retained_domain_template_count": template_audit[
            "retained_domain_template_count"
        ],
        "d84_template_policy": template_audit["template_policy"],
        "d84_reliability_mean": template_audit["reliability_mean"],
    }
    probe.d85.base.core.translate_to_consensus_robust_centers = (
        probe.core.translate_to_counterfactual_robust_centers
    )
    probe.d85.base.core.build_consensus_center_component_fit = (
        probe.core.build_counterfactual_center_component_fit
    )
    fit, component_records, transform_records = probe.d85.base.build_d84_fit(
        d42, templates, amplitude, ground_audit
    )
    classes, shots = 3, 4
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = means[labels] + 0.1 * rng.normal(size=(classes * shots, 288))
    coefficient, intercept, audit = fit(rows, labels, classes, shots)

    assert coefficient.shape == (classes, 288)
    assert intercept.shape == (classes,)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    assert len(component_records) == 2 * (shots + 1)
    assert len(transform_records) == 4 * (shots + 1)
    assert all(record["k_shot"] in (shots, shots - 1) for record in transform_records)
    assert audit["d84_query_metric_source"] == "target_support_only_d62"

