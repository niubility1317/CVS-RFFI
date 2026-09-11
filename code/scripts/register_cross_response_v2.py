"""Create the versioned candidate configuration; never freeze source parameters or launch."""
import copy
import json
from pathlib import Path
from scripts.core90_cross_response_matrix import DEFAULT_CONFIG, load_config


def configuration():
    c = load_config(DEFAULT_CONFIG)
    c["schema_version"] = 2
    c["cross_response"].update(implementation_version=2, data_seed=392005,
        diagnostic_interval=20, statistics_audit=False, geometry_min_norm=1e-8,
        direction_shrinkage=0., gain_strategy="legacy_gain", evidence_config=None,
        decision_mode="vectorized", decision_calibration=None, interaction_mode="raw",
        response_routing="full_error", mechanism_gate=None, response_decomposition=None,
        source_audit_enabled=False, source_baseline_fit_steps=None, source_baseline_fit_lr=None)
    v = c["variants"]
    v["U1_mask_off"] = dict(v["U1"], mixstyle_role_policy="original_mask")
    v["U3_delta"] = dict(v["U3"], decision_mode="delta")
    v["U3_delta_pairs"] = dict(v["U3"], decision_mode="delta_pairs")
    v["Ux_normalized"] = dict(v["Ux"], interaction_mode="normalized")
    v["U4_decomposed"] = dict(v["U4_additive"], response_routing="decomposed")
    v["U5_reliable"] = dict(v["U5"], gain_strategy="reliable_evidence")
    c["v2_registration"] = dict(status="IMPLEMENTATION_CANDIDATES_NOT_LAUNCH_AUTHORIZATION",
        primary_controls=["U1", "U3", "U3_delta", "U3_delta_pairs"],
        organization_controls=["U0", "U1_mask_off", "U1"],
        organization_identifiability="U1/U1_mask_off use identical complete block tickets and differ only in MixStyle mask; U0 is an ordinary-loader bridge, not a full 2x2 factorial",
        conditional_candidates=["U3_delta", "U3_delta_pairs", "U4_decomposed", "U5_reliable"],
        data_seed=392005, model_seeds=None, training_budget=dict(epochs=200,label_epochs=130,pseudo_epochs=70,batch_size=128,steps_per_epoch=49),
        confirmation_boundary=None, target_results_reused=False,
        source_parameters="UNFROZEN: no numeric defaults for noise quantiles/pair selection/gate thresholds/response reliability",
        launch=False)
    return c


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "configs" / "phase1_core90_cross_response_v2.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(configuration(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(target)
