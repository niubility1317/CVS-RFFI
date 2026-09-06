"""Single source of revision defaults and source-only experiment configurations.

Protocol-reference metadata is provenance, never proof of waveform alignment.
"""
import argparse
import math
from pathlib import Path


def _boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


CHOICES = {
    "version": ("v1", "v1r", "v2"),
    "reference_mode": ("estimated_reference", "protocol_reference"),
    "solver_mode": ("schur",),
    "anchor_mode": ("real8", "complex24"),
    "fusion_mode": ("off", "fixed"),
    "lr_clock": ("epoch", "effective_step"),
    "leo_ce_mode": ("legacy", "linear_early"),
    "sampler_mode": ("legacy", "balanced_tx_rx"),
    "estimator_variant": ("compact8", "legacy28_old_reference", "legacy28_estimated_reference"),
}
DEFAULTS = {
    "version": "v1", "reference_mode": "estimated_reference",
    "solver_mode": "schur", "anchor_mode": "complex24",
    "protocol_reference_path": "", "protocol_reference_metadata": "",
    "resp_ce_enabled": True, "cross_rx_enabled": False, "u_pair_enabled": False,
    "fusion_mode": "off", "fixed_rho": .05, "rho_cap": .25,
    "update_resp_from_fusion": False, "gate_calibration_enabled": None,
    "compute_only": False, "source_screen_only": None, "source_val_interval": 5,
    "fusion_start_epoch": 91, "lr_clock": "epoch", "leo_ce_mode": "legacy",
    "sampler_mode": "legacy", "cross_rx_margin": .2, "ema_decay": .99, "resume": "",
    "estimator_variant": "compact8",
}
LOSS_DEFAULTS = {"lambda_ecrs_cross_rx": .05, "lambda_ecrs_u_pair": .03,
                 "lambda_ecrs_fusion": 1.0}


def add_ecrs_revision_arguments(parser):
    for key, default in DEFAULTS.items():
        name = "ecrs_" + key
        if isinstance(default, bool) or default is None:
            parser.add_argument("--" + name, type=_boolean, nargs="?", const=True, default=default)
            parser.add_argument("--no_" + name, dest=name, action="store_false")
        else:
            parser.add_argument("--" + name, type=type(default), default=default,
                                **({"choices": CHOICES[key]} if key in CHOICES else {}))
    for name, default in LOSS_DEFAULTS.items():
        parser.add_argument("--" + name, type=float, default=default)
    return parser


def validate_ecrs_revision_args(args, *, require_reference_files=True):
    for key, default in DEFAULTS.items():
        if not hasattr(args, "ecrs_" + key):
            setattr(args, "ecrs_" + key, default)
    for name, default in LOSS_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    if getattr(args, "ecrs_raw_ce_weight", None) is None:
        args.ecrs_raw_ce_weight = .30 if args.ecrs_version == "v1" else 0.
    for name, value in vars(args).items():
        if (name.startswith("ecrs_") or name.startswith("lambda_ecrs_")) and isinstance(value, float):
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
    for key, choices in CHOICES.items():
        if getattr(args, "ecrs_" + key) not in choices:
            raise ValueError("unsupported ecrs_" + key)
    if args.ecrs_gate_calibration_enabled is None:
        args.ecrs_gate_calibration_enabled = args.ecrs_version == "v1"
    if args.ecrs_source_screen_only is None:
        args.ecrs_source_screen_only = args.ecrs_version != "v1"
    if not 0 < args.ecrs_rho_cap <= .25 or args.ecrs_fixed_rho > args.ecrs_rho_cap:
        raise ValueError("rho requires 0 < cap <= .25 and fixed_rho <= cap")
    if args.ecrs_source_val_interval < 1 or args.ecrs_fusion_start_epoch < 1:
        raise ValueError("epoch intervals must be positive")
    if not 0 <= args.ecrs_ema_decay < 1:
        raise ValueError("EMA decay must be in [0,1)")
    if args.ecrs_version != "v1":
        if args.ecrs_gate_calibration_enabled:
            raise ValueError("revision dynamic gate is deferred; use fixed fusion")
        if args.ecrs_fusion_mode == "fixed" and args.ecrs_fixed_rho <= 0:
            raise ValueError("fixed fusion requires positive rho for the zero-initialized projection")
        if args.ecrs_update_resp_from_fusion and args.ecrs_fusion_mode != "fixed":
            raise ValueError("response fusion gradients require fixed fusion")
        if args.ecrs_compute_only and (args.ecrs_resp_ce_enabled or args.ecrs_cross_rx_enabled
                                      or args.ecrs_u_pair_enabled or args.ecrs_fusion_mode != "off"):
            raise ValueError("compute-only control requires all revision objectives and fusion off")
    if args.ecrs_version != "v2" and (args.ecrs_cross_rx_enabled or args.ecrs_u_pair_enabled):
        raise ValueError("cross-RX and U representation learning require v2")
    if args.ecrs_version != "v2" and args.ecrs_estimator_variant != "compact8":
        raise ValueError("estimator bridges require v2")
    if args.ecrs_version == "v1r" and args.ecrs_fusion_mode != "off":
        raise ValueError("V1R is the nonfusion wiring control; fixed fusion requires v2")
    if args.ecrs_version == "v1" and args.ecrs_lr_clock != "epoch":
        raise ValueError("effective-step LR requires a revision runtime")
    if args.ecrs_reference_mode == "protocol_reference":
        if args.ecrs_version != "v2":
            raise ValueError("protocol reference requires v2")
        for name in ("ecrs_protocol_reference_path", "ecrs_protocol_reference_metadata"):
            if require_reference_files and (not getattr(args, name) or not Path(getattr(args, name)).is_file()):
                raise ValueError(name + " must name an existing explicit file; alignment remains unverified")
    return args


def ecrs_model_config(args, *, require_reference_files=True):
    validate_ecrs_revision_args(args, require_reference_files=require_reference_files)
    keys = ("version", "reference_mode", "protocol_reference_path", "protocol_reference_metadata",
            "solver_mode", "anchor_mode", "fusion_mode", "fixed_rho", "rho_cap",
            "update_resp_from_fusion", "compute_only", "estimator_variant")
    result = {key: getattr(args, "ecrs_" + key) for key in keys}
    result.update(response_dim=64, ridge_alpha=getattr(args, "ecrs_ridge_alpha", .01),
                  basis_mode=getattr(args, "ecrs_basis_mode", "fixed_spline"))
    return result


def should_validate_source(epoch, epochs, interval=5, boundaries=(40, 80, 90)):
    """One-based epochs: exactly 48 source checks for the declared 200-epoch plan."""
    if interval < 1 or epochs < 1:
        raise ValueError("epochs and interval must be positive")
    special = {1, epochs - 1, epochs}
    special.update(b + offset for b in boundaries for offset in (-1, 0, 1))
    return 1 <= epoch <= epochs and (epoch % interval == 0 or epoch in special)


SOURCE_PROTOCOL = {"source_rxs": [1, 3, 4, 6, 8], "source_days": [1, 2, 3],
                   "labeled": 6300, "unlabeled": 56700, "validation": 27000,
                   "equalized": 1, "selection_domain": "source_only"}
UNAVAILABLE_ROWS = {
    "B7": "dynamic gate is deferred pending implemented route and source B6 evidence",
}


def experiment_matrix():
    base = {"ecrs_version": "v1r", "ecrs_resp_ce_enabled": True,
            "ecrs_fusion_mode": "off", "ecrs_raw_ce_weight": 0.,
            "ecrs_gate_calibration_enabled": False, "ecrs_rung": "R7"}
    b4 = dict(base, ecrs_version="v2", ecrs_anchor_mode="complex24")
    b5 = dict(b4, ecrs_cross_rx_enabled=True, ecrs_u_pair_enabled=True)
    return {
        "B0": dict(base, use_ecrs=False, ecrs_resp_ce_enabled=False),
        "B1": dict(base, use_ecrs=True, ecrs_compute_only=True, ecrs_resp_ce_enabled=False),
        "B2-V1": dict(base, use_ecrs=True, ecrs_version="v1", ecrs_raw_ce_weight=.30),
        "B2": dict(base, use_ecrs=True),
        "B3": dict(b4, use_ecrs=True, ecrs_anchor_mode="real8"),
        "B3a": dict(b4, use_ecrs=True, ecrs_anchor_mode="real8", ecrs_estimator_variant="legacy28_old_reference"),
        "B3b": dict(b4, use_ecrs=True, ecrs_anchor_mode="real8", ecrs_estimator_variant="legacy28_estimated_reference"),
        "B3c": dict(b4, use_ecrs=True, ecrs_anchor_mode="real8", ecrs_estimator_variant="compact8"),
        "B4": dict(b4, use_ecrs=True),
        "B5-X": dict(b4, use_ecrs=True, ecrs_cross_rx_enabled=True),
        "B5-U": dict(b4, use_ecrs=True, ecrs_u_pair_enabled=True),
        "B5-XU": dict(b5, use_ecrs=True),
        "B6": dict(b5, use_ecrs=True, ecrs_fusion_mode="fixed"),
        "S-LR": dict(b5, use_ecrs=True, ecrs_lr_clock="effective_step"),
        "S-LEO": dict(b5, use_ecrs=True, ecrs_leo_ce_mode="linear_early"),
        "S-BATCH": dict(b5, use_ecrs=True, ecrs_sampler_mode="balanced_tx_rx"),
    }
