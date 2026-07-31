"""Build and independently seal the D105 source-only MetaBias4 asset.

This module is intentionally an asset boundary, not a Phase2 evaluator. It
accepts a verified Phase1 strict-tap archive only while constructing the
aggregate. The emitted component contains a serialized RXIDMetaBias4Bundle
plus hashes and aggregate receipts; it never retains raw IQ, source rows,
per-sample features, class handles, receiver names, or physical IDs.

The builder cannot self-authorize a formal Phase2 asset. It always emits a
non-promotable component first. A second, independent authority receipt can
seal an immutable copy only after all source-held gates are present and pass.
This keeps the D102 rejected analytic bundle outside the D105 trust chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from cvsrffi.rxid_metabias4_bundle import (
    CODE_DIM,
    DOMAIN_DIM,
    RXIDMetaBias4Bundle,
    RXIDMetaBias4BundleError,
    Z_DIM,
    build_rxid_metabias4_bundle,
    deserialize_rxid_metabias4_bundle,
    serialize_rxid_metabias4_bundle,
)
from cvsrffi.stage2_d105_cbrc import (
    D105CBRCBundleHandle,
    compute_d105_bundle_receipt_root,
    compute_d105_bundle_validator_receipt,
    make_d105_cbrc_bundle_handle,
)
from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA_V1,
    LEO_WEAK_CACHE_SET_SCHEMA_V1,
    load_verified_leo_weak_cache_set,
)
from cvsrffi.stage2_d105_phase1_authority import (
    AUTHORITY_ENVELOPE_NAME,
    AUTHORITY_ENVELOPE_SCHEMA,
    AUTHORITY_SIGNATURE_NAME,
    D102_REVOCATION_MANIFEST_NAME,
    D102_REVOCATION_SIGNATURE_NAME,
    D105AuthorityError,
    INDEPENDENT_REVIEW_RECEIPT_NAME,
    consume_authority_nonce_once,
    load_independent_review_receipt,
    load_signed_d102_revocation_manifest,
    load_signed_d105_authority_envelope,
    reject_revoked_d102_identity,
)


if TYPE_CHECKING:
    # The frozen D105 runtime manifest retains this historical module in its
    # static archive closure.  It is deliberately never imported or called by
    # the Phase1 D105 dual-backbone forward path.
    from cvsrffi.stage2_grb_jp4_adv_drqknn_bcrr import StrictForward as _StrictForward


SCHEMA = "cvs.phase1.d105.cbrc.strict_source_bundle.v1"
STRICT_TAP_SCHEMA = "cvs.phase1.d105.cbrc.strict_tap_receipt.v2"
D105_STRICT_TAP_FORWARD_BATCH_CAPACITY = 256
D105_STRICT_TAP_FORWARD_BATCH_POLICY = "fixed_256_zero_pad_then_slice_v1"
SOURCE_ACCESS_SCHEMA = "cvs.phase1.d105.cbrc.source_access_receipt.v1"
SOURCE_HELD_PREDICTION_SCHEMA = (
    "cvs.phase1.d105.cbrc.source_held_prediction_manifest.v1"
)
SOURCE_HELD_TRUTH_OPEN_SCHEMA = (
    "cvs.phase1.d105.cbrc.source_held_truth_open_receipt.v1"
)
SOURCE_HELD_SCORE_SCHEMA = "cvs.phase1.d105.cbrc.source_held_score_artifact.v2"
SOURCE_HELD_PREDICTION_COMMIT_SCHEMA = (
    "cvs.phase1.d105.cbrc.source_held_prediction_commit.v1"
)
SOURCE_HELD_GATE_SCHEMA = "cvs.phase1.d105.cbrc.source_held_gate.v1"
D105_CANDIDATE_METHOD_LOCK_SCHEMA = "cvs.stage2.d105.candidate_method_lock.v1"
D105_CANDIDATE_RUNTIME_MANIFEST_SCHEMA = (
    "cvs.stage2.d105.candidate_runtime_manifest.v1"
)
# ``AUTHORITY_SEAL_SCHEMA`` remains an import-compatible name only.  Its value
# is deliberately the signed v2 envelope schema; the old unsigned v1 receipt
# is never accepted by this module.
AUTHORITY_SEAL_SCHEMA = AUTHORITY_ENVELOPE_SCHEMA
PROTOCOL_SCHEMA = "p2_min_v1"

# These two constants are copied from the historical one-observation source
# selection implementation.  They intentionally live in the D105 runtime so
# ``tap-cache`` does not import the older exporter and its unrelated training
# and paper-reproduction dependency tree.
D105_TAP_CACHE_SELECTION_SALT_SCHEMA = (
    "cvs.phase1.singleobs_selection_salt_receipt.v1"
)
D105_TAP_CACHE_SELECTION_DOMAIN = b"P1_SINGLE_LEO_V1"

CANDIDATE_ID = "D105-CBRC+LPO-RC"
COMPONENT_STATUS = "PHASE1_COMPONENT_PENDING_INDEPENDENT_FORMAL_SEAL"
DIAGNOSTIC_STATUS = "PHASE1_COMPONENT_SOURCE_HELD_GATES_INCOMPLETE"
FORMAL_STATUS = "FORMAL_PHASE2_ELIGIBLE_SEALED"

# The authority helper is a transitive security dependency of this listed
# Phase1 module.  Pin it here so modifying the helper alone cannot evade the
# candidate runtime manifest's hash of ``stage2_d105_phase1_bundle.py``.
D105_PHASE1_AUTHORITY_MODULE_SHA256 = (
    "fe81a728bcd8e1047a40069b9d9954aed2af1c89b98633489ccf2b922b4364bd"
)

D105_CANDIDATE_RUNTIME_ENTRYPOINTS = {
    "phase1_asset": "cvsrffi.stage2_d105_phase1_bundle",
    "phase1_authority": "cvsrffi.stage2_d105_phase1_authority",
    "phase1_builder_cli": "scripts/build_d105_phase1_bundle.py",
    "phase1_authority_signer_cli": "scripts/sign_d105_phase1_authority.py",
    "cbrc": "cvsrffi.stage2_d105_cbrc",
    "feature_tap": "cvsrffi.stage2_d105_feature_tap:extract_d105_feature_tap",
    "feature_tap_smoke_cli": "scripts/run_d105_four_arm_real_feature_smoke.py",
    "four_arm": "cvsrffi.stage2_d105_four_arm",
    "head": "cvsrffi.stage2_lpo_rc_qknn",
    "query_evaluator": "cvsrffi.stage2_d105_query_evaluation:evaluate_d105_query_row",
    "target25_runner": "cvsrffi.stage2_d105_target25_runner",
    "target25_launcher": "cvsrffi.stage2_d105_target25_launcher",
    "target25_input_builder": (
        "cvsrffi.stage2_d105_target25_inputs:prepare_d105_target25_inputs"
    ),
    "target25_prepare_cli": "scripts/prepare_d105_target25_inputs.py",
    "target25_run_cli": "scripts/run_d105_target25.py",
}

D105_CANDIDATE_RUNTIME_CVSRFFI_FILES = (
    # Exact recursive local-Python closure of the ten ``cvsrffi`` D105
    # entrypoints above.  This includes the package initializer because
    # ``from cvsrffi import somph_runtime_trust`` executes it before resolving
    # the fixed trust anchor.  Do not replace this with a shallow entrypoint
    # list: a transitive helper can otherwise change signature, model, data,
    # or prediction semantics outside the signed runtime manifest.
    "cvsrffi/__init__.py",
    "cvsrffi/dual_feature_forward.py",
    "cvsrffi/leo_weak_cache.py",
    "cvsrffi/phase1_adv3b02_deployment_bundle.py",
    "cvsrffi/phase1_center_lowrank_prototype_bundle.py",
    "cvsrffi/phase1_grb_jp4_bundle.py",
    "cvsrffi/phase1_rb_metabias4_bundle.py",
    "cvsrffi/phase2_runtime_contract.py",
    "cvsrffi/rxid_metabias4_bundle.py",
    "cvsrffi/rxid_metabias4_held_execution.py",
    "cvsrffi/rxid_metabias4_phase1_trainer.py",
    "cvsrffi/rxid_metabias4_source_archive.py",
    "cvsrffi/somph_cache_build_matrix.py",
    "cvsrffi/somph_diagnostic_bundle_loader.py",
    "cvsrffi/somph_formal_matrix.py",
    "cvsrffi/somph_leo_weak_lineage_seal.py",
    "cvsrffi/somph_lineage_authority.py",
    "cvsrffi/somph_metric_scorer.py",
    "cvsrffi/somph_offline_target_package.py",
    "cvsrffi/somph_prediction_artifact.py",
    "cvsrffi/somph_predictor_bundle.py",
    "cvsrffi/somph_predictor_runtime.py",
    "cvsrffi/somph_runtime_trust.py",
    "cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py",
    "cvsrffi/stage2_d105_cbrc.py",
    "cvsrffi/stage2_d105_feature_tap.py",
    "cvsrffi/stage2_d105_four_arm.py",
    "cvsrffi/stage2_d105_phase1_authority.py",
    "cvsrffi/stage2_d105_phase1_bundle.py",
    "cvsrffi/stage2_d105_query_evaluation.py",
    "cvsrffi/stage2_d105_target25_inputs.py",
    "cvsrffi/stage2_d105_target25_launcher.py",
    "cvsrffi/stage2_d105_target25_runner.py",
    "cvsrffi/stage2_diag_cosine_exploration.py",
    "cvsrffi/stage2_dssc_zdom_jg_qknn_r4_bcrr.py",
    "cvsrffi/stage2_grb_jp4_adv_drqknn_bcrr.py",
    "cvsrffi/stage2_lpo_rc_qknn.py",
    "cvsrffi/stage2_metric_scorer.py",
    "cvsrffi/stage2_prediction_artifact.py",
    "cvsrffi/stage2_predictor_bundle.py",
    "cvsrffi/stage2_predictor_runtime.py",
    "cvsrffi/stage2_rb_metabias4_qknn.py",
    "cvsrffi/stage2_rxid_metabias4.py",
    "cvsrffi/stage2_svrn_bcr.py",
    "cvsrffi/stage2_zid_student_t_qknn.py",
    "training_controls.py",
)

# The D105 exact-checkpoint reconstruction path intentionally bypasses the
# generic ``checkpoint_loading`` fallback, whose implementation imports the
# complete SSDG training program.  The runtime instead depends only on this
# fixed model factory and the CVSincNet backbone it selects in this checkout.
# ``build_d105_exact_model_from_checkpoint`` rejects a changed factory origin
# before a model is constructed.
D105_CANDIDATE_RUNTIME_MODEL_FILES = (
    "baseline_origin_sat_view.py",
    "model.py",
    "model_dual_cvsincnet.py",
)

D105_CANDIDATE_RUNTIME_FILES = (
    D105_CANDIDATE_RUNTIME_CVSRFFI_FILES
    + D105_CANDIDATE_RUNTIME_MODEL_FILES
    + (
    "scripts/build_d105_phase1_bundle.py",
    "scripts/sign_d105_phase1_authority.py",
    "scripts/run_d105_four_arm_real_feature_smoke.py",
    "scripts/prepare_d105_target25_inputs.py",
    "scripts/run_d105_target25.py",
    )
)

BUNDLE_WIRE_NAME = "d105_phase1_aggregate.wire"
MANIFEST_NAME = "d105_phase1_bundle.manifest.json"
SEAL_NAME = "d105_phase1_bundle.manifest.sha256"
COMPONENT_MANIFEST_NAME = "d105_phase1_component.manifest.json"
HELD_GATE_NAME = "d105_source_held_gate.json"
SOURCE_ACCESS_RECEIPT_NAME = "d105_source_access_receipt.json"
STRICT_TAP_ARCHIVE_NAME = "d105_phase1_strict_tap.npz"
STRICT_TAP_RECEIPT_NAME = "d105_phase1_strict_tap_receipt.json"

STRICT_TAP_MEMBERS = (
    "pre_relu",
    "z_dom",
    "labels",
    "receiver_ids",
    "physical_ids",
)

FORMAL_PREREQUISITES = (
    "real_checkpoint_strict_tap",
    "source_only_target_rows_zero",
    "receiver_held_k1_k5_k10_complete",
    "receiver_held_all_noninferior",
    "class_loco_complete_and_noninferior",
    "tx_probe_max_balanced_accuracy_at_most_0_25",
    "int8_agreement_at_least_0_995_and_zero_large_margin_flip",
    "no_persistent_fp32_or_source_replay",
    "independent_review_p0_0_p1_0",
    "independent_phase2_authority_seal",
)

_EPS = 1.0e-10


_D105_CANDIDATE_METHOD_LOCK_FIELDS = {
    "schema",
    "candidate_id",
    "protocol_schema",
    "checkpoint_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_cbrc",
    "student_t_qknn",
    "four_arm",
    "source_held",
    "target25",
}


_D105_CANDIDATE_RUNTIME_MANIFEST_FIELDS = {
    "schema",
    "candidate_id",
    "protocol_schema",
    "checkpoint_sha256",
    "entrypoints",
    "core_file_sha256",
}


class D105Phase1BundleError(ValueError):
    """Raised when a D105 Phase1 source asset is malformed or untrusted."""


def _tensor_from_d105_float32_c_iq(
    value: np.ndarray,
    *,
    torch_module: Any,
    device: Any,
    error_type: type[Exception],
    name: str,
) -> Any:
    """Copy one validated D105 IQ batch without Torch's ndarray C-API bridge.

    N607's Torch 2.1 / NumPy 2.x pairing can reject a valid ``np.ndarray`` at
    ``torch.from_numpy`` because Torch retains an incompatible ndarray C-API
    identity.  ``frombuffer`` reads the already checked C-order bytes instead;
    the immediate clone makes the returned tensor independent of the NumPy
    buffer and its lifetime.
    """

    if (
        type(value) is not np.ndarray
        or value.dtype != np.float32
        or value.ndim != 3
        or value.shape[0] < 1
        or value.shape[1] != 2
        or value.shape[2] < 1
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise error_type(
            f"{name} tensor bridge requires finite C-contiguous float32 [N,2,T]"
        )
    try:
        copied = (
            torch_module.frombuffer(
                value,
                dtype=torch_module.float32,
                count=int(value.size),
            )
            .reshape(value.shape)
            .clone()
        )
        tensor = copied.to(device=device, dtype=torch_module.float32)
    except (RuntimeError, TypeError, ValueError) as error:
        raise error_type(f"{name} tensor bridge is unavailable") from error
    if (
        tensor.dtype != torch_module.float32
        or tuple(tensor.shape) != tuple(value.shape)
        or not bool(torch_module.isfinite(tensor).all().item())
    ):
        raise error_type(f"{name} tensor bridge output drift")
    return tensor


def _canonical_bytes(value: Any) -> bytes:
    """Encode a small metadata payload deterministically and fail on NaN."""

    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        convert(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 without following a symbolic link."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D105Phase1BundleError("expected a regular non-symlink file")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# This is the model-construction subset of the historical SSDG parser plus
# ``post_stage_common.build_baseline_model`` defaults.  It is deliberately
# local and finite: D105 uses the checkpoint's stored values when present and
# otherwise exactly these frozen defaults.  Keeping it here prevents the
# inference/Phase1 asset path from importing ``SSDG.train_ssdg`` (a training
# program with a much wider local import graph).
_D105_MODEL_RECONSTRUCTION_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "num_classes": 16,
        "dataset": "wisig",
        "sample_rate_hz": 0.0,
        "model_size": "M",
        "model_variant": "lite_d",
        "id_feature_key": "feat_joint",
        "dom_feature_key": "feat_imp",
        "branch_ablation": "no_dac",
        "domain_branch_ablation": "no_stats",
        "domain_enhancer": "rcn_stats",
        "domain_enhancer_strength": 0.35,
        "use_mixstyle": True,
        "mixstyle_p": 0.18,
        "mixstyle_alpha": 0.10,
        "mixstyle_eps": 1.0e-6,
        "mixstyle_layers": "time_down,t1",
        "mixstyle_use_domain_label": True,
        "mixstyle_mix": "same_tx_crossdomain",
        "mixstyle_strength": 0.70,
        "mixstyle_fallback": "skip",
        "use_circularity": True,
        "use_freq_stats": True,
        "use_pa_stats": True,
        "use_freq_band_gate": True,
        "freq_feature_source": "raw_fft",
        "pa_feature_source": "raw_iq",
        "pa_orders": None,
        "use_aux_spectral_stats": True,
        "channel_trim_scale": 1.0,
        "id_time_stability_mode": "off",
        "id_freq_stability_mode": "off",
        "domain_time_stability_mode": "off",
        "domain_freq_stability_mode": "off",
        "time_stability_channels": 8,
        "freq_stability_channels": 4,
        "fast_infer_when_no_aux": True,
        "arch_family": "cvsincnet",
    }
)


def _d105_reconstruction_model_kwargs(
    checkpoint_args: Mapping[str, Any], *, input_len: int, num_domains: int
) -> dict[str, Any]:
    """Normalize the fixed checkpoint-to-CVSincNet construction contract."""

    if type(checkpoint_args) is not dict:
        raise D105Phase1BundleError("checkpoint args must be an exact mapping")
    if isinstance(input_len, bool) or int(input_len) <= 0:
        raise D105Phase1BundleError("checkpoint reconstruction input length is invalid")
    if isinstance(num_domains, bool) or int(num_domains) <= 0:
        raise D105Phase1BundleError("checkpoint reconstruction domain count is invalid")
    merged = dict(_D105_MODEL_RECONSTRUCTION_DEFAULTS)
    merged.update(checkpoint_args)
    try:
        dataset = str(merged["dataset"])
        sample_rate_hz = float(merged["sample_rate_hz"])
        if sample_rate_hz <= 0.0:
            sample_rate_hz = 25.0e6 if dataset == "wisig" else 5.0e6
        kwargs = {
            "num_classes": int(merged["num_classes"]),
            "num_domains": int(num_domains),
            "model_size": str(merged["model_size"]),
            "dataset": dataset,
            "input_len": int(input_len),
            "sample_rate_hz": sample_rate_hz,
            "id_feature_key": str(merged["id_feature_key"]),
            "dom_feature_key": str(merged["dom_feature_key"]),
            "model_variant": str(merged["model_variant"]),
            "branch_ablation": str(merged["branch_ablation"]),
            "mixstyle_on": bool(merged["use_mixstyle"]),
            "mixstyle_p": float(merged["mixstyle_p"]),
            "mixstyle_alpha": float(merged["mixstyle_alpha"]),
            "mixstyle_eps": float(merged["mixstyle_eps"]),
            "mixstyle_layers": str(merged["mixstyle_layers"]),
            "mixstyle_use_domain_label": bool(
                merged["mixstyle_use_domain_label"]
            ),
            "mixstyle_mix": str(merged["mixstyle_mix"]),
            "mixstyle_strength": float(merged["mixstyle_strength"]),
            "mixstyle_fallback": str(merged["mixstyle_fallback"]),
            "domain_branch_ablation": str(merged["domain_branch_ablation"]),
            "domain_enhancer": str(merged["domain_enhancer"]),
            "domain_enhancer_strength": float(merged["domain_enhancer_strength"]),
            "use_circularity": bool(merged["use_circularity"]),
            "use_freq_stats": bool(merged["use_freq_stats"]),
            "use_pa_stats": bool(merged["use_pa_stats"]),
            "use_freq_band_gate": bool(merged["use_freq_band_gate"]),
            "freq_feature_source": str(merged["freq_feature_source"]),
            "pa_feature_source": str(merged["pa_feature_source"]),
            "pa_orders": merged["pa_orders"],
            "use_aux_spectral_stats": bool(merged["use_aux_spectral_stats"]),
            "channel_trim_scale": float(merged["channel_trim_scale"]),
            "id_time_stability_mode": str(merged["id_time_stability_mode"]),
            "id_freq_stability_mode": str(merged["id_freq_stability_mode"]),
            "domain_time_stability_mode": str(
                merged["domain_time_stability_mode"]
            ),
            "domain_freq_stability_mode": str(
                merged["domain_freq_stability_mode"]
            ),
            "time_stability_channels": int(merged["time_stability_channels"]),
            "freq_stability_channels": int(merged["freq_stability_channels"]),
            "fast_infer_when_no_aux": bool(merged["fast_infer_when_no_aux"]),
            "arch_family": str(merged["arch_family"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise D105Phase1BundleError(
            "checkpoint model-argument reconstruction drift"
        ) from error
    if kwargs["num_classes"] < 2 or kwargs["arch_family"] != "cvsincnet":
        raise D105Phase1BundleError("checkpoint model-family contract drift")
    return kwargs


def load_d105_exact_sha_bound_checkpoint(
    path: str | Path, expected_sha256: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Load an exact-byte checkpoint through the bounded legacy bridge.

    Some N607-compatible PyTorch releases cannot use ``weights_only`` with
    the historical checkpoint.  The fallback remains acceptable only after
    the file has matched the caller's exact SHA256, and the one safe global it
    may resolve is explicitly part of the D105 runtime manifest.
    """

    source = Path(path)
    expected = _require_sha256(expected_sha256, "checkpoint SHA256")
    if source.is_symlink() or not source.is_file() or sha256_file(source) != expected:
        raise D105Phase1BundleError("checkpoint SHA256 drift")
    try:
        import torch
        from baseline_origin_sat_view import SatViewStage
    except ImportError as error:  # pragma: no cover - deployment dependency.
        raise D105Phase1BundleError("D105 checkpoint loader dependencies are unavailable") from error
    try:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is not None:
            with safe_globals([SatViewStage]):
                payload = torch.load(source, map_location="cpu", weights_only=True)
            policy = "weights_only_with_explicit_safe_globals"
            weights_only = True
        else:
            try:
                payload = torch.load(source, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(source, map_location="cpu")
            policy = "legacy_pickle_exact_frozen_sha_only"
            weights_only = False
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise D105Phase1BundleError("exact SHA-bound checkpoint load failed") from error
    if not isinstance(payload, Mapping):
        raise D105Phase1BundleError("checkpoint load did not return a mapping")
    return payload, {
        "policy": policy,
        "torch_version": str(torch.__version__),
        "safe_globals_available": safe_globals is not None,
        "weights_only": weights_only,
        "exact_frozen_checkpoint_sha256_required": expected,
        "caller_selected_checkpoint_allowed": False,
    }


def build_d105_exact_model_from_checkpoint(
    checkpoint: Mapping[str, Any], *, input_len: int, device: Any
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct the frozen ADV3B02 model without importing SSDG training.

    The only executable local model dependency is the checked-in CVSincNet
    factory.  The runtime manifest lists both files, and the origin checks
    below fail closed if an alternate ``model_modified`` or path-injected
    factory would otherwise be selected.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in deployment.
        raise D105Phase1BundleError("D105 checkpoint reconstruction requires PyTorch") from error
    if not isinstance(checkpoint, Mapping):
        raise D105Phase1BundleError("checkpoint must be a mapping")
    args = checkpoint.get("args")
    raw_state = checkpoint.get("model")
    if type(args) is not dict or not isinstance(raw_state, Mapping):
        raise D105Phase1BundleError("checkpoint requires exact args/model fields")
    state: dict[str, Any] = {}
    for raw_key, value in raw_state.items():
        key = str(raw_key)
        if not key or not torch.is_tensor(value):
            raise D105Phase1BundleError("checkpoint model state tensor contract drift")
        normalized = key[7:] if key.startswith("module.") else key
        if not normalized or normalized in state:
            raise D105Phase1BundleError("checkpoint model-state key normalization drift")
        state[normalized] = value
    num_domains = 0
    for name in (
        "dom_head.net.3.bias",
        "dom_head.net.3.weight",
        "adv_head.net.3.bias",
        "adv_head.net.3.weight",
    ):
        value = state.get(name)
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) > 0:
            num_domains = int(value.shape[0])
            break
    if num_domains <= 0:
        raise D105Phase1BundleError("cannot infer checkpoint domain count")
    kwargs = _d105_reconstruction_model_kwargs(
        args, input_len=int(input_len), num_domains=num_domains
    )
    try:
        import model_dual_cvsincnet as model_factory_module
        import model as model_backbone_module
    except ImportError as error:
        raise D105Phase1BundleError(
            "D105 CVSincNet model factory is unavailable"
        ) from error
    code_root = Path(__file__).resolve().parents[1]
    expected_factory = (code_root / "model_dual_cvsincnet.py").resolve()
    expected_backbone = (code_root / "model.py").resolve()
    if (
        Path(str(getattr(model_factory_module, "__file__", ""))).resolve()
        != expected_factory
        or Path(str(getattr(model_backbone_module, "__file__", ""))).resolve()
        != expected_backbone
        or getattr(model_factory_module.build_dual_model, "__module__", None)
        != "model_dual_cvsincnet"
        or getattr(model_factory_module.build_single_model, "__module__", None)
        != "model"
    ):
        raise D105Phase1BundleError("D105 model factory origin drift")
    try:
        model = model_factory_module.build_dual_model(**kwargs).to(device)
        incompatible = model.load_state_dict(state, strict=False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise D105Phase1BundleError(
            "strict D105 checkpoint reconstruction shape mismatch"
        ) from error
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise D105Phase1BundleError(
            "strict D105 checkpoint reconstruction failed: "
            f"missing={missing} unexpected={unexpected}"
        )
    model.eval()
    return model, {
        "loader": "d105_minimal_cvsincnet_checkpoint_reconstruction_v1",
        "model_factory": "model_dual_cvsincnet.build_dual_model",
        "backbone_factory": "model.build_model",
        "checkpoint_load_strict": True,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "skipped_mismatch": 0,
        "state_tensor_count": len(state),
        "num_domains_from_state": num_domains,
        "input_len": int(input_len),
        "eval_mode": True,
    }


def load_d105_tap_cache_selection_salt(
    path: str | Path,
    expected_sha256: str,
    *,
    checkpoint_sha256: str,
) -> dict[str, str]:
    """Read the frozen one-observation salt without importing old exporters."""

    source = Path(path)
    expected = _require_sha256(expected_sha256, "selection salt receipt SHA256")
    checkpoint = _require_sha256(checkpoint_sha256, "checkpoint SHA256")
    if (
        source.is_symlink()
        or not source.is_file()
        or sha256_file(source) != expected
    ):
        raise D105Phase1BundleError("selection salt receipt path/SHA256 drift")
    try:
        receipt = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105Phase1BundleError(
            "selection salt receipt must be strict UTF-8 JSON"
        ) from error
    expected_keys = {
        "schema",
        "status",
        "artifact_stage",
        "bundle_id",
        "phase1_checkpoint_sha256",
        "selection_salt_sha256",
        "target_access",
    }
    if (
        type(receipt) is not dict
        or set(receipt) != expected_keys
        or receipt.get("schema") != D105_TAP_CACHE_SELECTION_SALT_SCHEMA
        or receipt.get("status") != "SEALED_BEFORE_TARGET_ACCESS"
        or receipt.get("artifact_stage") != "phase1_offline_before_target_access"
        or _require_sha256(receipt.get("bundle_id"), "selection-salt bundle_id")
        != str(receipt.get("bundle_id"))
        or receipt.get("phase1_checkpoint_sha256") != checkpoint
        or receipt.get("target_access") is not False
    ):
        raise D105Phase1BundleError("selection salt receipt lineage drift")
    return {
        "path": str(source.resolve()),
        "sha256": expected,
        "selection_salt_sha256": _require_sha256(
            receipt.get("selection_salt_sha256"), "selection salt"
        ),
    }


def load_d105_tap_cache_source_validation_set(
    path: str | Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    """Load only the historical v1 source-validation cache lineage."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D105Phase1BundleError("source cache set must be a regular file")
    try:
        return load_verified_leo_weak_cache_set(
            source,
            expected_scope="source_validation",
            allowed_roles={"source"},
            accepted_outer_schemas=frozenset({LEO_WEAK_CACHE_SET_SCHEMA_V1}),
            accepted_inner_schemas=frozenset({LEO_WEAK_CACHE_SCHEMA_V1}),
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise D105Phase1BundleError(
            "verified v1 source-validation cache-set validation failed"
        ) from error


def _d105_tap_cache_selection_index(
    selection_salt_sha256: str, physical_id: str
) -> int:
    salt = bytes.fromhex(_require_sha256(selection_salt_sha256, "selection salt"))
    identifier = str(physical_id)
    if not identifier:
        raise D105Phase1BundleError("physical_id must be nonempty")
    digest = hashlib.sha256(
        D105_TAP_CACHE_SELECTION_DOMAIN + salt + identifier.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % len(
        FORMAL_LEO_WEAK_SCENARIOS
    )


def select_d105_tap_cache_observations(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    selection_salt_sha256: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Select the one fixed weak observation per source physical sample.

    This is deliberately byte-compatible with the historical helper for a
    valid v1 source cache, while making the logic part of the D105 runtime
    manifest rather than an unsealed export script.
    """

    if tuple(arrays_by_scenario) != FORMAL_LEO_WEAK_SCENARIOS:
        raise D105Phase1BundleError("all three ordered scenarios are required")
    required = {
        "leo_weak_iq",
        "sample_ids",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "dataset_role",
        "sat_scenarios",
        "overlay_ids",
    }
    indexes: dict[str, dict[str, int]] = {}
    ids: dict[str, list[str]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        missing = required - set(arrays)
        if missing:
            raise D105Phase1BundleError(
                f"verified cache lacks fields: {sorted(missing)}"
            )
        sample_ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
        iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)
        if (
            not sample_ids
            or len(sample_ids) != len(set(sample_ids))
            or iq.ndim != 3
            or iq.shape[1] != 2
            or len(iq) != len(sample_ids)
            or not np.isfinite(iq).all()
        ):
            raise D105Phase1BundleError(
                f"verified cache row contract drift: {scenario}"
            )
        if any(
            len(np.asarray(arrays[name])) != len(sample_ids)
            for name in required - {"leo_weak_iq"}
        ):
            raise D105Phase1BundleError(
                f"verified cache row count drift: {scenario}"
            )
        if np.asarray(arrays["sat_scenarios"]).astype(str).tolist() != [
            scenario
        ] * len(sample_ids):
            raise D105Phase1BundleError(
                f"verified cache scenario drift: {scenario}"
            )
        overlay_ids = np.asarray(arrays["overlay_ids"]).astype(str).tolist()
        if any(not value for value in overlay_ids) or len(overlay_ids) != len(
            set(overlay_ids)
        ):
            raise D105Phase1BundleError(f"verified overlay_ids drift: {scenario}")
        ids[scenario] = sample_ids
        indexes[scenario] = {value: index for index, value in enumerate(sample_ids)}
    reference = ids[FORMAL_LEO_WEAK_SCENARIOS[0]]
    if any(
        set(ids[scenario]) != set(reference)
        for scenario in FORMAL_LEO_WEAK_SCENARIOS[1:]
    ):
        raise D105Phase1BundleError(
            "cache scenarios do not share one selectable physical-ID set"
        )
    metadata: dict[str, list[str]] = {
        name: []
        for name in (
            "labels",
            "receiver_ids",
            "day_ids",
            "physical_ids",
            "scenario_names",
            "observation_ids",
        )
    }
    selected_iq: list[np.ndarray] = []
    for physical_id in reference:
        identities: list[tuple[str, str, str]] = []
        roles: list[str] = []
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            arrays = arrays_by_scenario[scenario]
            index = indexes[scenario][physical_id]
            identities.append(
                (
                    str(arrays["tx_ids"][index]),
                    str(arrays["rx_ids"][index]),
                    str(arrays["day_ids"][index]),
                )
            )
            roles.append(str(arrays["dataset_role"][index]))
        if len(set(identities)) != 1 or set(roles) != {"source"}:
            raise D105Phase1BundleError(
                f"physical identity/role drift: {physical_id}"
            )
        scenario = FORMAL_LEO_WEAK_SCENARIOS[
            _d105_tap_cache_selection_index(selection_salt_sha256, physical_id)
        ]
        index = indexes[scenario][physical_id]
        arrays = arrays_by_scenario[scenario]
        metadata["labels"].append(identities[0][0])
        metadata["receiver_ids"].append(identities[0][1])
        metadata["day_ids"].append(identities[0][2])
        metadata["physical_ids"].append(physical_id)
        metadata["scenario_names"].append(scenario)
        metadata["observation_ids"].append(str(arrays["overlay_ids"][index]))
        selected_iq.append(np.asarray(arrays["leo_weak_iq"][index], dtype=np.float32))
    if len(metadata["observation_ids"]) != len(set(metadata["observation_ids"])):
        raise D105Phase1BundleError("selected observation IDs are not unique")
    return (
        {key: np.asarray(value, dtype=np.str_) for key, value in metadata.items()},
        np.ascontiguousarray(np.stack(selected_iq), dtype=np.float32),
    )


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or text != text.lower() or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise D105Phase1BundleError(f"{name} must be a lowercase SHA256")
    return text


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise D105Phase1BundleError("object arrays are forbidden")
    if array.dtype.kind in {"U", "S"}:
        descriptor = {"dtype": "utf8-string", "shape": list(array.shape)}
        payload = _canonical_bytes(array.astype(str).tolist())
    else:
        canonical = np.ascontiguousarray(array)
        descriptor = {"dtype": canonical.dtype.str, "shape": list(canonical.shape)}
        payload = canonical.tobytes(order="C")
    return _sha256_bytes(_canonical_bytes(descriptor) + b"\0" + payload)


def _physical_root(values: Sequence[str]) -> str:
    canonical = tuple(str(item) for item in values)
    if not canonical or any(not item for item in canonical) or len(set(canonical)) != len(
        canonical
    ):
        raise D105Phase1BundleError(
            "physical IDs must be non-empty, unique, and non-blank"
        )
    return _sha256_bytes(_canonical_bytes(sorted(canonical)))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        copied = np.ascontiguousarray(value).copy()
        copied.setflags(write=False)
        return copied
    if isinstance(value, np.generic):
        return value.item()
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D105Phase1BundleError(f"{name} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105Phase1BundleError(f"{name} is not valid UTF-8 JSON") from error
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise D105Phase1BundleError(f"{name} must be canonical JSON")
    return value


def _load_immutable_json(path: str | Path, *, name: str) -> dict[str, Any]:
    """Read a canonical JSON artifact only after its write bits are closed."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D105Phase1BundleError(f"{name} must be a regular non-symlink file")
    mode = source.stat().st_mode
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Phase1BundleError(f"{name} must be read-only before truth open")
    return _load_json(source, name=name)


def _write_new_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(value)
    _write_new(path, payload)
    os.chmod(path, stat.S_IREAD)
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Phase1BundleError("immutable JSON output remains writable")
    return _sha256_bytes(payload)


def load_d105_candidate_runtime_manifest(
    path: str | Path, *, expected_checkpoint_sha256: str | None = None
) -> dict[str, Any]:
    """Load the canonical D105 implementation/runtime manifest by contents."""

    manifest = _load_json(path, name="D105 candidate runtime manifest")
    _require_exact_keys(
        manifest,
        _D105_CANDIDATE_RUNTIME_MANIFEST_FIELDS,
        "D105 candidate runtime manifest",
    )
    expected_entrypoints = D105_CANDIDATE_RUNTIME_ENTRYPOINTS
    expected_files = set(D105_CANDIDATE_RUNTIME_FILES)
    if (
        manifest["schema"] != D105_CANDIDATE_RUNTIME_MANIFEST_SCHEMA
        or manifest["candidate_id"] != CANDIDATE_ID
        or manifest["protocol_schema"] != PROTOCOL_SCHEMA
        or manifest["entrypoints"] != expected_entrypoints
        or type(manifest["core_file_sha256"]) is not dict
        or set(manifest["core_file_sha256"]) != expected_files
    ):
        raise D105Phase1BundleError("D105 candidate runtime manifest identity drift")
    _require_sha256(manifest["checkpoint_sha256"], "checkpoint_sha256")
    if (
        expected_checkpoint_sha256 is not None
        and manifest["checkpoint_sha256"]
        != _require_sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    ):
        raise D105Phase1BundleError("D105 candidate runtime manifest checkpoint drift")
    code_root = Path(__file__).resolve().parents[1]
    observed_core_file_sha256: dict[str, str] = {}
    for relative_path in sorted(expected_files):
        expected_sha = _require_sha256(
            manifest["core_file_sha256"][relative_path],
            "candidate runtime core file SHA256",
        )
        core_file = code_root / relative_path
        if not core_file.is_file() or core_file.is_symlink():
            raise D105Phase1BundleError(
                "D105 candidate runtime core file is missing or symbolic"
            )
        observed_sha = sha256_file(core_file)
        observed_core_file_sha256[relative_path] = observed_sha
        if observed_sha != expected_sha:
            raise D105Phase1BundleError(
                "D105 candidate runtime core file SHA256 drift"
            )
    authority_helper = code_root / "cvsrffi/stage2_d105_phase1_authority.py"
    if (
        not authority_helper.is_file()
        or authority_helper.is_symlink()
        or sha256_file(authority_helper) != D105_PHASE1_AUTHORITY_MODULE_SHA256
    ):
        raise D105Phase1BundleError("D105 authority helper SHA256 drift")
    _reject_d102(manifest, name="D105 candidate runtime manifest")
    return {
        "manifest": manifest,
        "d105_candidate_runtime_manifest_sha256": _sha256_bytes(
            _canonical_bytes(manifest)
        ),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "observed_core_file_sha256": observed_core_file_sha256,
    }


def load_d105_candidate_method_lock(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_runtime_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the one canonical D105 candidate lock; data locks are rejected.

    ``runtime_sha256`` in the Phase1/asset chain denotes the D105 candidate
    implementation/runtime-manifest identity, not the historical D92 data
    materialization runtime.  The exact nested closure below prevents a caller
    from replacing the lock by an arbitrary SHA-only file.
    """

    lock = _load_json(path, name="D105 candidate method lock")
    _require_exact_keys(lock, _D105_CANDIDATE_METHOD_LOCK_FIELDS, "D105 candidate method lock")
    if (
        lock["schema"] != D105_CANDIDATE_METHOD_LOCK_SCHEMA
        or lock["candidate_id"] != CANDIDATE_ID
        or lock["protocol_schema"] != PROTOCOL_SCHEMA
    ):
        raise D105Phase1BundleError("D105 candidate method lock identity drift")
    for field in ("checkpoint_sha256", "d105_candidate_runtime_manifest_sha256"):
        _require_sha256(lock[field], field)
    if (
        expected_checkpoint_sha256 is not None
        and lock["checkpoint_sha256"]
        != _require_sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    ):
        raise D105Phase1BundleError("D105 candidate method lock checkpoint drift")
    if (
        expected_runtime_sha256 is not None
        and lock["d105_candidate_runtime_manifest_sha256"]
        != _require_sha256(expected_runtime_sha256, "expected_runtime_sha256")
    ):
        raise D105Phase1BundleError("D105 candidate method lock runtime drift")
    expected_cbrc = {
        "semantic_revision": "cbrc_mb4_task_balanced_huber_irls4_fp16_v1",
        "code_dim": CODE_DIM,
        "domain_dim": DOMAIN_DIM,
        "allowed_k": [1, 5, 10],
        "irls_steps": 4,
        "old_new_task_mass": [0.5, 0.5],
        "k1_zero_coefficient": True,
        "ground_old_multiprototype_enabled": False,
        "deployment_coefficient_dtype": "float16",
        "query_transform": "relu_l2norm_pre_relu_plus_mb4",
        "query_state_updates": 0,
    }
    expected_student = {
        "student_nu": 3.0,
        "kernel_effective_dim": 12,
        "kernel_volume_gamma": 1.0,
        "shared_h0": 0.35,
        "scale_prior_strength": 2.0,
        "scale_min_ratio": 0.5,
        "scale_max_ratio": 2.0,
        "temperature": 0.85,
        "support_storage": "int8_fp16_scale",
    }
    expected_four_arm = {
        "arms": ["M0", "M_DA", "M_HEAD", "M_JOINT"],
        "same_da_state_for_da_and_joint": True,
        "same_head_code_config_for_head_and_joint": True,
        "query_truth_surface": False,
    }
    expected_held = {
        "receiver_held_k": [1, 5, 10],
        "class_loco_k": 1,
        "tx_probe_algorithm": "receiver_held_ridge_l2_0.01",
        "tx_probe_max_balanced_accuracy": 0.25,
        "int8_min_top1_agreement": 0.995,
        "large_margin_minimum": _LARGE_MARGIN_MINIMUM,
        "large_margin_flip_max": 0,
        "truth_open_after_prediction": True,
    }
    expected_target25 = {
        "seed": 713102,
        "claim_scope": "DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE",
        "formal_launch_authority": False,
        "slices": [[10, 5], [10, 10], [10, 20], [5, 20], [1, 20]],
        "leo_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "outer_row_count": 25,
        "scenario_arm_pair_count": 300,
        "state_prediction_surface_count": 600,
    }
    if (
        lock["d105_cbrc"] != expected_cbrc
        or lock["student_t_qknn"] != expected_student
        or lock["four_arm"] != expected_four_arm
        or lock["source_held"] != expected_held
        or lock["target25"] != expected_target25
    ):
        raise D105Phase1BundleError("D105 candidate method lock implementation closure drift")
    return {
        "lock": lock,
        "d105_candidate_method_lock_sha256": _sha256_bytes(_canonical_bytes(lock)),
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "runtime_sha256": lock["d105_candidate_runtime_manifest_sha256"],
    }


def _validate_candidate_lock_for_tap(
    candidate_method_lock: str | Path,
    candidate_runtime_manifest: str | Path,
    tap: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = load_d105_candidate_runtime_manifest(
        candidate_runtime_manifest,
        expected_checkpoint_sha256=str(tap["checkpoint_sha256"]),
    )
    if runtime["d105_candidate_runtime_manifest_sha256"] != tap["runtime_sha256"]:
        raise D105Phase1BundleError("strict tap candidate runtime manifest SHA drift")
    loaded = load_d105_candidate_method_lock(
        candidate_method_lock,
        expected_checkpoint_sha256=str(tap["checkpoint_sha256"]),
        expected_runtime_sha256=runtime["d105_candidate_runtime_manifest_sha256"],
    )
    if loaded["d105_candidate_method_lock_sha256"] != tap["method_lock_sha256"]:
        raise D105Phase1BundleError("strict tap candidate method lock SHA drift")
    return loaded


def _reject_d102(value: Any, *, name: str) -> None:
    """Reject obvious stale labels in addition to signed-content revocation.

    This is only a defensive spelling check for legacy artifacts.  Formal D105
    admission never relies on it: ``_load_d102_revocation`` verifies a pinned,
    signed immutable-content list and ``_reject_revoked_identity`` compares
    hashes, so renaming an old D102 file cannot bypass the boundary.
    """

    forbidden = (
        "d102",
        "rb-metabias4",
        "rb_metabias4",
        "analytic_initializer",
        "phase1_rb_metabias4",
    )

    def check(item: Any, field: str | None = None) -> None:
        if isinstance(item, Mapping):
            for key, member in item.items():
                check(member, str(key))
            return
        if isinstance(item, (tuple, list)):
            for member in item:
                check(member, field)
            return
        if field == "d102_rejected_bundle_reused":
            if item is not False:
                raise D105Phase1BundleError(
                    f"{name} declares rejected D102 bundle reuse"
                )
            return
        if isinstance(item, str):
            # Content-addressed identities are arbitrary hexadecimal values:
            # their bytes can coincidentally contain the textual token
            # ``d102``.  The signed revocation path below owns content-based
            # rejection; this legacy spelling guard applies only to labels.
            if (
                len(item) == 64
                and item == item.lower()
                and all(char in "0123456789abcdef" for char in item)
            ):
                return
            if any(token in item.lower() for token in forbidden):
                raise D105Phase1BundleError(
                    f"{name} refers to a rejected D102 lineage"
                )

    check(value)


def _load_d102_revocation(
    manifest_path: str | Path,
    signature_path: str | Path,
) -> dict[str, Any]:
    try:
        return load_signed_d102_revocation_manifest(manifest_path, signature_path)
    except D105AuthorityError as error:
        raise D105Phase1BundleError("signed D102 revocation validation failed") from error


def _reject_revoked_identity(
    revocation: Mapping[str, Any],
    **identity: str | None,
) -> None:
    try:
        reject_revoked_d102_identity(revocation["manifest"], **identity)
    except (D105AuthorityError, KeyError, TypeError) as error:
        raise D105Phase1BundleError(
            "D102 signed revocation rejects immutable content identity"
        ) from error


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise D105Phase1BundleError(f"{name} exact schema drift")


_STRICT_TAP_FIELDS = {
    "schema",
    "candidate_id",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "source_access_receipt_sha256",
    "tap_archive_sha256",
    "tap_archive_members",
    "row_count",
    "forward_batch_capacity",
    "forward_invocation_count",
    "last_batch_real_rows",
    "last_batch_padding_rows",
    "forward_batch_policy",
    "pre_relu_sha256",
    "z_dom_sha256",
    "physical_id_root_sha256",
    "d102_revocation_manifest_sha256",
    "execution_path",
    "hook_exact_bytes",
    "strict_pre_relu_path",
    "zid_relu_parity_verified",
    "z_dom_present",
    "source_only",
    "target_rows",
    "query_rows",
    "raw_iq_retained",
    "clean_iq_retained",
    "source_archive_phase1_only",
    "d102_rejected_bundle_reused",
}


_SOURCE_HELD_PREDICTION_FIELDS = {
    "schema",
    "candidate_id",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "strict_tap_receipt_sha256",
    "source_aggregate_lineage_sha256",
    "source_only",
    "target_rows",
    "query_rows",
    "receiver_tokens",
    "class_tokens",
    "scored_prediction_rows",
    "tx_probe_prediction_rows",
    "d102_rejected_bundle_reused",
}


_SOURCE_HELD_TRUTH_OPEN_FIELDS = {
    "schema",
    "candidate_id",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "strict_tap_receipt_sha256",
    "source_aggregate_lineage_sha256",
    "source_held_prediction_manifest_sha256",
    "prediction_manifest_immutable",
    "truth_opened_after_prediction",
    "source_only",
    "target_rows",
    "query_rows",
    "d102_rejected_bundle_reused",
}


_SOURCE_HELD_SCORE_FIELDS = {
    "schema",
    "candidate_id",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "strict_tap_receipt_sha256",
    "source_aggregate_lineage_sha256",
    "source_held_prediction_manifest_sha256",
    "source_held_truth_open_receipt_sha256",
    "source_only",
    "target_rows",
    "query_rows",
    "scored_truth_rows",
    "tx_probe_truth_rows",
    "d102_rejected_bundle_reused",
}


_PREDICTION_ROW_FIELDS = {
    "row_id",
    "fold_kind",
    "held_receiver_token",
    "held_class_token",
    "K",
    "query_physical_ids",
    "m0_predictions",
    "d105_fp32_predictions",
    "d105_int8_predictions",
    "d105_fp32_top2_margins",
    "prediction_commit_sha256",
    "query_rows_used_for_fit",
}


_TX_PREDICTION_ROW_FIELDS = {
    "held_receiver_token",
    "physical_ids",
    "predictions",
    "prediction_commit_sha256",
}


_TRUTH_ROW_FIELDS = {"row_id", "truth_labels"}
_TX_TRUTH_ROW_FIELDS = {"held_receiver_token", "truth_labels"}


_SOURCE_HELD_GATE_FIELDS = {
    "schema",
    "candidate_id",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "strict_tap_receipt_sha256",
    "source_aggregate_lineage_sha256",
    "source_held_prediction_manifest_sha256",
    "source_held_truth_open_receipt_sha256",
    "source_held_score_artifact_sha256",
    "source_only",
    "target_rows",
    "query_rows",
    "receiver_count",
    "class_count",
    "receiver_held_complete",
    "receiver_held_k",
    "receiver_held_all_noninferior",
    "receiver_held_row_count",
    "receiver_held_failing_row_count",
    "receiver_held_worst_ba_delta",
    "receiver_held_worst_floor_delta",
    "receiver_held_min_net_correct",
    "class_loco_complete",
    "class_loco_all_noninferior",
    "class_loco_row_count",
    "class_loco_failing_row_count",
    "class_loco_worst_ba_delta",
    "class_loco_worst_floor_delta",
    "class_loco_min_net_correct",
    "tx_probe_mean_balanced_accuracy",
    "tx_probe_max_balanced_accuracy",
    "tx_probe_gate_pass",
    "quantization_top1_agreement",
    "quantization_large_margin_flip_count",
    "quantization_gate_pass",
    "persistent_fp32_sidecar",
    "raw_iq_persisted",
    "source_replay_persisted",
    "d102_rejected_bundle_reused",
}


_LARGE_MARGIN_MINIMUM = 0.10


_AUTHORITY_SEAL_FIELDS = {
    # Authority field closure is implemented in stage2_d105_phase1_authority.
    # Keep this symbol private/empty so stale v1 callers cannot accidentally
    # obtain a permissive in-module receipt schema.
}


_SOURCE_ACCESS_FIELDS = {
    "schema",
    "candidate_id",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "source_iq_sha256",
    "source_labels_sha256",
    "source_receiver_ids_sha256",
    "source_physical_id_root_sha256",
    "d102_revocation_manifest_sha256",
    "source_only",
    "target_rows",
    "query_rows",
    "received_iq_persisted",
    "raw_iq_persisted",
    "clean_iq_persisted",
    "d102_rejected_bundle_reused",
}


@dataclass(frozen=True, slots=True)
class StrictTapRows:
    """Transient Phase1 rows; instances must never be saved in a bundle."""

    pre_relu: np.ndarray
    z_dom: np.ndarray
    labels: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    physical_ids: tuple[str, ...]
    archive_sha256: str
    strict_tap_receipt_sha256: str

    def __post_init__(self) -> None:
        pre = np.asarray(self.pre_relu)
        domain = np.asarray(self.z_dom)
        rows = len(pre)
        if (
            pre.dtype != np.float32
            or domain.dtype != np.float32
            or pre.shape != (rows, Z_DIM)
            or domain.shape != (rows, Z_DIM)
            or rows < DOMAIN_DIM + 2
            or not np.isfinite(pre).all()
            or not np.isfinite(domain).all()
        ):
            raise D105Phase1BundleError("strict tap feature rows drift")
        labels = tuple(str(item) for item in self.labels)
        receivers = tuple(str(item) for item in self.receiver_ids)
        physical = tuple(str(item) for item in self.physical_ids)
        if (
            len(labels) != rows
            or len(receivers) != rows
            or len(physical) != rows
            or any(not item for item in labels + receivers + physical)
            or len(set(physical)) != rows
        ):
            raise D105Phase1BundleError("strict tap text/physical ID closure drift")
        for field in ("archive_sha256", "strict_tap_receipt_sha256"):
            _require_sha256(getattr(self, field), field)
        pre_copy = np.ascontiguousarray(pre, dtype=np.float32).copy()
        dom_copy = np.ascontiguousarray(domain, dtype=np.float32).copy()
        pre_copy.setflags(write=False)
        dom_copy.setflags(write=False)
        object.__setattr__(self, "pre_relu", pre_copy)
        object.__setattr__(self, "z_dom", dom_copy)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "receiver_ids", receivers)
        object.__setattr__(self, "physical_ids", physical)


@dataclass(frozen=True, slots=True)
class D105Phase1Asset:
    """Loaded aggregate-only asset, with an optional independent formal seal."""

    bundle: RXIDMetaBias4Bundle
    manifest: Mapping[str, Any]
    manifest_sha256: str
    formal_phase2_eligible: bool
    validated_bundle_id_sha256: str | None = None
    validator_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.bundle) is not RXIDMetaBias4Bundle:
            raise D105Phase1BundleError("asset requires an exact RXID bundle")
        manifest = _thaw(self.manifest)
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        if _sha256_bytes(_canonical_bytes(manifest) + b"\n") != self.manifest_sha256:
            raise D105Phase1BundleError("asset manifest hash drift")
        if bool(manifest.get("formal_phase2_eligible")) != bool(
            self.formal_phase2_eligible
        ):
            raise D105Phase1BundleError("asset formal status drift")
        if self.formal_phase2_eligible:
            _require_sha256(
                self.validated_bundle_id_sha256, "validated_bundle_id_sha256"
            )
            _require_sha256(
                self.validator_receipt_sha256, "validator_receipt_sha256"
            )
        elif (
            self.validated_bundle_id_sha256 is not None
            or self.validator_receipt_sha256 is not None
        ):
            raise D105Phase1BundleError(
                "unsealed component must not expose a runtime validator identity"
            )
        object.__setattr__(self, "manifest", _freeze(manifest))


@dataclass(frozen=True, slots=True)
class _D105StrictForward:
    """Phase1-compatible view of the D105-only dual-backbone feature tap."""

    z_id: np.ndarray
    hidden: np.ndarray
    pre_relu: np.ndarray
    hook_exact_bytes: bool
    z_dom: np.ndarray
    execution_path: str


def _strict_forward(model: Any, received_iq: Any) -> _D105StrictForward:
    """Run the D105-authoritative same-IQ dual-backbone strict tap.

    D105 needs the domain feature produced by
    ``dom_backbone.feat_imp -> dom_enhancer(feat_imp, received_iq)``.  The
    older GRB helper invokes only the identity backbone for an eager model and
    therefore cannot satisfy this Phase1 boundary.  This is a single strict
    path: a D105 contract failure is surfaced as an error and never falls back
    to a legacy or identity-only forward.
    """

    from cvsrffi.dual_feature_forward import DualFeatureForwardError
    from cvsrffi.stage2_d105_feature_tap import (
        D105FeatureTapError,
        extract_d105_feature_tap,
    )

    try:
        tap = extract_d105_feature_tap(model, received_iq)
    except (D105FeatureTapError, DualFeatureForwardError, RuntimeError, TypeError) as error:
        raise D105Phase1BundleError(
            "D105 strict dual-backbone feature tap failed"
        ) from error
    return _D105StrictForward(
        z_id=tap.z_id,
        hidden=tap.hidden,
        pre_relu=tap.pre_relu,
        hook_exact_bytes=True,
        z_dom=tap.z_dom,
        # Keep the persisted receipt vocabulary stable while its implementation
        # remains the D105 eager forward-hook path.
        execution_path="eager_forward_hook",
    )


def _strict_tap_float32_rows(
    forward: Any, field: str, *, rows: int
) -> np.ndarray:
    """Read one strict-tap output with deterministic, field-level failures."""

    value = np.asarray(getattr(forward, field, None))
    if value.dtype != np.float32:
        raise D105Phase1BundleError(f"strict tap {field} must use float32")
    if value.shape != (rows, Z_DIM):
        raise D105Phase1BundleError(
            f"strict tap {field} must have shape [N,{Z_DIM}]"
        )
    if not np.isfinite(value).all():
        raise D105Phase1BundleError(f"strict tap {field} must be finite")
    return value


def _validate_d105_strict_forward(
    forward: Any, *, rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Fail closed on every Phase1 strict-tap contract field separately."""

    if getattr(forward, "hook_exact_bytes", None) is not True:
        raise D105Phase1BundleError("strict tap hook_exact_bytes must be true")
    z_id = _strict_tap_float32_rows(forward, "z_id", rows=rows)
    z_dom = _strict_tap_float32_rows(forward, "z_dom", rows=rows)
    pre_relu = _strict_tap_float32_rows(forward, "pre_relu", rows=rows)
    if not np.array_equal(z_id, np.maximum(pre_relu, np.float32(0.0))):
        raise D105Phase1BundleError("strict tap z_id/pre_relu ReLU binding drift")
    execution_path = str(getattr(forward, "execution_path", ""))
    if execution_path not in (
        "torchscript_exported_functional_tap",
        "eager_forward_hook",
    ):
        raise D105Phase1BundleError("strict tap execution path is unrecognized")
    return z_id, z_dom, pre_relu, execution_path


def _validate_source_access_receipt(
    receipt: Mapping[str, Any],
    *,
    source_iq: np.ndarray,
    labels: np.ndarray,
    receiver_ids: np.ndarray,
    physical_ids: np.ndarray,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    d102_revocation_manifest_sha256: str,
) -> str:
    _require_exact_keys(receipt, _SOURCE_ACCESS_FIELDS, "source access receipt")
    _reject_d102(receipt, name="source access receipt")
    if (
        receipt["schema"] != SOURCE_ACCESS_SCHEMA
        or receipt["candidate_id"] != CANDIDATE_ID
        or receipt["checkpoint_sha256"] != checkpoint_sha256
        or receipt["runtime_sha256"] != runtime_sha256
        or receipt["method_lock_sha256"] != method_lock_sha256
        or receipt["d102_revocation_manifest_sha256"]
        != d102_revocation_manifest_sha256
        or receipt["source_iq_sha256"] != _array_sha256(source_iq)
        or receipt["source_labels_sha256"] != _array_sha256(labels)
        or receipt["source_receiver_ids_sha256"] != _array_sha256(receiver_ids)
        or receipt["source_physical_id_root_sha256"]
        != _physical_root(physical_ids.astype(str).tolist())
        or receipt["source_only"] is not True
        or receipt["target_rows"] != 0
        or receipt["query_rows"] != 0
        or receipt["received_iq_persisted"] is not False
        or receipt["raw_iq_persisted"] is not False
        or receipt["clean_iq_persisted"] is not False
        or receipt["d102_rejected_bundle_reused"] is not False
    ):
        raise D105Phase1BundleError("source access receipt binding/lifecycle drift")
    for field in (
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "source_iq_sha256",
        "source_labels_sha256",
        "source_receiver_ids_sha256",
        "source_physical_id_root_sha256",
        "d102_revocation_manifest_sha256",
    ):
        _require_sha256(receipt[field], field)
    return _sha256_bytes(_canonical_bytes(receipt))


def build_d105_source_access_receipt(
    *,
    source_received_iq: np.ndarray,
    source_labels: Sequence[str],
    source_receiver_ids: Sequence[str],
    source_physical_ids: Sequence[str],
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    d102_revocation_manifest_sha256: str,
) -> dict[str, Any]:
    """Create the hash-only receipt consumed by a source-only strict tap."""

    iq = np.asarray(source_received_iq)
    labels = np.asarray(tuple(str(item) for item in source_labels), dtype=np.str_)
    receivers = np.asarray(
        tuple(str(item) for item in source_receiver_ids), dtype=np.str_
    )
    physical = np.asarray(
        tuple(str(item) for item in source_physical_ids), dtype=np.str_
    )
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or len(iq) < 1
        or not np.isfinite(iq).all()
        or labels.shape != (len(iq),)
        or receivers.shape != (len(iq),)
        or physical.shape != (len(iq),)
        or any(not item for item in labels.tolist() + receivers.tolist() + physical.tolist())
    ):
        raise D105Phase1BundleError("source access receipt input closure drift")
    _physical_root(physical.tolist())
    return {
        "schema": SOURCE_ACCESS_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": _require_sha256(
            checkpoint_sha256, "checkpoint_sha256"
        ),
        "runtime_sha256": _require_sha256(runtime_sha256, "runtime_sha256"),
        "method_lock_sha256": _require_sha256(
            method_lock_sha256, "method_lock_sha256"
        ),
        "source_iq_sha256": _array_sha256(iq),
        "source_labels_sha256": _array_sha256(labels),
        "source_receiver_ids_sha256": _array_sha256(receivers),
        "source_physical_id_root_sha256": _physical_root(physical.tolist()),
        "d102_revocation_manifest_sha256": _require_sha256(
            d102_revocation_manifest_sha256, "d102_revocation_manifest_sha256"
        ),
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "received_iq_persisted": False,
        "raw_iq_persisted": False,
        "clean_iq_persisted": False,
        "d102_rejected_bundle_reused": False,
    }


def export_d105_phase1_strict_tap(
    *,
    model: Any,
    source_received_iq: np.ndarray,
    source_labels: Sequence[str],
    source_receiver_ids: Sequence[str],
    source_physical_ids: Sequence[str],
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    source_access_receipt: str | Path | Mapping[str, Any],
    d102_revocation_manifest: str | Path,
    d102_revocation_signature: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = D105_STRICT_TAP_FORWARD_BATCH_CAPACITY,
) -> dict[str, Any]:
    """Export a strict source-only D105 tap from a real frozen checkpoint.

    The received IQ resides only in the caller buffer while forwarding. The
    output archive deliberately contains just the two D105 feature views and
    Phase1 metadata needed for class-balanced aggregation. The strict receipt
    binds the output arrays to the supplied checkpoint, runtime, method lock,
    source-input receipt, and strict hook path. ``batch_size`` remains a
    compatibility input for the older tap-runtime CLI, but cannot change the
    fixed 256-row forward capacity recorded by the receipt.
    """

    checkpoint = _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    runtime = _require_sha256(runtime_sha256, "runtime_sha256")
    method_lock = _require_sha256(method_lock_sha256, "method_lock_sha256")
    revocation = _load_d102_revocation(
        d102_revocation_manifest, d102_revocation_signature
    )
    revocation_sha = _require_sha256(
        revocation["manifest_sha256"], "d102_revocation_manifest_sha256"
    )
    iq = np.asarray(source_received_iq)
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or len(iq) < 1
        or not np.isfinite(iq).all()
    ):
        raise D105Phase1BundleError(
            "source received IQ must be finite float32 [N,2,T]"
        )
    count = len(iq)
    labels = np.asarray(tuple(str(item) for item in source_labels), dtype=np.str_)
    receivers = np.asarray(
        tuple(str(item) for item in source_receiver_ids), dtype=np.str_
    )
    physical = np.asarray(
        tuple(str(item) for item in source_physical_ids), dtype=np.str_
    )
    if (
        labels.shape != (count,)
        or receivers.shape != (count,)
        or physical.shape != (count,)
        or any(not item for item in labels.tolist() + receivers.tolist() + physical.tolist())
        or len(set(physical.tolist())) != count
    ):
        raise D105Phase1BundleError("source tap metadata/physical ID closure drift")
    if (
        type(batch_size) is not int
        or not 1 <= batch_size <= D105_STRICT_TAP_FORWARD_BATCH_CAPACITY
    ):
        raise D105Phase1BundleError("strict tap batch_size must be in [1,256]")
    access = (
        _load_json(source_access_receipt, name="source access receipt")
        if isinstance(source_access_receipt, (str, Path))
        else dict(source_access_receipt)
    )
    access_sha = _validate_source_access_receipt(
        access,
        source_iq=iq,
        labels=labels,
        receiver_ids=receivers,
        physical_ids=physical,
        checkpoint_sha256=checkpoint,
        runtime_sha256=runtime,
        method_lock_sha256=method_lock,
        d102_revocation_manifest_sha256=revocation_sha,
    )
    try:
        import torch
    except ImportError as error:
        raise D105Phase1BundleError("strict tap requires PyTorch") from error
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise D105Phase1BundleError("strict tap device is invalid") from error
    if torch_device.type == "cuda" and (
        not torch.cuda.is_available()
        or torch_device.index is None
        or torch_device.index >= torch.cuda.device_count()
    ):
        raise D105Phase1BundleError("requested strict tap CUDA device is unavailable")
    if bool(getattr(model, "training", True)):
        raise D105Phase1BundleError("strict tap model must be in eval mode")
    pre_rows: list[np.ndarray] = []
    dom_rows: list[np.ndarray] = []
    execution_paths: set[str] = set()
    forward_invocation_count = 0
    last_batch_real_rows = 0
    capacity = D105_STRICT_TAP_FORWARD_BATCH_CAPACITY
    for start in range(0, count, capacity):
        batch = np.ascontiguousarray(iq[start : start + capacity], dtype=np.float32)
        real_rows = len(batch)
        padded = np.zeros((capacity, *batch.shape[1:]), dtype=np.float32)
        padded[:real_rows] = batch
        tensor = _tensor_from_d105_float32_c_iq(
            padded,
            torch_module=torch,
            device=torch_device,
            error_type=D105Phase1BundleError,
            name="strict tap batch",
        )
        forward = _strict_forward(model, tensor)
        z_id, z_dom, pre_relu, execution_path = _validate_d105_strict_forward(
            forward, rows=capacity
        )
        execution_paths.add(execution_path)
        pre_rows.append(np.ascontiguousarray(pre_relu[:real_rows], dtype=np.float32))
        dom_rows.append(np.ascontiguousarray(z_dom[:real_rows], dtype=np.float32))
        forward_invocation_count += 1
        last_batch_real_rows = real_rows
    if len(execution_paths) != 1:
        raise D105Phase1BundleError("strict tap execution path changed across batches")
    arrays = {
        "pre_relu": np.ascontiguousarray(np.concatenate(pre_rows), dtype=np.float32),
        "z_dom": np.ascontiguousarray(np.concatenate(dom_rows), dtype=np.float32),
        "labels": labels,
        "receiver_ids": receivers,
        "physical_ids": physical,
    }
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        raise D105Phase1BundleError(f"output already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    archive = root / STRICT_TAP_ARCHIVE_NAME
    _write_new(root / SOURCE_ACCESS_RECEIPT_NAME, _canonical_bytes(access))
    with archive.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    archive_sha = sha256_file(archive)
    _reject_revoked_identity(revocation, tap_archive_sha256=archive_sha)
    receipt = {
        "schema": STRICT_TAP_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": checkpoint,
        "runtime_sha256": runtime,
        "method_lock_sha256": method_lock,
        "source_access_receipt_sha256": access_sha,
        "tap_archive_sha256": archive_sha,
        "tap_archive_members": list(STRICT_TAP_MEMBERS),
        "row_count": count,
        "forward_batch_capacity": capacity,
        "forward_invocation_count": forward_invocation_count,
        "last_batch_real_rows": last_batch_real_rows,
        "last_batch_padding_rows": capacity - last_batch_real_rows,
        "forward_batch_policy": D105_STRICT_TAP_FORWARD_BATCH_POLICY,
        "pre_relu_sha256": _array_sha256(arrays["pre_relu"]),
        "z_dom_sha256": _array_sha256(arrays["z_dom"]),
        "physical_id_root_sha256": _physical_root(physical.tolist()),
        "d102_revocation_manifest_sha256": revocation_sha,
        "execution_path": next(iter(execution_paths)),
        "hook_exact_bytes": True,
        "strict_pre_relu_path": True,
        "zid_relu_parity_verified": True,
        "z_dom_present": True,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "raw_iq_retained": False,
        "clean_iq_retained": False,
        "source_archive_phase1_only": True,
        "d102_rejected_bundle_reused": False,
    }
    receipt_path = root / STRICT_TAP_RECEIPT_NAME
    _write_new(receipt_path, _canonical_bytes(receipt))
    _write_new(root / D102_REVOCATION_MANIFEST_NAME, revocation["manifest_bytes"])
    _write_new(root / D102_REVOCATION_SIGNATURE_NAME, revocation["signature"])
    return {
        "strict_tap_archive": str(archive),
        "strict_tap_archive_sha256": archive_sha,
        "strict_tap_receipt": str(receipt_path),
        "strict_tap_receipt_sha256": _sha256_bytes(_canonical_bytes(receipt)),
        "source_access_receipt_sha256": access_sha,
        "d102_revocation_manifest_sha256": revocation_sha,
        "checkpoint_sha256": checkpoint,
        "runtime_sha256": runtime,
        "method_lock_sha256": method_lock,
        "row_count": count,
        "raw_iq_retained": False,
        "formal_phase2_eligible": False,
    }


def _validate_strict_tap_receipt(
    receipt: Mapping[str, Any],
    *,
    archive_sha256: str,
    arrays: Mapping[str, np.ndarray],
) -> str:
    _require_exact_keys(receipt, _STRICT_TAP_FIELDS, "strict tap receipt")
    _reject_d102(receipt, name="strict tap receipt")
    if (
        receipt["schema"] != STRICT_TAP_SCHEMA
        or receipt["candidate_id"] != CANDIDATE_ID
        or receipt["tap_archive_members"] != list(STRICT_TAP_MEMBERS)
        or receipt["tap_archive_sha256"] != archive_sha256
        or type(receipt["row_count"]) is not int
        or receipt["row_count"] <= 0
        or receipt["row_count"] != len(arrays["pre_relu"])
        or type(receipt["forward_batch_capacity"]) is not int
        or receipt["forward_batch_capacity"]
        != D105_STRICT_TAP_FORWARD_BATCH_CAPACITY
        or type(receipt["forward_invocation_count"]) is not int
        or receipt["forward_invocation_count"]
        != (
            len(arrays["pre_relu"]) + D105_STRICT_TAP_FORWARD_BATCH_CAPACITY - 1
        )
        // D105_STRICT_TAP_FORWARD_BATCH_CAPACITY
        or type(receipt["last_batch_real_rows"]) is not int
        or receipt["last_batch_real_rows"]
        != ((len(arrays["pre_relu"]) - 1) % D105_STRICT_TAP_FORWARD_BATCH_CAPACITY) + 1
        or type(receipt["last_batch_padding_rows"]) is not int
        or receipt["last_batch_padding_rows"]
        != D105_STRICT_TAP_FORWARD_BATCH_CAPACITY
        - receipt["last_batch_real_rows"]
        or receipt["forward_batch_policy"]
        != D105_STRICT_TAP_FORWARD_BATCH_POLICY
        or receipt["pre_relu_sha256"] != _array_sha256(arrays["pre_relu"])
        or receipt["z_dom_sha256"] != _array_sha256(arrays["z_dom"])
        or receipt["physical_id_root_sha256"]
        != _physical_root(arrays["physical_ids"].astype(str).tolist())
        or not isinstance(receipt["d102_revocation_manifest_sha256"], str)
        or receipt["execution_path"]
        not in ("strict_zid_with_hook", "torchscript_exported_functional_tap", "eager_forward_hook")
        or receipt["strict_pre_relu_path"] is not True
        or receipt["hook_exact_bytes"] is not True
        or receipt["zid_relu_parity_verified"] is not True
        or receipt["z_dom_present"] is not True
        or receipt["source_only"] is not True
        or receipt["target_rows"] != 0
        or receipt["query_rows"] != 0
        or receipt["raw_iq_retained"] is not False
        or receipt["clean_iq_retained"] is not False
        or receipt["source_archive_phase1_only"] is not True
        or receipt["d102_rejected_bundle_reused"] is not False
    ):
        raise D105Phase1BundleError("strict tap receipt closure drift")
    for field in (
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "source_access_receipt_sha256",
        "tap_archive_sha256",
        "pre_relu_sha256",
        "z_dom_sha256",
        "physical_id_root_sha256",
        "d102_revocation_manifest_sha256",
    ):
        _require_sha256(receipt[field], field)
    return _sha256_bytes(_canonical_bytes(receipt))


def load_d105_strict_tap_rows(
    archive_path: str | Path, strict_tap_receipt_path: str | Path
) -> tuple[StrictTapRows, dict[str, Any]]:
    """Read one Phase1-only strict-tap archive without retaining it downstream."""

    archive = Path(archive_path)
    if not archive.is_file() or archive.is_symlink():
        raise D105Phase1BundleError("strict tap archive must be a regular file")
    archive_sha = sha256_file(archive)
    try:
        with np.load(archive, allow_pickle=False) as loaded:
            if tuple(loaded.files) != STRICT_TAP_MEMBERS:
                raise D105Phase1BundleError(
                    "strict tap archive member allowlist/order drift"
                )
            arrays = {
                name: np.ascontiguousarray(np.array(loaded[name], copy=True))
                for name in STRICT_TAP_MEMBERS
            }
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, D105Phase1BundleError):
            raise
        raise D105Phase1BundleError("strict tap archive cannot be read safely") from error
    count = len(arrays["pre_relu"])
    for name in ("pre_relu", "z_dom"):
        value = arrays[name]
        if (
            value.dtype != np.float32
            or value.shape != (count, Z_DIM)
            or not np.isfinite(value).all()
        ):
            raise D105Phase1BundleError(f"strict tap {name} contract drift")
    for name in ("labels", "receiver_ids", "physical_ids"):
        value = arrays[name]
        if (
            value.ndim != 1
            or len(value) != count
            or value.dtype.kind not in {"U", "S"}
            or any(not item for item in value.astype(str).tolist())
        ):
            raise D105Phase1BundleError(f"strict tap {name} text contract drift")
    receipt = _load_json(strict_tap_receipt_path, name="strict tap receipt")
    receipt_sha = _validate_strict_tap_receipt(
        receipt, archive_sha256=archive_sha, arrays=arrays
    )
    rows = StrictTapRows(
        pre_relu=arrays["pre_relu"],
        z_dom=arrays["z_dom"],
        labels=tuple(arrays["labels"].astype(str).tolist()),
        receiver_ids=tuple(arrays["receiver_ids"].astype(str).tolist()),
        physical_ids=tuple(arrays["physical_ids"].astype(str).tolist()),
        archive_sha256=archive_sha,
        strict_tap_receipt_sha256=receipt_sha,
    )
    return rows, receipt


def compute_d105_source_aggregate_lineage(
    *,
    strict_tap_receipt_sha256: str,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
) -> str:
    """Name immutable aggregate semantics without introducing a gate cycle."""

    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema": SCHEMA + ".source_aggregate_lineage.v1",
                "candidate_id": CANDIDATE_ID,
                "strict_tap_receipt_sha256": _require_sha256(
                    strict_tap_receipt_sha256, "strict_tap_receipt_sha256"
                ),
                "checkpoint_sha256": _require_sha256(
                    checkpoint_sha256, "checkpoint_sha256"
                ),
                "runtime_sha256": _require_sha256(runtime_sha256, "runtime_sha256"),
                "method_lock_sha256": _require_sha256(
                    method_lock_sha256, "method_lock_sha256"
                ),
                "aggregation_algorithm": (
                    "class_centered_receiver_aggregate_canonical_svd_v1"
                ),
                "payload_abi": "rxid_metabias4_int8_fp16_v1",
                "target_rows": 0,
            }
        )
    )


def _tokens(values: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise D105Phase1BundleError(f"{name} must be a JSON list")
    result = tuple(str(item) for item in values)
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise D105Phase1BundleError(f"{name} must contain unique non-empty tokens")
    return result


def _label_sequence(values: Any, *, name: str) -> tuple[str, ...]:
    """Return non-empty labels while deliberately permitting repeated classes."""

    if not isinstance(values, list):
        raise D105Phase1BundleError(f"{name} must be a JSON list")
    result = tuple(str(item) for item in values)
    if not result or any(not item for item in result):
        raise D105Phase1BundleError(f"{name} must contain non-empty labels")
    return result


def _balanced_accuracy_and_floor(
    truth: Sequence[str], predicted: Sequence[str], classes: Sequence[str]
) -> tuple[float, float]:
    truth_values = np.asarray(tuple(str(item) for item in truth), dtype=np.str_)
    predicted_values = np.asarray(
        tuple(str(item) for item in predicted), dtype=np.str_
    )
    if len(truth_values) < 1 or truth_values.shape != predicted_values.shape:
        raise D105Phase1BundleError("score labels/predictions alignment drift")
    class_values = tuple(str(item) for item in classes)
    if (
        any(item not in class_values for item in truth_values.tolist())
        or any(item not in class_values for item in predicted_values.tolist())
        or any(not np.any(truth_values == item) for item in class_values)
    ):
        raise D105Phase1BundleError(
            "every score row must cover all registered source classes"
        )
    values = [
        float(np.mean(predicted_values[truth_values == item] == item))
        for item in class_values
    ]
    return float(np.mean(values)), float(np.min(values))


def _prediction_commit_payload(
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    strict_tap_receipt_sha256: str,
    source_aggregate_lineage_sha256: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(row, _PREDICTION_ROW_FIELDS, "source-held prediction row")
    if row["query_rows_used_for_fit"] != 0:
        raise D105Phase1BundleError("source-held prediction query lifecycle drift")
    if row["fold_kind"] not in ("receiver_held", "class_loco"):
        raise D105Phase1BundleError("source-held fold kind drift")
    if type(row["K"]) is not int or row["K"] not in (1, 5, 10):
        raise D105Phase1BundleError("source-held K drift")
    physical = _tokens(row["query_physical_ids"], name="query physical IDs")
    m0 = _label_sequence(row["m0_predictions"], name="M0 predictions")
    fp32 = _label_sequence(
        row["d105_fp32_predictions"], name="D105 FP32 predictions"
    )
    int8 = _label_sequence(
        row["d105_int8_predictions"], name="D105 INT8 predictions"
    )
    margins = row["d105_fp32_top2_margins"]
    if (
        len(m0) != len(physical)
        or len(fp32) != len(physical)
        or len(int8) != len(physical)
        or not isinstance(margins, list)
        or len(margins) != len(physical)
    ):
        raise D105Phase1BundleError("prediction commit row length drift")
    numeric_margins = tuple(float(item) for item in margins)
    if any(not math.isfinite(item) or item < 0.0 for item in numeric_margins):
        raise D105Phase1BundleError("prediction commit margin field drift")
    return {
        "schema": SOURCE_HELD_PREDICTION_COMMIT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_sha256": runtime_sha256,
        "method_lock_sha256": method_lock_sha256,
        "strict_tap_receipt_sha256": strict_tap_receipt_sha256,
        "source_aggregate_lineage_sha256": source_aggregate_lineage_sha256,
        "row_id": str(row["row_id"]),
        "fold_kind": row["fold_kind"],
        "held_receiver_token": str(row["held_receiver_token"]),
        "held_class_token": (
            None if row["held_class_token"] is None else str(row["held_class_token"])
        ),
        "K": row["K"],
        "query_physical_ids": list(physical),
        "m0_predictions": list(m0),
        "d105_fp32_predictions": list(fp32),
        "d105_int8_predictions": list(int8),
        "d105_fp32_top2_margins": list(numeric_margins),
        "query_truth_present": False,
        "query_rows_used_for_fit": 0,
    }


def compute_d105_source_held_prediction_commit(
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    strict_tap_receipt_sha256: str,
    source_aggregate_lineage_sha256: str,
    row: Mapping[str, Any],
) -> str:
    """Return the pre-truth immutable commit expected in a held score artifact."""

    payload = _prediction_commit_payload(
        checkpoint_sha256=_require_sha256(checkpoint_sha256, "checkpoint_sha256"),
        runtime_sha256=_require_sha256(runtime_sha256, "runtime_sha256"),
        method_lock_sha256=_require_sha256(method_lock_sha256, "method_lock_sha256"),
        strict_tap_receipt_sha256=_require_sha256(
            strict_tap_receipt_sha256, "strict_tap_receipt_sha256"
        ),
        source_aggregate_lineage_sha256=_require_sha256(
            source_aggregate_lineage_sha256,
            "source_aggregate_lineage_sha256",
        ),
        row=row,
    )
    return _sha256_bytes(_canonical_bytes(payload))


def _tx_prediction_commit_payload(
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    strict_tap_receipt_sha256: str,
    source_aggregate_lineage_sha256: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(row, _TX_PREDICTION_ROW_FIELDS, "TX prediction row")
    receiver = str(row["held_receiver_token"])
    physical = _tokens(row["physical_ids"], name="TX probe physical IDs")
    predictions = _label_sequence(row["predictions"], name="TX predictions")
    if not receiver or len(physical) != len(predictions):
        raise D105Phase1BundleError("TX prediction commit row length/identity drift")
    return {
        "schema": SOURCE_HELD_PREDICTION_COMMIT_SCHEMA + ".tx_probe.v1",
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_sha256": runtime_sha256,
        "method_lock_sha256": method_lock_sha256,
        "strict_tap_receipt_sha256": strict_tap_receipt_sha256,
        "source_aggregate_lineage_sha256": source_aggregate_lineage_sha256,
        "held_receiver_token": receiver,
        "physical_ids": list(physical),
        "predictions": list(predictions),
        "truth_present": False,
        "query_rows_used_for_fit": 0,
    }


def compute_d105_source_held_tx_prediction_commit(
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    strict_tap_receipt_sha256: str,
    source_aggregate_lineage_sha256: str,
    row: Mapping[str, Any],
) -> str:
    """Return the pre-truth immutable TX-probe prediction commitment."""

    return _sha256_bytes(
        _canonical_bytes(
            _tx_prediction_commit_payload(
                checkpoint_sha256=_require_sha256(
                    checkpoint_sha256, "checkpoint_sha256"
                ),
                runtime_sha256=_require_sha256(runtime_sha256, "runtime_sha256"),
                method_lock_sha256=_require_sha256(
                    method_lock_sha256, "method_lock_sha256"
                ),
                strict_tap_receipt_sha256=_require_sha256(
                    strict_tap_receipt_sha256, "strict_tap_receipt_sha256"
                ),
                source_aggregate_lineage_sha256=_require_sha256(
                    source_aggregate_lineage_sha256,
                    "source_aggregate_lineage_sha256",
                ),
                row=row,
            )
        )
    )


def _validate_source_held_header(
    value: Mapping[str, Any],
    *,
    expected_fields: set[str],
    expected_schema: str,
    name: str,
    strict_tap_receipt_sha256: str,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    expected_lineage: str,
) -> None:
    _require_exact_keys(value, expected_fields, name)
    _reject_d102(value, name=name)
    if (
        value["schema"] != expected_schema
        or value["candidate_id"] != CANDIDATE_ID
        or value["checkpoint_sha256"] != checkpoint_sha256
        or value["runtime_sha256"] != runtime_sha256
        or value["method_lock_sha256"] != method_lock_sha256
        or value["strict_tap_receipt_sha256"] != strict_tap_receipt_sha256
        or value["source_aggregate_lineage_sha256"] != expected_lineage
        or value["source_only"] is not True
        or value["target_rows"] != 0
        or value["query_rows"] != 0
        or value["d102_rejected_bundle_reused"] is not False
    ):
        raise D105Phase1BundleError(f"{name} binding/lifecycle drift")
    for field in (
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "strict_tap_receipt_sha256",
        "source_aggregate_lineage_sha256",
    ):
        _require_sha256(value[field], field)


def _source_held_gate_from_evidence(
    prediction_manifest: Mapping[str, Any],
    truth_open_receipt: Mapping[str, Any],
    score: Mapping[str, Any],
    *,
    strict_tap_receipt_sha256: str,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Join immutable pre-truth predictions to an independent truth-side score."""

    expected_lineage = compute_d105_source_aggregate_lineage(
        strict_tap_receipt_sha256=strict_tap_receipt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
    )
    _validate_source_held_header(
        prediction_manifest,
        expected_fields=_SOURCE_HELD_PREDICTION_FIELDS,
        expected_schema=SOURCE_HELD_PREDICTION_SCHEMA,
        name="source-held prediction manifest",
        strict_tap_receipt_sha256=strict_tap_receipt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        expected_lineage=expected_lineage,
    )
    prediction_sha = _sha256_bytes(_canonical_bytes(prediction_manifest))
    _validate_source_held_header(
        truth_open_receipt,
        expected_fields=_SOURCE_HELD_TRUTH_OPEN_FIELDS,
        expected_schema=SOURCE_HELD_TRUTH_OPEN_SCHEMA,
        name="source-held truth-open receipt",
        strict_tap_receipt_sha256=strict_tap_receipt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        expected_lineage=expected_lineage,
    )
    if (
        truth_open_receipt["source_held_prediction_manifest_sha256"]
        != prediction_sha
        or truth_open_receipt["prediction_manifest_immutable"] is not True
        or truth_open_receipt["truth_opened_after_prediction"] is not True
    ):
        raise D105Phase1BundleError("truth-open/prediction ordering drift")
    _require_sha256(
        truth_open_receipt["source_held_prediction_manifest_sha256"],
        "source_held_prediction_manifest_sha256",
    )
    truth_open_sha = _sha256_bytes(_canonical_bytes(truth_open_receipt))
    _validate_source_held_header(
        score,
        expected_fields=_SOURCE_HELD_SCORE_FIELDS,
        expected_schema=SOURCE_HELD_SCORE_SCHEMA,
        name="source-held score artifact",
        strict_tap_receipt_sha256=strict_tap_receipt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        expected_lineage=expected_lineage,
    )
    if (
        score["source_held_prediction_manifest_sha256"] != prediction_sha
        or score["source_held_truth_open_receipt_sha256"] != truth_open_sha
    ):
        raise D105Phase1BundleError("source-held score ordering/binding drift")
    _require_sha256(
        score["source_held_prediction_manifest_sha256"],
        "source_held_prediction_manifest_sha256",
    )
    _require_sha256(
        score["source_held_truth_open_receipt_sha256"],
        "source_held_truth_open_receipt_sha256",
    )
    receivers = _tokens(prediction_manifest["receiver_tokens"], name="receiver tokens")
    classes = _tokens(prediction_manifest["class_tokens"], name="class tokens")
    prediction_rows = prediction_manifest["scored_prediction_rows"]
    tx_prediction_rows = prediction_manifest["tx_probe_prediction_rows"]
    truth_rows = score["scored_truth_rows"]
    tx_truth_rows = score["tx_probe_truth_rows"]
    if not all(isinstance(items, list) for items in (
        prediction_rows, tx_prediction_rows, truth_rows, tx_truth_rows
    )):
        raise D105Phase1BundleError("source-held evidence row list drift")
    truth_by_id: dict[str, tuple[str, ...]] = {}
    for truth_row in truth_rows:
        if type(truth_row) is not dict:
            raise D105Phase1BundleError("source-held truth row type drift")
        _require_exact_keys(truth_row, _TRUTH_ROW_FIELDS, "source-held truth row")
        row_id = str(truth_row["row_id"])
        if not row_id or row_id in truth_by_id:
            raise D105Phase1BundleError("source-held truth row identity drift")
        truth_by_id[row_id] = _label_sequence(
            truth_row["truth_labels"], name="truth labels"
        )
    tx_truth_by_receiver: dict[str, tuple[str, ...]] = {}
    for truth_row in tx_truth_rows:
        if type(truth_row) is not dict:
            raise D105Phase1BundleError("TX truth row type drift")
        _require_exact_keys(truth_row, _TX_TRUTH_ROW_FIELDS, "TX truth row")
        receiver = str(truth_row["held_receiver_token"])
        if not receiver or receiver in tx_truth_by_receiver:
            raise D105Phase1BundleError("TX truth row identity drift")
        tx_truth_by_receiver[receiver] = _label_sequence(
            truth_row["truth_labels"], name="TX probe truth labels"
        )
    expected_receiver = {(receiver, k) for receiver in receivers for k in (1, 5, 10)}
    expected_loco = {(receiver, label) for receiver in receivers for label in classes}
    seen_receiver: set[tuple[str, int]] = set()
    seen_loco: set[tuple[str, str]] = set()
    prediction_row_ids: set[str] = set()
    receiver_failures = 0
    loco_failures = 0
    receiver_ba_deltas: list[float] = []
    receiver_floor_deltas: list[float] = []
    receiver_net: list[int] = []
    loco_ba_deltas: list[float] = []
    loco_floor_deltas: list[float] = []
    loco_net: list[int] = []
    agreements: list[float] = []
    flips: list[int] = []
    for row in prediction_rows:
        if type(row) is not dict:
            raise D105Phase1BundleError("source-held prediction row type drift")
        commit = compute_d105_source_held_prediction_commit(
            checkpoint_sha256=checkpoint_sha256,
            runtime_sha256=runtime_sha256,
            method_lock_sha256=method_lock_sha256,
            strict_tap_receipt_sha256=strict_tap_receipt_sha256,
            source_aggregate_lineage_sha256=expected_lineage,
            row=row,
        )
        if row["prediction_commit_sha256"] != commit:
            raise D105Phase1BundleError("source-held prediction commit hash drift")
        row_id = str(row["row_id"])
        receiver = str(row["held_receiver_token"])
        held_class = row["held_class_token"]
        if (
            not row_id
            or row_id in prediction_row_ids
            or receiver not in receivers
            or row_id not in truth_by_id
        ):
            raise D105Phase1BundleError("source-held prediction/truth row identity drift")
        prediction_row_ids.add(row_id)
        truth = truth_by_id[row_id]
        physical = _tokens(row["query_physical_ids"], name="query physical IDs")
        if len(truth) != len(physical):
            raise D105Phase1BundleError("source-held prediction/truth length drift")
        base_ba, base_floor = _balanced_accuracy_and_floor(
            truth, row["m0_predictions"], classes
        )
        d105_ba, d105_floor = _balanced_accuracy_and_floor(
            truth, row["d105_int8_predictions"], classes
        )
        truth_values = np.asarray(truth, dtype=np.str_)
        base_correct = np.asarray(row["m0_predictions"], dtype=np.str_) == truth_values
        d105_correct = (
            np.asarray(row["d105_int8_predictions"], dtype=np.str_) == truth_values
        )
        net_correct = int(np.sum(d105_correct) - np.sum(base_correct))
        noninferior = (
            d105_ba + 1.0e-12 >= base_ba
            and d105_floor + 1.0e-12 >= base_floor
            and net_correct >= 0
        )
        fp32_predictions = np.asarray(row["d105_fp32_predictions"], dtype=np.str_)
        int8_predictions = np.asarray(row["d105_int8_predictions"], dtype=np.str_)
        margins = np.asarray(row["d105_fp32_top2_margins"], dtype=np.float64)
        agreements.append(float(np.mean(fp32_predictions == int8_predictions)))
        flips.append(int(np.sum(
            (fp32_predictions != int8_predictions)
            & (margins >= _LARGE_MARGIN_MINIMUM)
        )))
        if row["fold_kind"] == "receiver_held":
            if held_class is not None:
                raise D105Phase1BundleError("receiver-held row must not name a class")
            key = (receiver, int(row["K"]))
            if key in seen_receiver:
                raise D105Phase1BundleError("duplicate receiver-held coverage row")
            seen_receiver.add(key)
            receiver_failures += int(not noninferior)
            receiver_ba_deltas.append(float(d105_ba - base_ba))
            receiver_floor_deltas.append(float(d105_floor - base_floor))
            receiver_net.append(net_correct)
        else:
            if int(row["K"]) != 1 or str(held_class) not in classes:
                raise D105Phase1BundleError("class-LOCO row lifecycle/K drift")
            key = (receiver, str(held_class))
            if key in seen_loco:
                raise D105Phase1BundleError("duplicate class-LOCO coverage row")
            seen_loco.add(key)
            loco_failures += int(not noninferior)
            loco_ba_deltas.append(float(d105_ba - base_ba))
            loco_floor_deltas.append(float(d105_floor - base_floor))
            loco_net.append(net_correct)
    if set(truth_by_id) != prediction_row_ids:
        raise D105Phase1BundleError("source-held truth row coverage drift")
    if seen_receiver != expected_receiver or seen_loco != expected_loco:
        raise D105Phase1BundleError("source-held receiver/LOCO coverage is incomplete")
    tx_by_receiver: dict[str, float] = {}
    for row in tx_prediction_rows:
        if type(row) is not dict:
            raise D105Phase1BundleError("TX prediction row type drift")
        _require_exact_keys(row, _TX_PREDICTION_ROW_FIELDS, "TX prediction row")
        receiver = str(row["held_receiver_token"])
        physical = _tokens(row["physical_ids"], name="TX probe physical IDs")
        predictions = _label_sequence(row["predictions"], name="TX predictions")
        truth = tx_truth_by_receiver.get(receiver)
        if receiver not in receivers or receiver in tx_by_receiver or truth is None:
            raise D105Phase1BundleError("TX prediction/truth receiver coverage drift")
        if len(physical) != len(truth) or len(predictions) != len(truth):
            raise D105Phase1BundleError("TX probe row length drift")
        expected_commit = compute_d105_source_held_tx_prediction_commit(
            checkpoint_sha256=checkpoint_sha256,
            runtime_sha256=runtime_sha256,
            method_lock_sha256=method_lock_sha256,
            strict_tap_receipt_sha256=strict_tap_receipt_sha256,
            source_aggregate_lineage_sha256=expected_lineage,
            row=row,
        )
        if row["prediction_commit_sha256"] != expected_commit:
            raise D105Phase1BundleError("TX probe prediction commit hash drift")
        ba, _ = _balanced_accuracy_and_floor(truth, predictions, classes)
        tx_by_receiver[receiver] = ba
    if set(tx_by_receiver) != set(receivers) or set(tx_truth_by_receiver) != set(receivers):
        raise D105Phase1BundleError("TX probe coverage is incomplete")
    tx_values = list(tx_by_receiver.values())
    agreement_min = float(min(agreements))
    flip_total = int(sum(flips))
    receiver_noninferior = receiver_failures == 0
    loco_noninferior = loco_failures == 0
    tx_mean = float(np.mean(tx_values))
    tx_max = float(max(tx_values))
    quantization_pass = agreement_min >= 0.995 and flip_total == 0
    missing: list[str] = []
    if not receiver_noninferior:
        missing.append("receiver_held_all_noninferior")
    if not loco_noninferior:
        missing.append("class_loco_complete_and_noninferior")
    if tx_max > 0.25:
        missing.append("tx_probe_max_balanced_accuracy_at_most_0_25")
    if not quantization_pass:
        missing.append("int8_agreement_at_least_0_995_and_zero_large_margin_flip")
    summary = {
        "receiver_held_complete": True,
        "receiver_held_k": [1, 5, 10],
        "receiver_held_all_noninferior": receiver_noninferior,
        "receiver_held_row_count": len(seen_receiver),
        "receiver_held_failing_row_count": receiver_failures,
        "receiver_held_worst_ba_delta": float(min(receiver_ba_deltas)),
        "receiver_held_worst_floor_delta": float(min(receiver_floor_deltas)),
        "receiver_held_min_net_correct": int(min(receiver_net)),
        "class_loco_complete": True,
        "class_loco_all_noninferior": loco_noninferior,
        "class_loco_row_count": len(seen_loco),
        "class_loco_failing_row_count": loco_failures,
        "class_loco_worst_ba_delta": float(min(loco_ba_deltas)),
        "class_loco_worst_floor_delta": float(min(loco_floor_deltas)),
        "class_loco_min_net_correct": int(min(loco_net)),
        "tx_probe_mean_balanced_accuracy": tx_mean,
        "tx_probe_max_balanced_accuracy": tx_max,
        "tx_probe_gate_pass": tx_max <= 0.25,
        "quantization_top1_agreement": agreement_min,
        "quantization_large_margin_flip_count": flip_total,
        "quantization_gate_pass": quantization_pass,
        "persistent_fp32_sidecar": False,
        "raw_iq_persisted": False,
        "source_replay_persisted": False,
    }
    gate = {
        "schema": SOURCE_HELD_GATE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_sha256": runtime_sha256,
        "method_lock_sha256": method_lock_sha256,
        "strict_tap_receipt_sha256": strict_tap_receipt_sha256,
        "source_aggregate_lineage_sha256": expected_lineage,
        "source_held_prediction_manifest_sha256": prediction_sha,
        "source_held_truth_open_receipt_sha256": truth_open_sha,
        "source_held_score_artifact_sha256": _sha256_bytes(_canonical_bytes(score)),
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "receiver_count": len(receivers),
        "class_count": len(classes),
        **summary,
        "d102_rejected_bundle_reused": False,
    }
    return gate, missing, summary


def _validate_derived_source_held_gate(
    gate: Mapping[str, Any],
    *,
    strict_tap_receipt_sha256: str,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
) -> dict[str, Any]:
    """Validate the builder-derived, aggregate-safe gate receipt.

    This receipt is intentionally not accepted as a build input.  It is only
    persisted after the builder has recomputed it from a truth-containing
    source-held score artifact.  The validator here protects the immutable
    component/asset copy from tampering without reopening source rows.
    """

    _require_exact_keys(gate, _SOURCE_HELD_GATE_FIELDS, "source-held gate receipt")
    _reject_d102(gate, name="source-held gate receipt")
    expected_lineage = compute_d105_source_aggregate_lineage(
        strict_tap_receipt_sha256=strict_tap_receipt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
    )
    if (
        gate["schema"] != SOURCE_HELD_GATE_SCHEMA
        or gate["candidate_id"] != CANDIDATE_ID
        or gate["checkpoint_sha256"] != checkpoint_sha256
        or gate["runtime_sha256"] != runtime_sha256
        or gate["method_lock_sha256"] != method_lock_sha256
        or gate["strict_tap_receipt_sha256"] != strict_tap_receipt_sha256
        or gate["source_aggregate_lineage_sha256"] != expected_lineage
        or gate["source_only"] is not True
        or gate["target_rows"] != 0
        or gate["query_rows"] != 0
        or gate["d102_rejected_bundle_reused"] is not False
        or gate["receiver_held_k"] != [1, 5, 10]
        or gate["persistent_fp32_sidecar"] is not False
        or gate["raw_iq_persisted"] is not False
        or gate["source_replay_persisted"] is not False
    ):
        raise D105Phase1BundleError("source-held derived gate binding/lifecycle drift")
    for field in (
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "strict_tap_receipt_sha256",
        "source_aggregate_lineage_sha256",
        "source_held_prediction_manifest_sha256",
        "source_held_truth_open_receipt_sha256",
        "source_held_score_artifact_sha256",
    ):
        _require_sha256(gate[field], field)
    for field in (
        "receiver_count",
        "class_count",
        "receiver_held_row_count",
        "receiver_held_failing_row_count",
        "class_loco_row_count",
        "class_loco_failing_row_count",
        "quantization_large_margin_flip_count",
    ):
        if type(gate[field]) is not int or gate[field] < 0:
            raise D105Phase1BundleError("source-held derived gate integer drift")
    for field in (
        "receiver_held_min_net_correct",
        "class_loco_min_net_correct",
    ):
        if type(gate[field]) is not int:
            raise D105Phase1BundleError("source-held derived gate integer drift")
    if gate["receiver_count"] < 2 or gate["class_count"] < 2:
        raise D105Phase1BundleError("source-held derived gate coverage cardinality drift")
    for field in (
        "receiver_held_complete",
        "receiver_held_all_noninferior",
        "class_loco_complete",
        "class_loco_all_noninferior",
        "tx_probe_gate_pass",
        "quantization_gate_pass",
    ):
        if type(gate[field]) is not bool:
            raise D105Phase1BundleError("source-held derived gate boolean drift")
    for field in (
        "receiver_held_worst_ba_delta",
        "receiver_held_worst_floor_delta",
        "class_loco_worst_ba_delta",
        "class_loco_worst_floor_delta",
        "tx_probe_mean_balanced_accuracy",
        "tx_probe_max_balanced_accuracy",
        "quantization_top1_agreement",
    ):
        value = float(gate[field])
        if not math.isfinite(value):
            raise D105Phase1BundleError("source-held derived gate numeric drift")
    if not 0.0 <= float(gate["tx_probe_mean_balanced_accuracy"]) <= 1.0:
        raise D105Phase1BundleError("source-held derived TX mean drift")
    if not 0.0 <= float(gate["tx_probe_max_balanced_accuracy"]) <= 1.0:
        raise D105Phase1BundleError("source-held derived TX max drift")
    if not 0.0 <= float(gate["quantization_top1_agreement"]) <= 1.0:
        raise D105Phase1BundleError("source-held derived agreement drift")
    return {
        key: gate[key]
        for key in (
            "receiver_held_complete",
            "receiver_held_k",
            "receiver_held_all_noninferior",
            "receiver_held_row_count",
            "receiver_held_failing_row_count",
            "receiver_held_worst_ba_delta",
            "receiver_held_worst_floor_delta",
            "receiver_held_min_net_correct",
            "class_loco_complete",
            "class_loco_all_noninferior",
            "class_loco_row_count",
            "class_loco_failing_row_count",
            "class_loco_worst_ba_delta",
            "class_loco_worst_floor_delta",
            "class_loco_min_net_correct",
            "tx_probe_mean_balanced_accuracy",
            "tx_probe_max_balanced_accuracy",
            "tx_probe_gate_pass",
            "quantization_top1_agreement",
            "quantization_large_margin_flip_count",
            "quantization_gate_pass",
            "persistent_fp32_sidecar",
            "raw_iq_persisted",
            "source_replay_persisted",
        )
    }


def derive_d105_source_held_gate(
    strict_tap_archive: str | Path,
    strict_tap_receipt: str | Path,
    candidate_method_lock: str | Path,
    candidate_runtime_manifest: str | Path,
    source_held_prediction_manifest: str | Path,
    source_held_truth_open_receipt: str | Path,
    source_held_score_artifact: str | Path,
) -> dict[str, Any]:
    """Independently recompute a D105 held gate from committed row evidence.

    The pre-truth prediction manifest, subsequent truth-open receipt, and
    truth-side score are three separate immutable inputs. This function joins
    them only after checking their order and commitments, then recomputes all
    receiver/LOCO/TX/INT8 gates. It never writes source rows into a
    Phase2-visible output.
    """

    rows, tap = load_d105_strict_tap_rows(strict_tap_archive, strict_tap_receipt)
    _validate_candidate_lock_for_tap(
        candidate_method_lock, candidate_runtime_manifest, tap
    )
    prediction = _load_immutable_json(
        source_held_prediction_manifest, name="source-held prediction manifest"
    )
    truth_open = _load_immutable_json(
        source_held_truth_open_receipt, name="source-held truth-open receipt"
    )
    score = _load_immutable_json(
        source_held_score_artifact, name="source-held score artifact"
    )
    gate, missing, summary = _source_held_gate_from_evidence(
        prediction,
        truth_open,
        score,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    _validate_derived_source_held_gate(
        gate,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    return {
        "gate": gate,
        "gate_sha256": _sha256_bytes(_canonical_bytes(gate)),
        "formal_prerequisites_missing": list(missing),
        "summary": summary,
    }


def _validate_prediction_manifest_for_source_rows(
    manifest: Mapping[str, Any],
    *,
    strict_tap_receipt_sha256: str,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Validate a complete no-truth source-held prediction surface."""

    lineage = compute_d105_source_aggregate_lineage(
        strict_tap_receipt_sha256=strict_tap_receipt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
    )
    _validate_source_held_header(
        manifest,
        expected_fields=_SOURCE_HELD_PREDICTION_FIELDS,
        expected_schema=SOURCE_HELD_PREDICTION_SCHEMA,
        name="source-held prediction manifest",
        strict_tap_receipt_sha256=strict_tap_receipt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        expected_lineage=lineage,
    )
    receivers = _tokens(manifest["receiver_tokens"], name="receiver tokens")
    classes = _tokens(manifest["class_tokens"], name="class tokens")
    expected_receiver = {(receiver, k) for receiver in receivers for k in (1, 5, 10)}
    expected_loco = {(receiver, label) for receiver in receivers for label in classes}
    seen_receiver: set[tuple[str, int]] = set()
    seen_loco: set[tuple[str, str]] = set()
    row_ids: set[str] = set()
    rows = manifest["scored_prediction_rows"]
    if not isinstance(rows, list):
        raise D105Phase1BundleError("source-held prediction rows must be a JSON list")
    for row in rows:
        if type(row) is not dict:
            raise D105Phase1BundleError("source-held prediction row type drift")
        expected_commit = compute_d105_source_held_prediction_commit(
            checkpoint_sha256=checkpoint_sha256,
            runtime_sha256=runtime_sha256,
            method_lock_sha256=method_lock_sha256,
            strict_tap_receipt_sha256=strict_tap_receipt_sha256,
            source_aggregate_lineage_sha256=lineage,
            row=row,
        )
        if row["prediction_commit_sha256"] != expected_commit:
            raise D105Phase1BundleError("source-held prediction commit hash drift")
        row_id = str(row["row_id"])
        receiver = str(row["held_receiver_token"])
        if not row_id or row_id in row_ids or receiver not in receivers:
            raise D105Phase1BundleError("source-held prediction row identity drift")
        row_ids.add(row_id)
        if row["fold_kind"] == "receiver_held":
            if row["held_class_token"] is not None:
                raise D105Phase1BundleError("receiver-held prediction names a class")
            key = (receiver, int(row["K"]))
            if key in seen_receiver:
                raise D105Phase1BundleError("duplicate receiver-held prediction row")
            seen_receiver.add(key)
        else:
            held_class = str(row["held_class_token"])
            if int(row["K"]) != 1 or held_class not in classes:
                raise D105Phase1BundleError("class-LOCO prediction lifecycle/K drift")
            key = (receiver, held_class)
            if key in seen_loco:
                raise D105Phase1BundleError("duplicate class-LOCO prediction row")
            seen_loco.add(key)
    if seen_receiver != expected_receiver or seen_loco != expected_loco:
        raise D105Phase1BundleError("source-held prediction coverage is incomplete")
    tx_rows = manifest["tx_probe_prediction_rows"]
    if not isinstance(tx_rows, list):
        raise D105Phase1BundleError("TX prediction rows must be a JSON list")
    tx_seen: set[str] = set()
    for row in tx_rows:
        if type(row) is not dict:
            raise D105Phase1BundleError("TX prediction row type drift")
        _require_exact_keys(row, _TX_PREDICTION_ROW_FIELDS, "TX prediction row")
        receiver = str(row["held_receiver_token"])
        expected_commit = compute_d105_source_held_tx_prediction_commit(
            checkpoint_sha256=checkpoint_sha256,
            runtime_sha256=runtime_sha256,
            method_lock_sha256=method_lock_sha256,
            strict_tap_receipt_sha256=strict_tap_receipt_sha256,
            source_aggregate_lineage_sha256=lineage,
            row=row,
        )
        if (
            receiver not in receivers
            or receiver in tx_seen
            or row["prediction_commit_sha256"] != expected_commit
        ):
            raise D105Phase1BundleError("TX prediction row identity/commit drift")
        tx_seen.add(receiver)
    if tx_seen != set(receivers):
        raise D105Phase1BundleError("TX prediction coverage is incomplete")
    return receivers, classes, lineage


def open_d105_source_held_truth(
    strict_tap_archive: str | Path,
    strict_tap_receipt: str | Path,
    candidate_method_lock: str | Path,
    candidate_runtime_manifest: str | Path,
    source_held_prediction_manifest: str | Path,
    output_receipt: str | Path,
) -> dict[str, Any]:
    """Open the source-held truth side only after complete predictions freeze."""

    rows, tap = load_d105_strict_tap_rows(strict_tap_archive, strict_tap_receipt)
    _validate_candidate_lock_for_tap(
        candidate_method_lock, candidate_runtime_manifest, tap
    )
    prediction = _load_immutable_json(
        source_held_prediction_manifest, name="source-held prediction manifest"
    )
    _, _, lineage = _validate_prediction_manifest_for_source_rows(
        prediction,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    prediction_sha = _sha256_bytes(_canonical_bytes(prediction))
    receipt = {
        "schema": SOURCE_HELD_TRUTH_OPEN_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": str(tap["checkpoint_sha256"]),
        "runtime_sha256": str(tap["runtime_sha256"]),
        "method_lock_sha256": str(tap["method_lock_sha256"]),
        "strict_tap_receipt_sha256": rows.strict_tap_receipt_sha256,
        "source_aggregate_lineage_sha256": lineage,
        "source_held_prediction_manifest_sha256": prediction_sha,
        "prediction_manifest_immutable": True,
        "truth_opened_after_prediction": True,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "d102_rejected_bundle_reused": False,
    }
    path = Path(output_receipt)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise D105Phase1BundleError("truth-open output must be a new child file")
    receipt_sha = _write_new_immutable_json(path, receipt)
    return {
        "truth_open_receipt": str(path),
        "truth_open_receipt_sha256": receipt_sha,
        "source_held_prediction_manifest_sha256": prediction_sha,
        "truth_labels_persisted": False,
        "formal_phase2_eligible": False,
    }


def score_d105_source_held_truth(
    strict_tap_archive: str | Path,
    strict_tap_receipt: str | Path,
    candidate_method_lock: str | Path,
    candidate_runtime_manifest: str | Path,
    source_held_prediction_manifest: str | Path,
    source_held_truth_open_receipt: str | Path,
    output_score_artifact: str | Path,
) -> dict[str, Any]:
    """Independently join frozen predictions to strict-tap truth labels once."""

    rows, tap = load_d105_strict_tap_rows(strict_tap_archive, strict_tap_receipt)
    _validate_candidate_lock_for_tap(
        candidate_method_lock, candidate_runtime_manifest, tap
    )
    prediction = _load_immutable_json(
        source_held_prediction_manifest, name="source-held prediction manifest"
    )
    receivers, _, lineage = _validate_prediction_manifest_for_source_rows(
        prediction,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    truth_open = _load_immutable_json(
        source_held_truth_open_receipt, name="source-held truth-open receipt"
    )
    prediction_sha = _sha256_bytes(_canonical_bytes(prediction))
    _validate_source_held_header(
        truth_open,
        expected_fields=_SOURCE_HELD_TRUTH_OPEN_FIELDS,
        expected_schema=SOURCE_HELD_TRUTH_OPEN_SCHEMA,
        name="source-held truth-open receipt",
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
        expected_lineage=lineage,
    )
    if (
        truth_open["source_held_prediction_manifest_sha256"] != prediction_sha
        or truth_open["prediction_manifest_immutable"] is not True
        or truth_open["truth_opened_after_prediction"] is not True
    ):
        raise D105Phase1BundleError("truth scorer ordering receipt drift")
    physical_to_label = dict(zip(rows.physical_ids, rows.labels, strict=True))
    scored_truth_rows: list[dict[str, Any]] = []
    for row in prediction["scored_prediction_rows"]:
        physical = _tokens(row["query_physical_ids"], name="query physical IDs")
        try:
            truth = [physical_to_label[item] for item in physical]
        except KeyError as error:
            raise D105Phase1BundleError(
                "prediction query physical ID is outside the strict source tap"
            ) from error
        scored_truth_rows.append(
            {"row_id": str(row["row_id"]), "truth_labels": truth}
        )
    tx_truth_rows: list[dict[str, Any]] = []
    for row in prediction["tx_probe_prediction_rows"]:
        physical = _tokens(row["physical_ids"], name="TX probe physical IDs")
        try:
            truth = [physical_to_label[item] for item in physical]
        except KeyError as error:
            raise D105Phase1BundleError(
                "TX prediction physical ID is outside the strict source tap"
            ) from error
        tx_truth_rows.append(
            {
                "held_receiver_token": str(row["held_receiver_token"]),
                "truth_labels": truth,
            }
        )
    if len(tx_truth_rows) != len(receivers):
        raise D105Phase1BundleError("truth scorer TX coverage drift")
    truth_open_sha = _sha256_bytes(_canonical_bytes(truth_open))
    score = {
        "schema": SOURCE_HELD_SCORE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": str(tap["checkpoint_sha256"]),
        "runtime_sha256": str(tap["runtime_sha256"]),
        "method_lock_sha256": str(tap["method_lock_sha256"]),
        "strict_tap_receipt_sha256": rows.strict_tap_receipt_sha256,
        "source_aggregate_lineage_sha256": lineage,
        "source_held_prediction_manifest_sha256": prediction_sha,
        "source_held_truth_open_receipt_sha256": truth_open_sha,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "scored_truth_rows": scored_truth_rows,
        "tx_probe_truth_rows": tx_truth_rows,
        "d102_rejected_bundle_reused": False,
    }
    path = Path(output_score_artifact)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise D105Phase1BundleError("truth score output must be a new child file")
    score_sha = _write_new_immutable_json(path, score)
    # Exercise the independent gate path before publishing a successful score.
    _source_held_gate_from_evidence(
        prediction,
        truth_open,
        score,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    return {
        "source_held_score_artifact": str(path),
        "source_held_score_artifact_sha256": score_sha,
        "source_held_prediction_manifest_sha256": prediction_sha,
        "source_held_truth_open_receipt_sha256": truth_open_sha,
        "target_rows": 0,
        "formal_phase2_eligible": False,
    }


def _source_subset_rows(
    rows: StrictTapRows, include: np.ndarray, *, scope: Mapping[str, Any]
) -> StrictTapRows:
    mask = np.asarray(include, dtype=bool)
    if mask.shape != (len(rows.pre_relu),) or int(np.sum(mask)) < DOMAIN_DIM + 2:
        raise D105Phase1BundleError("source-held fold aggregate subset is too small")
    physical = tuple(np.asarray(rows.physical_ids, dtype=np.str_)[mask].tolist())
    subset_archive_sha = _sha256_bytes(
        _canonical_bytes(
            {
                "schema": SCHEMA + ".source_held_fold_subset.v1",
                "parent_archive_sha256": rows.archive_sha256,
                "physical_root_sha256": _physical_root(physical),
                "scope": dict(scope),
            }
        )
    )
    return StrictTapRows(
        pre_relu=np.asarray(rows.pre_relu[mask], dtype=np.float32),
        z_dom=np.asarray(rows.z_dom[mask], dtype=np.float32),
        labels=tuple(np.asarray(rows.labels, dtype=np.str_)[mask].tolist()),
        receiver_ids=tuple(np.asarray(rows.receiver_ids, dtype=np.str_)[mask].tolist()),
        physical_ids=physical,
        archive_sha256=subset_archive_sha,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
    )


def _compile_source_held_fold_bundle(
    rows: StrictTapRows,
    tap: Mapping[str, Any],
    *,
    scope: Mapping[str, Any],
) -> RXIDMetaBias4Bundle:
    """Compile a transient, non-deployment cross-fit D105 aggregate bundle."""

    (
        u,
        b,
        bank_g,
        bank_t,
        precision,
        sigma,
        min_counts,
        class_counts,
        aggregation,
    ) = _build_aggregate_parameters(rows)
    scope_sha = _sha256_bytes(_canonical_bytes(dict(scope)))
    training = {
        "schema": SCHEMA + ".source_held_fold_training.v1",
        "strict_tap_receipt_sha256": rows.strict_tap_receipt_sha256,
        "fold_subset_sha256": rows.archive_sha256,
        "scope_sha256": scope_sha,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
    }
    nested = {
        "schema": SCHEMA + ".source_held_fold_nested.v1",
        "scope_sha256": scope_sha,
        "deployment_or_phase2_asset": False,
    }
    tx = {
        "schema": SCHEMA + ".source_held_fold_tx_placeholder.v1",
        "scope_sha256": scope_sha,
        "balanced_accuracy": 0.25,
        "used_for_gate": False,
    }
    provisional_quantization = {
        "schema": SCHEMA + ".source_held_fold_quantization.v1",
        "mode": "rxid_metabias4_row_int8_fp16_scale_and_log_int8",
        "persistent_fp32_sidecar": False,
    }
    common = {
        "cell_min_physical_count": min_counts,
        "cell_class_count": class_counts,
        "checkpoint_sha256": str(tap["checkpoint_sha256"]),
        "runtime_sha256": str(tap["runtime_sha256"]),
        "method_lock_sha256": str(tap["method_lock_sha256"]),
        "training_receipt_sha256": _sha256_bytes(_canonical_bytes(training)),
        "nested_receipt_sha256": _sha256_bytes(_canonical_bytes(nested)),
        "tx_probe_receipt_sha256": _sha256_bytes(_canonical_bytes(tx)),
        "aggregation_receipt_sha256": _sha256_bytes(_canonical_bytes(aggregation)),
        "tx_probe_mean_balanced_accuracy": 0.25,
        "tx_probe_max_balanced_accuracy": 0.25,
    }
    try:
        provisional = build_rxid_metabias4_bundle(
            u,
            b,
            bank_g,
            bank_t,
            precision,
            sigma,
            quantization_receipt_sha256=_sha256_bytes(
                _canonical_bytes(provisional_quantization)
            ),
            **common,
        )
        quantization = _quantization_summary(
            provisional,
            u=u,
            b=b,
            bank_g=bank_g,
            bank_t=bank_t,
            precision=precision,
            sigma=sigma,
        )
        return build_rxid_metabias4_bundle(
            u,
            b,
            bank_g,
            bank_t,
            precision,
            sigma,
            quantization_receipt_sha256=_sha256_bytes(
                _canonical_bytes(quantization)
            ),
            **common,
        )
    except RXIDMetaBias4BundleError as error:
        raise D105Phase1BundleError(
            "source-held transient D105 fold bundle compilation failed"
        ) from error


def _source_held_fold_handle(
    bundle: RXIDMetaBias4Bundle, *, scope: Mapping[str, Any]
) -> D105CBRCBundleHandle:
    """Give the non-deployment fold a typed integrity handle, never a seal."""

    scope_sha = _sha256_bytes(_canonical_bytes(dict(scope)))
    validated_id = _sha256_bytes(
        _canonical_bytes(
            {
                "schema": SCHEMA + ".source_held_non_deployment_handle.v1",
                "scope_sha256": scope_sha,
                "content_root_sha256": bundle.content_root_sha256,
            }
        )
    )
    receipt_root = compute_d105_bundle_receipt_root(bundle)
    validator = compute_d105_bundle_validator_receipt(
        validated_bundle_id_sha256=validated_id,
        expected_content_root_sha256=bundle.content_root_sha256,
        checkpoint_sha256=bundle.checkpoint_sha256,
        runtime_sha256=bundle.runtime_sha256,
        method_lock_sha256=bundle.method_lock_sha256,
        receipt_root_sha256=receipt_root,
    )
    try:
        return make_d105_cbrc_bundle_handle(
            bundle,
            validated_bundle_id_sha256=validated_id,
            validator_receipt_sha256=validator,
            expected_content_root_sha256=bundle.content_root_sha256,
        )
    except ValueError as error:
        raise D105Phase1BundleError("source-held fold handle construction failed") from error


def _source_held_lock(
    *, active_k: int, method_lock_sha256: str, strict_tap_receipt_sha256: str
) -> Any:
    from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock

    return Phase1ZIDStudentTLock(
        active_k=active_k,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256=method_lock_sha256,
        quantization_margin_audit_sha256=strict_tap_receipt_sha256,
    )


def _source_held_support_query_indices(
    rows: StrictTapRows, *, held_receiver: str, classes: Sequence[str], k_shot: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(rows.labels, dtype=np.str_)
    receivers = np.asarray(rows.receiver_ids, dtype=np.str_)
    physical = np.asarray(rows.physical_ids, dtype=np.str_)
    support: list[int] = []
    query: list[int] = []
    for label in classes:
        candidates = np.flatnonzero(
            (receivers == held_receiver) & (labels == str(label))
        )
        ordered = sorted(candidates.tolist(), key=lambda index: str(physical[index]))
        if len(ordered) < k_shot + 1:
            raise D105Phase1BundleError(
                f"source-held receiver/class lacks K{k_shot} support plus query"
            )
        support.extend(ordered[:k_shot])
        query.extend(ordered[k_shot:])
    support_rows = np.asarray(support, dtype=np.int64)
    query_rows = np.asarray(query, dtype=np.int64)
    if set(physical[support_rows]).intersection(physical[query_rows]):
        raise D105Phase1BundleError("source-held support/query physical overlap")
    return support_rows, query_rows


def _source_held_prediction_row(
    rows: StrictTapRows,
    tap: Mapping[str, Any],
    *,
    held_receiver: str,
    held_class: str | None,
    k_shot: int,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    """Execute one no-truth receiver-held or class-LOCO D105 prediction row."""

    from cvsrffi.stage2_d105_cbrc import (
        compute_d105_support_binding_root,
        fit_d105_cbrc_state,
        transform_d105_cbrc,
    )
    from cvsrffi.stage2_zid_student_t_qknn import (
        _score_with_support,
        build_typed_zid_support_bank,
        identity_shared_psd_metric,
        normalize_zid_rows,
        score_zid_student_t_logits,
    )

    labels = np.asarray(rows.labels, dtype=np.str_)
    receivers = np.asarray(rows.receiver_ids, dtype=np.str_)
    scope = {
        "fold_kind": "receiver_held" if held_class is None else "class_loco",
        "held_receiver_token": held_receiver,
        "held_class_token": held_class,
        "K": k_shot,
    }
    include = receivers != held_receiver
    if held_class is not None:
        include &= labels != held_class
    fold_rows = _source_subset_rows(rows, include, scope=scope)
    bundle = _compile_source_held_fold_bundle(fold_rows, tap, scope=scope)
    handle = _source_held_fold_handle(bundle, scope=scope)
    support_index, query_index = _source_held_support_query_indices(
        rows, held_receiver=held_receiver, classes=classes, k_shot=k_shot
    )
    physical = np.asarray(rows.physical_ids, dtype=np.str_)
    support_ids = tuple(physical[support_index].tolist())
    query_ids = tuple(physical[query_index].tolist())
    support_labels = tuple(labels[support_index].tolist())
    support_pre = np.asarray(rows.pre_relu[support_index], dtype=np.float32)
    support_dom = np.asarray(rows.z_dom[support_index], dtype=np.float32)
    support_receipt = compute_d105_support_binding_root(
        support_pre,
        support_dom,
        support_labels,
        support_ids,
        classes,
        classes,
        (),
        active_k=k_shot,
        stage="S_B",
    )
    state = fit_d105_cbrc_state(
        bundle,
        handle,
        support_pre,
        support_dom,
        support_labels,
        support_ids,
        classes,
        classes,
        (),
        active_k=k_shot,
        stage="S_B",
        support_receipt_sha256=support_receipt,
    )
    query_pre = np.asarray(rows.pre_relu[query_index], dtype=np.float32)
    base_support = normalize_zid_rows(np.maximum(support_pre, np.float32(0.0)))
    base_query = normalize_zid_rows(np.maximum(query_pre, np.float32(0.0)))
    da_support = transform_d105_cbrc(state, support_pre)
    da_query = transform_d105_cbrc(state, query_pre)
    lock = _source_held_lock(
        active_k=k_shot,
        method_lock_sha256=str(tap["method_lock_sha256"]),
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
    )
    metric = identity_shared_psd_metric(config=lock)
    base_bank = build_typed_zid_support_bank(
        base_support, support_labels, classes, config=lock
    )
    da_bank = build_typed_zid_support_bank(
        da_support, support_labels, classes, config=lock
    )
    m0_logits = score_zid_student_t_logits(base_bank, base_query, metric=metric)
    int8_logits = score_zid_student_t_logits(da_bank, da_query, metric=metric)
    class_index = {label: index for index, label in enumerate(da_bank.classes)}
    fp32_logits = _score_with_support(
        support=normalize_zid_rows(da_support).astype(np.float64),
        class_indices=np.asarray(
            [class_index[label] for label in support_labels], dtype=np.int16
        ),
        support_counts=da_bank.support_counts,
        class_scales=da_bank.class_scales_fp16.astype(np.float64),
        query=normalize_zid_rows(da_query).astype(np.float64),
        config=lock,
        metric=metric,
    )
    if not np.isfinite(fp32_logits).all() or fp32_logits.shape != int8_logits.shape:
        raise D105Phase1BundleError("source-held FP32/INT8 score closure drift")
    fp32_ordered = np.sort(np.asarray(fp32_logits, dtype=np.float64), axis=1)
    margins = fp32_ordered[:, -1] - fp32_ordered[:, -2]
    row = {
        "row_id": _sha256_bytes(
            _canonical_bytes(
                {
                    "schema": SOURCE_HELD_PREDICTION_SCHEMA + ".row_id.v1",
                    "scope": scope,
                    "query_physical_ids": list(query_ids),
                }
            )
        ),
        "fold_kind": scope["fold_kind"],
        "held_receiver_token": held_receiver,
        "held_class_token": held_class,
        "K": k_shot,
        "query_physical_ids": list(query_ids),
        "m0_predictions": [
            da_bank.classes[index] for index in np.argmax(m0_logits, axis=1).tolist()
        ],
        "d105_fp32_predictions": [
            da_bank.classes[index] for index in np.argmax(fp32_logits, axis=1).tolist()
        ],
        "d105_int8_predictions": [
            da_bank.classes[index] for index in np.argmax(int8_logits, axis=1).tolist()
        ],
        "d105_fp32_top2_margins": [float(item) for item in margins.tolist()],
        "prediction_commit_sha256": "0" * 64,
        "query_rows_used_for_fit": 0,
    }
    row["prediction_commit_sha256"] = compute_d105_source_held_prediction_commit(
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        source_aggregate_lineage_sha256=compute_d105_source_aggregate_lineage(
            strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
            checkpoint_sha256=str(tap["checkpoint_sha256"]),
            runtime_sha256=str(tap["runtime_sha256"]),
            method_lock_sha256=str(tap["method_lock_sha256"]),
        ),
        row=row,
    )
    return row


def _ridge_probe_predictions(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, class_count: int
) -> np.ndarray:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.int64)
    test = np.asarray(test_x, dtype=np.float64)
    if (
        x.ndim != 2
        or test.ndim != 2
        or x.shape[1] != test.shape[1]
        or len(x) != len(y)
        or class_count < 2
        or set(y.tolist()) != set(range(class_count))
    ):
        raise D105Phase1BundleError("source-held TX ridge probe input closure drift")
    one_hot = np.eye(class_count, dtype=np.float64)[y]
    ridge = 1.0e-2
    try:
        weight = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ one_hot)
    except np.linalg.LinAlgError as error:
        raise D105Phase1BundleError("source-held TX ridge probe solve failed") from error
    return np.asarray(np.argmax(test @ weight, axis=1), dtype=np.int64)


def _source_held_tx_prediction_row(
    rows: StrictTapRows,
    tap: Mapping[str, Any],
    *,
    held_receiver: str,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    labels = np.asarray(rows.labels, dtype=np.str_)
    receivers = np.asarray(rows.receiver_ids, dtype=np.str_)
    scope = {"fold_kind": "tx_probe", "held_receiver_token": held_receiver}
    fold_rows = _source_subset_rows(rows, receivers != held_receiver, scope=scope)
    bundle = _compile_source_held_fold_bundle(fold_rows, tap, scope=scope)
    u = np.asarray(bundle.decode_u(), dtype=np.float64)
    train_mask = receivers != held_receiver
    test_mask = ~train_mask
    train = np.asarray(rows.z_dom[train_mask], dtype=np.float64) @ u.T
    test = np.asarray(rows.z_dom[test_mask], dtype=np.float64) @ u.T
    for value in (train, test):
        norm = np.linalg.norm(value, axis=1, keepdims=True)
        if np.any(norm <= _EPS) or not np.isfinite(norm).all():
            raise D105Phase1BundleError("source-held TX probe domain norm drift")
        value /= norm
    class_index = {label: index for index, label in enumerate(classes)}
    train_y = np.asarray([class_index[item] for item in labels[train_mask]], dtype=np.int64)
    predicted = _ridge_probe_predictions(train, train_y, test, len(classes))
    physical = np.asarray(rows.physical_ids, dtype=np.str_)[test_mask]
    row = {
        "held_receiver_token": held_receiver,
        "physical_ids": physical.tolist(),
        "predictions": [classes[index] for index in predicted.tolist()],
        "prediction_commit_sha256": "0" * 64,
    }
    row["prediction_commit_sha256"] = compute_d105_source_held_tx_prediction_commit(
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        source_aggregate_lineage_sha256=compute_d105_source_aggregate_lineage(
            strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
            checkpoint_sha256=str(tap["checkpoint_sha256"]),
            runtime_sha256=str(tap["runtime_sha256"]),
            method_lock_sha256=str(tap["method_lock_sha256"]),
        ),
        row=row,
    )
    return row


def execute_d105_source_held_predictions(
    strict_tap_archive: str | Path,
    strict_tap_receipt: str | Path,
    candidate_method_lock: str | Path,
    candidate_runtime_manifest: str | Path,
    output_prediction_manifest: str | Path,
) -> dict[str, Any]:
    """Run the frozen D105 receiver-held/LOCO/TX prediction matrix without truth."""

    rows, tap = load_d105_strict_tap_rows(strict_tap_archive, strict_tap_receipt)
    _validate_candidate_lock_for_tap(
        candidate_method_lock, candidate_runtime_manifest, tap
    )
    receivers = tuple(sorted(set(rows.receiver_ids)))
    classes = tuple(sorted(set(rows.labels)))
    # A class-LOCO fold removes one class and one receiver from the aggregate
    # fit.  With only three receivers, the two remaining receiver residual
    # blocks provide at most three independent directions after the held class
    # is removed, whereas the frozen CBRC code has width four.  Refuse that
    # underidentified evidence surface instead of silently reducing the code.
    if len(receivers) < 4 or len(classes) < 4:
        raise D105Phase1BundleError(
            "source-held predictor requires at least four receivers and four classes"
        )
    prediction_rows = [
        _source_held_prediction_row(
            rows,
            tap,
            held_receiver=receiver,
            held_class=None,
            k_shot=k_shot,
            classes=classes,
        )
        for receiver in receivers
        for k_shot in (1, 5, 10)
    ]
    prediction_rows.extend(
        _source_held_prediction_row(
            rows,
            tap,
            held_receiver=receiver,
            held_class=label,
            k_shot=1,
            classes=classes,
        )
        for receiver in receivers
        for label in classes
    )
    tx_rows = [
        _source_held_tx_prediction_row(
            rows, tap, held_receiver=receiver, classes=classes
        )
        for receiver in receivers
    ]
    lineage = compute_d105_source_aggregate_lineage(
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    manifest = {
        "schema": SOURCE_HELD_PREDICTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": str(tap["checkpoint_sha256"]),
        "runtime_sha256": str(tap["runtime_sha256"]),
        "method_lock_sha256": str(tap["method_lock_sha256"]),
        "strict_tap_receipt_sha256": rows.strict_tap_receipt_sha256,
        "source_aggregate_lineage_sha256": lineage,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "receiver_tokens": list(receivers),
        "class_tokens": list(classes),
        "scored_prediction_rows": prediction_rows,
        "tx_probe_prediction_rows": tx_rows,
        "d102_rejected_bundle_reused": False,
    }
    _validate_prediction_manifest_for_source_rows(
        manifest,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    path = Path(output_prediction_manifest)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise D105Phase1BundleError("prediction manifest output must be a new child file")
    manifest_sha = _write_new_immutable_json(path, manifest)
    return {
        "source_held_prediction_manifest": str(path),
        "source_held_prediction_manifest_sha256": manifest_sha,
        "receiver_held_row_count": len(receivers) * 3,
        "class_loco_row_count": len(receivers) * len(classes),
        "tx_probe_row_count": len(receivers),
        "prediction_truth_present": False,
        "formal_phase2_eligible": False,
    }


def _canonical_svd_rows(
    values: np.ndarray, component_count: int, *, name: str
) -> np.ndarray:
    """Return deterministic right singular vectors as float32 rows."""

    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != Z_DIM
        or len(matrix) < component_count
        or not np.isfinite(matrix).all()
    ):
        raise D105Phase1BundleError(f"{name} SVD input contract drift")
    try:
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise D105Phase1BundleError(f"{name} SVD failed") from error
    if len(singular) < component_count or np.any(singular[:component_count] <= _EPS):
        raise D105Phase1BundleError(f"{name} has insufficient nonzero source rank")
    result = np.asarray(vh[:component_count], dtype=np.float64)
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    cast = np.ascontiguousarray(result, dtype=np.float32)
    if not np.isfinite(cast).all() or np.any(np.linalg.norm(cast, axis=1) <= 0.0):
        raise D105Phase1BundleError(f"{name} canonical SVD closure failed")
    return cast


def _unit_row(value: np.ndarray, *, name: str) -> np.ndarray:
    row = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(row))
    if not math.isfinite(norm) or norm <= _EPS:
        raise D105Phase1BundleError(f"{name} has a zero/invalid aggregate direction")
    return np.asarray(row / norm, dtype=np.float32)


def _build_aggregate_parameters(
    rows: StrictTapRows,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Create class-free receiver aggregates transiently from a strict tap."""

    labels = np.asarray(rows.labels, dtype=np.str_)
    receivers = np.asarray(rows.receiver_ids, dtype=np.str_)
    classes = tuple(sorted(np.unique(labels).tolist()))
    receiver_values = tuple(sorted(np.unique(receivers).tolist()))
    if len(classes) < 2 or len(receiver_values) < 2:
        raise D105Phase1BundleError(
            "D105 receiver aggregate requires at least two source classes/receivers"
        )
    class_pre: dict[str, np.ndarray] = {}
    class_dom: dict[str, np.ndarray] = {}
    class_cell_pre: dict[tuple[str, str], np.ndarray] = {}
    class_cell_dom: dict[tuple[str, str], np.ndarray] = {}
    class_cell_count: dict[tuple[str, str], int] = {}
    for label in classes:
        indices = np.flatnonzero(labels == label)
        class_pre[label] = np.mean(rows.pre_relu[indices], axis=0, dtype=np.float64)
        class_dom[label] = np.mean(rows.z_dom[indices], axis=0, dtype=np.float64)
        for receiver in receiver_values:
            local = indices[receivers[indices] == receiver]
            if len(local) < 2:
                raise D105Phase1BundleError(
                    "each receiver/class aggregate needs at least two physical rows"
                )
            class_cell_count[(receiver, label)] = int(len(local))
            class_cell_pre[(receiver, label)] = np.mean(
                rows.pre_relu[local], axis=0, dtype=np.float64
            )
            class_cell_dom[(receiver, label)] = np.mean(
                rows.z_dom[local], axis=0, dtype=np.float64
            )

    # Labels are used only to remove source identity trends during transient
    # Phase1 aggregation. No class label appears in the outgoing package.
    pre_residual_cells = np.asarray(
        [
            class_cell_pre[(receiver, label)] - class_pre[label]
            for receiver in receiver_values
            for label in classes
        ],
        dtype=np.float64,
    )
    dom_residual_rows = np.asarray(
        [
            rows.z_dom[index].astype(np.float64) - class_dom[labels[index]]
            for index in range(len(rows.z_dom))
        ],
        dtype=np.float64,
    )
    b = _canonical_svd_rows(pre_residual_cells, CODE_DIM, name="MetaBias B").T
    u = _canonical_svd_rows(dom_residual_rows, DOMAIN_DIM, name="domain encoder U")
    bank_g_rows: list[np.ndarray] = []
    bank_t_rows: list[np.ndarray] = []
    precision_rows: list[np.ndarray] = []
    sigma_rows: list[float] = []
    min_counts: list[int] = []
    class_counts: list[int] = []
    for receiver in receiver_values:
        encoded: list[np.ndarray] = []
        codes: list[np.ndarray] = []
        for label in classes:
            dom_delta = class_cell_dom[(receiver, label)] - class_dom[label]
            encoded.append(
                _unit_row(np.asarray(u, dtype=np.float64) @ dom_delta, name="bank g")
            )
            pre_delta = class_cell_pre[(receiver, label)] - class_pre[label]
            codes.append(
                np.asarray(
                    -(np.asarray(b, dtype=np.float64).T @ pre_delta),
                    dtype=np.float32,
                )
            )
        g = _unit_row(
            np.mean(np.asarray(encoded, dtype=np.float64), axis=0), name="bank g"
        )
        code_matrix = np.asarray(codes, dtype=np.float64)
        precision = np.clip(
            1.0 / (np.var(code_matrix, axis=0) + (20.0 ** -1)),
            0.05,
            20.0,
        )
        angular = [
            max(0.0, 1.0 - float(np.dot(np.asarray(item, dtype=np.float64), g)))
            for item in encoded
        ]
        sigma = float(
            np.clip(math.sqrt(float(np.mean(angular)) + _EPS), 0.05, 2.0)
        )
        bank_g_rows.append(g)
        bank_t_rows.append(
            np.mean(code_matrix, axis=0, dtype=np.float64).astype(np.float32)
        )
        precision_rows.append(np.asarray(precision, dtype=np.float32))
        sigma_rows.append(sigma)
        min_counts.append(
            min(class_cell_count[(receiver, label)] for label in classes)
        )
        class_counts.append(len(classes))

    aggregation = {
        "schema": SCHEMA + ".receiver_aggregate_receipt.v1",
        "source_row_count": int(len(rows.pre_relu)),
        "receiver_aggregate_cell_count": int(len(receiver_values)),
        "minimum_classes_per_receiver_aggregate": int(min(class_counts)),
        "minimum_physical_per_receiver_class": int(min(min_counts)),
        "class_balanced_receiver_aggregation": True,
        "class_free_payload": True,
        "payload_contains_class_handles": False,
        "payload_contains_receiver_names": False,
        "payload_contains_physical_ids": False,
        "source_row_features_retained": False,
        "source_archive_path_retained": False,
    }
    return (
        np.asarray(u, dtype=np.float32),
        np.asarray(b, dtype=np.float32),
        np.asarray(bank_g_rows, dtype=np.float32),
        np.asarray(bank_t_rows, dtype=np.float32),
        np.asarray(precision_rows, dtype=np.float32),
        np.asarray(sigma_rows, dtype=np.float32),
        np.asarray(min_counts, dtype=np.int16),
        np.asarray(class_counts, dtype=np.int16),
        aggregation,
    )


def _quantization_summary(
    bundle: RXIDMetaBias4Bundle,
    *,
    u: np.ndarray,
    b: np.ndarray,
    bank_g: np.ndarray,
    bank_t: np.ndarray,
    precision: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, Any]:
    values = (
        ("u_max_abs_error", bundle.decode_u(), u),
        ("b_max_abs_error", bundle.decode_b(), b),
        ("bank_g_max_abs_error", bundle.decode_bank_g(), bank_g),
        ("bank_t_max_abs_error", bundle.decode_bank_t(), bank_t),
        ("bank_precision_max_abs_error", bundle.decode_bank_precision(), precision),
        ("bank_sigma_max_abs_error", bundle.decode_bank_sigma(), sigma),
    )
    result: dict[str, Any] = {
        "schema": SCHEMA + ".quantization_receipt.v1",
        "mode": "rxid_metabias4_row_int8_fp16_scale_and_log_int8",
        "persistent_fp32_sidecar": False,
        "teacher_arrays_persisted": False,
    }
    for name, decoded, teacher in values:
        result[name] = float(
            np.max(
                np.abs(
                    np.asarray(decoded, dtype=np.float64)
                    - np.asarray(teacher, dtype=np.float64)
                )
            )
        )
    return result


def _component_manifest(
    *,
    bundle: RXIDMetaBias4Bundle,
    wire_sha256: str,
    strict_tap_receipt_sha256: str,
    source_held_gate_sha256: str,
    d105_candidate_method_lock_sha256: str,
    d105_candidate_runtime_manifest_sha256: str,
    d102_revocation_manifest_sha256: str,
    d102_revocation_signature_sha256: str,
    gate_summary: Mapping[str, Any],
    gate_missing: Sequence[str],
    aggregation: Mapping[str, Any],
    quantization: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        *gate_missing,
        "independent_review_p0_0_p1_0",
        "independent_phase2_authority_seal",
    ]
    ordered_missing = list(dict.fromkeys(missing))
    status = COMPONENT_STATUS if len(gate_missing) == 0 else DIAGNOSTIC_STATUS
    return {
        "schema": SCHEMA,
        "artifact_kind": "D105_PHASE1_COMPONENT",
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "status": status,
        "formal_phase2_eligible": False,
        "formal_phase2_eligibility_prerequisites": list(FORMAL_PREREQUISITES),
        "formal_phase2_eligibility_missing": ordered_missing,
        "bundle_wire_name": BUNDLE_WIRE_NAME,
        "bundle_wire_sha256": wire_sha256,
        "bundle_content_root_sha256": bundle.content_root_sha256,
        "bundle_numeric_state_bytes": bundle.numeric_state_bytes,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "runtime_sha256": bundle.runtime_sha256,
        "method_lock_sha256": bundle.method_lock_sha256,
        "d105_candidate_method_lock_sha256": d105_candidate_method_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": (
            d105_candidate_runtime_manifest_sha256
        ),
        "d102_revocation_manifest_sha256": d102_revocation_manifest_sha256,
        "d102_revocation_signature_sha256": d102_revocation_signature_sha256,
        "bundle_receipt_root_sha256": compute_d105_bundle_receipt_root(bundle),
        "strict_tap_receipt_sha256": strict_tap_receipt_sha256,
        "source_held_gate_receipt_sha256": source_held_gate_sha256,
        "training_receipt_sha256": bundle.training_receipt_sha256,
        "nested_receipt_sha256": bundle.nested_receipt_sha256,
        "tx_probe_receipt_sha256": bundle.tx_probe_receipt_sha256,
        "aggregation_receipt_sha256": bundle.aggregation_receipt_sha256,
        "quantization_receipt_sha256": bundle.quantization_receipt_sha256,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "raw_iq_retained": False,
        "clean_iq_retained": False,
        "source_row_features_retained": False,
        "source_replay_access": False,
        "source_archive_path_retained": False,
        "payload_contains_class_handles": False,
        "payload_contains_receiver_names": False,
        "payload_contains_physical_ids": False,
        "d102_rejected_bundle_reused": False,
        "receiver_aggregate_summary": dict(aggregation),
        "source_held_gate_summary": dict(gate_summary),
        "quantization_summary": dict(quantization),
    }


_COMPONENT_MANIFEST_FIELDS = {
    "schema",
    "artifact_kind",
    "candidate_id",
    "protocol_schema",
    "status",
    "formal_phase2_eligible",
    "formal_phase2_eligibility_prerequisites",
    "formal_phase2_eligibility_missing",
    "bundle_wire_name",
    "bundle_wire_sha256",
    "bundle_content_root_sha256",
    "bundle_numeric_state_bytes",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "d105_candidate_method_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d102_revocation_manifest_sha256",
    "d102_revocation_signature_sha256",
    "bundle_receipt_root_sha256",
    "strict_tap_receipt_sha256",
    "source_held_gate_receipt_sha256",
    "training_receipt_sha256",
    "nested_receipt_sha256",
    "tx_probe_receipt_sha256",
    "aggregation_receipt_sha256",
    "quantization_receipt_sha256",
    "source_only",
    "target_rows",
    "query_rows",
    "raw_iq_retained",
    "clean_iq_retained",
    "source_row_features_retained",
    "source_replay_access",
    "source_archive_path_retained",
    "payload_contains_class_handles",
    "payload_contains_receiver_names",
    "payload_contains_physical_ids",
    "d102_rejected_bundle_reused",
    "receiver_aggregate_summary",
    "source_held_gate_summary",
    "quantization_summary",
}


_SEALED_MANIFEST_FIELDS = _COMPONENT_MANIFEST_FIELDS | {
    "component_manifest_sha256",
    "authority_envelope_sha256",
    "authority_signature_sha256",
    "independent_review_receipt_sha256",
    "d102_revocation_signature_sha256",
    "authority_nonce_sha256",
    "authority_nonce_ledger_identity_sha256",
    "authority_run_id",
    "authority_git_commit",
    "validated_bundle_id_sha256",
    "validator_receipt_sha256",
}


def _validate_common_manifest(
    manifest: Mapping[str, Any], bundle: RXIDMetaBias4Bundle, wire: bytes
) -> None:
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("protocol_schema") != PROTOCOL_SCHEMA
        or manifest.get("bundle_wire_name") != BUNDLE_WIRE_NAME
        or manifest.get("bundle_wire_sha256") != _sha256_bytes(wire)
        or manifest.get("bundle_content_root_sha256") != bundle.content_root_sha256
        or manifest.get("bundle_numeric_state_bytes") != bundle.numeric_state_bytes
        or manifest.get("checkpoint_sha256") != bundle.checkpoint_sha256
        or manifest.get("runtime_sha256") != bundle.runtime_sha256
        or manifest.get("method_lock_sha256") != bundle.method_lock_sha256
        or manifest.get("d105_candidate_method_lock_sha256")
        != bundle.method_lock_sha256
        or manifest.get("d105_candidate_runtime_manifest_sha256")
        != bundle.runtime_sha256
        or not isinstance(manifest.get("d102_revocation_manifest_sha256"), str)
        or not isinstance(manifest.get("d102_revocation_signature_sha256"), str)
        or manifest.get("bundle_receipt_root_sha256")
        != compute_d105_bundle_receipt_root(bundle)
        or manifest.get("training_receipt_sha256") != bundle.training_receipt_sha256
        or manifest.get("nested_receipt_sha256") != bundle.nested_receipt_sha256
        or manifest.get("tx_probe_receipt_sha256") != bundle.tx_probe_receipt_sha256
        or manifest.get("aggregation_receipt_sha256")
        != bundle.aggregation_receipt_sha256
        or manifest.get("quantization_receipt_sha256")
        != bundle.quantization_receipt_sha256
        or manifest.get("source_only") is not True
        or manifest.get("target_rows") != 0
        or manifest.get("query_rows") != 0
        or manifest.get("raw_iq_retained") is not False
        or manifest.get("clean_iq_retained") is not False
        or manifest.get("source_row_features_retained") is not False
        or manifest.get("source_replay_access") is not False
        or manifest.get("source_archive_path_retained") is not False
        or manifest.get("payload_contains_class_handles") is not False
        or manifest.get("payload_contains_receiver_names") is not False
        or manifest.get("payload_contains_physical_ids") is not False
        or manifest.get("d102_rejected_bundle_reused") is not False
        or manifest.get("formal_phase2_eligibility_prerequisites")
        != list(FORMAL_PREREQUISITES)
    ):
        raise D105Phase1BundleError("D105 asset common manifest closure drift")
    for field in (
        "bundle_wire_sha256",
        "bundle_content_root_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "d105_candidate_method_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d102_revocation_manifest_sha256",
        "d102_revocation_signature_sha256",
        "bundle_receipt_root_sha256",
        "strict_tap_receipt_sha256",
        "source_held_gate_receipt_sha256",
        "training_receipt_sha256",
        "nested_receipt_sha256",
        "tx_probe_receipt_sha256",
        "aggregation_receipt_sha256",
        "quantization_receipt_sha256",
    ):
        _require_sha256(manifest.get(field), field)
    if not bundle.tx_probe_gate_pass:
        raise D105Phase1BundleError("serialized asset must pass the TX probe gate")
    _reject_d102(manifest, name="D105 asset manifest")


def _create_component_output(
    output_dir: str | Path,
    *,
    wire: bytes,
    manifest: Mapping[str, Any],
    source_held_gate: Mapping[str, Any],
    d102_revocation_manifest_bytes: bytes,
    d102_revocation_signature: bytes,
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        raise D105Phase1BundleError(f"output already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    manifest_bytes = _canonical_bytes(manifest) + b"\n"
    gate_bytes = _canonical_bytes(source_held_gate)
    _write_new(root / BUNDLE_WIRE_NAME, wire)
    _write_new(root / HELD_GATE_NAME, gate_bytes)
    _write_new(root / D102_REVOCATION_MANIFEST_NAME, d102_revocation_manifest_bytes)
    _write_new(root / D102_REVOCATION_SIGNATURE_NAME, d102_revocation_signature)
    _write_new(root / MANIFEST_NAME, manifest_bytes)
    _write_new(
        root / SEAL_NAME,
        f"{_sha256_bytes(manifest_bytes)}  {MANIFEST_NAME}\n".encode("ascii"),
    )
    return {
        "bundle_root": str(root),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "bundle_wire_sha256": _sha256_bytes(wire),
        "bundle_content_root_sha256": manifest["bundle_content_root_sha256"],
        "source_held_gate_receipt_sha256": _sha256_bytes(gate_bytes),
        "formal_phase2_eligible": False,
        "status": manifest["status"],
    }


def build_d105_phase1_component(
    strict_tap_archive: str | Path,
    strict_tap_receipt: str | Path,
    candidate_method_lock: str | Path,
    candidate_runtime_manifest: str | Path,
    source_held_prediction_manifest: str | Path,
    source_held_truth_open_receipt: str | Path,
    source_held_score_artifact: str | Path,
    d102_revocation_manifest: str | Path,
    d102_revocation_signature: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build one immutable non-formal D105 source aggregate component."""

    revocation = _load_d102_revocation(
        d102_revocation_manifest, d102_revocation_signature
    )
    rows, tap = load_d105_strict_tap_rows(strict_tap_archive, strict_tap_receipt)
    if tap["d102_revocation_manifest_sha256"] != revocation["manifest_sha256"]:
        raise D105Phase1BundleError("strict tap D102 revocation SHA256 drift")
    _reject_revoked_identity(
        revocation,
        tap_archive_sha256=rows.archive_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
    )
    candidate_identity = _validate_candidate_lock_for_tap(
        candidate_method_lock, candidate_runtime_manifest, tap
    )
    prediction = _load_immutable_json(
        source_held_prediction_manifest, name="source-held prediction manifest"
    )
    truth_open = _load_immutable_json(
        source_held_truth_open_receipt, name="source-held truth-open receipt"
    )
    score = _load_immutable_json(
        source_held_score_artifact, name="source-held score artifact"
    )
    gate, gate_missing, gate_summary = _source_held_gate_from_evidence(
        prediction,
        truth_open,
        score,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    _validate_derived_source_held_gate(
        gate,
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        checkpoint_sha256=str(tap["checkpoint_sha256"]),
        runtime_sha256=str(tap["runtime_sha256"]),
        method_lock_sha256=str(tap["method_lock_sha256"]),
    )
    gate_sha = _sha256_bytes(_canonical_bytes(gate))
    (
        u,
        b,
        bank_g,
        bank_t,
        precision,
        sigma,
        min_counts,
        class_counts,
        aggregation,
    ) = _build_aggregate_parameters(rows)
    training_receipt = {
        "schema": SCHEMA + ".training_receipt.v1",
        "candidate_id": CANDIDATE_ID,
        "strict_tap_receipt_sha256": rows.strict_tap_receipt_sha256,
        "tap_archive_sha256": rows.archive_sha256,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "aggregation_algorithm": "class_centered_receiver_aggregate_canonical_svd_v1",
        "source_row_features_retained": False,
        "d102_rejected_bundle_reused": False,
    }
    aggregation_sha = _sha256_bytes(_canonical_bytes(aggregation))
    nested_receipt = {
        "schema": SCHEMA + ".nested_source_held_binding.v1",
        "source_held_gate_receipt_sha256": gate_sha,
        "strict_tap_receipt_sha256": rows.strict_tap_receipt_sha256,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "d102_rejected_bundle_reused": False,
    }
    tx_probe_receipt = {
        "schema": SCHEMA + ".tx_probe_binding.v1",
        "source_held_gate_receipt_sha256": gate_sha,
        "mean_balanced_accuracy": gate_summary["tx_probe_mean_balanced_accuracy"],
        "max_balanced_accuracy": gate_summary["tx_probe_max_balanced_accuracy"],
        "gate_pass": gate_summary["tx_probe_gate_pass"],
    }
    provisional_quantization_sha = _sha256_bytes(
        _canonical_bytes(
            {
                "schema": SCHEMA + ".quantization_receipt_binding.v1",
                "algorithm": "rxid_metabias4_row_int8_fp16_scale_and_log_int8",
                "persistent_fp32_sidecar": False,
                "teacher_arrays_persisted": False,
            }
        )
    )
    try:
        bundle = build_rxid_metabias4_bundle(
            u,
            b,
            bank_g,
            bank_t,
            precision,
            sigma,
            cell_min_physical_count=min_counts,
            cell_class_count=class_counts,
            checkpoint_sha256=str(tap["checkpoint_sha256"]),
            runtime_sha256=str(tap["runtime_sha256"]),
            method_lock_sha256=str(tap["method_lock_sha256"]),
            training_receipt_sha256=_sha256_bytes(_canonical_bytes(training_receipt)),
            nested_receipt_sha256=_sha256_bytes(_canonical_bytes(nested_receipt)),
            tx_probe_receipt_sha256=_sha256_bytes(_canonical_bytes(tx_probe_receipt)),
            aggregation_receipt_sha256=aggregation_sha,
            quantization_receipt_sha256=provisional_quantization_sha,
            tx_probe_mean_balanced_accuracy=gate_summary[
                "tx_probe_mean_balanced_accuracy"
            ],
            tx_probe_max_balanced_accuracy=gate_summary[
                "tx_probe_max_balanced_accuracy"
            ],
        )
    except RXIDMetaBias4BundleError as error:
        raise D105Phase1BundleError("D105 aggregate bundle compilation failed") from error
    quantization = _quantization_summary(
        bundle,
        u=u,
        b=b,
        bank_g=bank_g,
        bank_t=bank_t,
        precision=precision,
        sigma=sigma,
    )
    quantization_sha = _sha256_bytes(_canonical_bytes(quantization))
    try:
        bundle = build_rxid_metabias4_bundle(
            u,
            b,
            bank_g,
            bank_t,
            precision,
            sigma,
            cell_min_physical_count=min_counts,
            cell_class_count=class_counts,
            checkpoint_sha256=str(tap["checkpoint_sha256"]),
            runtime_sha256=str(tap["runtime_sha256"]),
            method_lock_sha256=str(tap["method_lock_sha256"]),
            training_receipt_sha256=_sha256_bytes(_canonical_bytes(training_receipt)),
            nested_receipt_sha256=_sha256_bytes(_canonical_bytes(nested_receipt)),
            tx_probe_receipt_sha256=_sha256_bytes(_canonical_bytes(tx_probe_receipt)),
            aggregation_receipt_sha256=aggregation_sha,
            quantization_receipt_sha256=quantization_sha,
            tx_probe_mean_balanced_accuracy=gate_summary[
                "tx_probe_mean_balanced_accuracy"
            ],
            tx_probe_max_balanced_accuracy=gate_summary[
                "tx_probe_max_balanced_accuracy"
            ],
        )
        wire = serialize_rxid_metabias4_bundle(bundle)
    except RXIDMetaBias4BundleError as error:
        raise D105Phase1BundleError(
            "D105 aggregate bundle cannot be serialized after quantization closure"
        ) from error
    manifest = _component_manifest(
        bundle=bundle,
        wire_sha256=_sha256_bytes(wire),
        strict_tap_receipt_sha256=rows.strict_tap_receipt_sha256,
        source_held_gate_sha256=gate_sha,
        d105_candidate_method_lock_sha256=candidate_identity[
            "d105_candidate_method_lock_sha256"
        ],
        d105_candidate_runtime_manifest_sha256=str(tap["runtime_sha256"]),
        d102_revocation_manifest_sha256=str(revocation["manifest_sha256"]),
        d102_revocation_signature_sha256=str(revocation["signature_sha256"]),
        gate_summary=gate_summary,
        gate_missing=gate_missing,
        aggregation=aggregation,
        quantization=quantization,
    )
    return _create_component_output(
        output_dir,
        wire=wire,
        manifest=manifest,
        source_held_gate=gate,
        d102_revocation_manifest_bytes=revocation["manifest_bytes"],
        d102_revocation_signature=revocation["signature"],
    )


def _gate_missing_from_summary(gate: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if gate["receiver_held_complete"] is not True:
        missing.append("receiver_held_k1_k5_k10_complete")
    if gate["receiver_held_all_noninferior"] is not True:
        missing.append("receiver_held_all_noninferior")
    if (
        gate["class_loco_complete"] is not True
        or gate["class_loco_all_noninferior"] is not True
    ):
        missing.append("class_loco_complete_and_noninferior")
    if gate["tx_probe_gate_pass"] is not True:
        missing.append("tx_probe_max_balanced_accuracy_at_most_0_25")
    if gate["quantization_gate_pass"] is not True:
        missing.append("int8_agreement_at_least_0_995_and_zero_large_margin_flip")
    return missing


def _validate_persisted_gate(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    gate_path = root / HELD_GATE_NAME
    gate = _load_json(gate_path, name="persisted source-held gate receipt")
    summary = _validate_derived_source_held_gate(
        gate,
        strict_tap_receipt_sha256=str(manifest["strict_tap_receipt_sha256"]),
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        runtime_sha256=str(manifest["runtime_sha256"]),
        method_lock_sha256=str(manifest["method_lock_sha256"]),
    )
    if (
        _sha256_bytes(_canonical_bytes(gate))
        != manifest["source_held_gate_receipt_sha256"]
        or summary != manifest["source_held_gate_summary"]
    ):
        raise D105Phase1BundleError("persisted source-held gate/manifest drift")
    gate_missing = _gate_missing_from_summary(gate)
    expected_missing = [
        *gate_missing,
        "independent_review_p0_0_p1_0",
        "independent_phase2_authority_seal",
    ]
    expected_status = COMPONENT_STATUS if not gate_missing else DIAGNOSTIC_STATUS
    if (
        manifest["status"] != expected_status
        or manifest["formal_phase2_eligibility_missing"] != expected_missing
    ):
        raise D105Phase1BundleError("persisted source-held gate status drift")
    return gate


def _validate_persisted_d102_revocation(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    bundle_wire_sha256: str,
    bundle_content_root_sha256: str,
    component_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the copied signed D102 list and reject immutable-content reuse."""

    revocation = _load_d102_revocation(
        root / D102_REVOCATION_MANIFEST_NAME,
        root / D102_REVOCATION_SIGNATURE_NAME,
    )
    if (
        revocation["manifest_sha256"] != manifest["d102_revocation_manifest_sha256"]
        or revocation["signature_sha256"] != manifest[
            "d102_revocation_signature_sha256"
        ]
    ):
        raise D105Phase1BundleError("persisted D102 revocation binding drift")
    _reject_revoked_identity(
        revocation,
        bundle_manifest_sha256=component_manifest_sha256,
        bundle_payload_sha256=bundle_wire_sha256,
        bundle_content_root_sha256=bundle_content_root_sha256,
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        method_lock_sha256=str(manifest["method_lock_sha256"]),
        runtime_sha256=str(manifest["runtime_sha256"]),
    )
    return revocation


def _read_bundle_members(root: Path) -> tuple[dict[str, Any], bytes, str]:
    if not root.is_dir() or root.is_symlink():
        raise D105Phase1BundleError("asset root must be a normal directory")
    manifest_path = root / MANIFEST_NAME
    wire_path = root / BUNDLE_WIRE_NAME
    seal_path = root / SEAL_NAME
    for member in (manifest_path, wire_path, seal_path):
        if not member.is_file() or member.is_symlink():
            raise D105Phase1BundleError("asset member missing or symbolic link")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105Phase1BundleError("asset manifest JSON drift") from error
    if type(manifest) is not dict or manifest_bytes != _canonical_bytes(manifest) + b"\n":
        raise D105Phase1BundleError("asset manifest canonical encoding drift")
    manifest_sha = _sha256_bytes(manifest_bytes)
    tokens = seal_path.read_text(encoding="ascii").strip().split()
    if tokens != [manifest_sha, MANIFEST_NAME]:
        raise D105Phase1BundleError("asset manifest SHA256 seal drift")
    wire = wire_path.read_bytes()
    if not wire:
        raise D105Phase1BundleError("asset wire is empty")
    return manifest, wire, manifest_sha


def load_d105_phase1_asset(
    bundle_dir: str | Path, *, require_formal_phase2_eligible: bool = False
) -> D105Phase1Asset:
    """Load a component or formal asset with byte and binding checks."""

    root = Path(bundle_dir)
    manifest, wire, manifest_sha = _read_bundle_members(root)
    kind = manifest.get("artifact_kind")
    if kind == "D105_PHASE1_COMPONENT":
        _require_exact_keys(manifest, _COMPONENT_MANIFEST_FIELDS, "component manifest")
        if manifest.get("formal_phase2_eligible") is not False:
            raise D105Phase1BundleError("component cannot self-authorize Phase2")
        if manifest.get("status") not in (COMPONENT_STATUS, DIAGNOSTIC_STATUS):
            raise D105Phase1BundleError("component status drift")
        _validate_persisted_gate(root, manifest)
        extra = {item.name for item in root.iterdir()} - {
            BUNDLE_WIRE_NAME,
            MANIFEST_NAME,
            SEAL_NAME,
            HELD_GATE_NAME,
            D102_REVOCATION_MANIFEST_NAME,
            D102_REVOCATION_SIGNATURE_NAME,
        }
        if extra:
            raise D105Phase1BundleError("component artifact allowlist drift")
        formal = False
        validated_id = None
        validator_receipt = None
        component_manifest_sha_for_revocation = manifest_sha
    elif kind == "D105_PHASE1_FORMAL_SEALED":
        _require_exact_keys(manifest, _SEALED_MANIFEST_FIELDS, "sealed manifest")
        if (
            manifest.get("formal_phase2_eligible") is not True
            or manifest.get("status") != FORMAL_STATUS
        ):
            raise D105Phase1BundleError("formal seal status drift")
        component_path = root / COMPONENT_MANIFEST_NAME
        if not component_path.is_file() or component_path.is_symlink():
            raise D105Phase1BundleError("sealed asset lacks component manifest copy")
        component_bytes = component_path.read_bytes()
        try:
            component = json.loads(component_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise D105Phase1BundleError(
                "sealed component manifest copy drift"
            ) from error
        if (
            type(component) is not dict
            or component_bytes != _canonical_bytes(component) + b"\n"
            or _sha256_bytes(component_bytes)
            != manifest.get("component_manifest_sha256")
        ):
            raise D105Phase1BundleError("sealed component manifest binding drift")
        _require_exact_keys(
            component, _COMPONENT_MANIFEST_FIELDS, "copied component manifest"
        )
        if component.get("formal_phase2_eligible") is not False:
            raise D105Phase1BundleError("copied component formal status drift")
        if (
            component.get("bundle_wire_sha256") != manifest.get("bundle_wire_sha256")
            or component.get("bundle_content_root_sha256")
            != manifest.get("bundle_content_root_sha256")
            or component.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
            or component.get("runtime_sha256") != manifest.get("runtime_sha256")
            or component.get("method_lock_sha256")
            != manifest.get("method_lock_sha256")
            or component.get("d105_candidate_method_lock_sha256")
            != manifest.get("d105_candidate_method_lock_sha256")
            or component.get("d105_candidate_runtime_manifest_sha256")
            != manifest.get("d105_candidate_runtime_manifest_sha256")
            or component.get("d102_revocation_manifest_sha256")
            != manifest.get("d102_revocation_manifest_sha256")
            or component.get("d102_revocation_signature_sha256")
            != manifest.get("d102_revocation_signature_sha256")
            or component.get("strict_tap_receipt_sha256")
            != manifest.get("strict_tap_receipt_sha256")
            or component.get("source_held_gate_receipt_sha256")
            != manifest.get("source_held_gate_receipt_sha256")
        ):
            raise D105Phase1BundleError("sealed component/common binding drift")
        _validate_persisted_gate(root, component)
        extra = {item.name for item in root.iterdir()} - {
            BUNDLE_WIRE_NAME,
            MANIFEST_NAME,
            SEAL_NAME,
            COMPONENT_MANIFEST_NAME,
            HELD_GATE_NAME,
            D102_REVOCATION_MANIFEST_NAME,
            D102_REVOCATION_SIGNATURE_NAME,
            AUTHORITY_ENVELOPE_NAME,
            AUTHORITY_SIGNATURE_NAME,
            INDEPENDENT_REVIEW_RECEIPT_NAME,
        }
        if extra:
            raise D105Phase1BundleError("sealed artifact allowlist drift")
        formal = True
        validated_id = _require_sha256(
            manifest.get("validated_bundle_id_sha256"),
            "validated_bundle_id_sha256",
        )
        validator_receipt = _require_sha256(
            manifest.get("validator_receipt_sha256"), "validator_receipt_sha256"
        )
        component_manifest_sha_for_revocation = manifest["component_manifest_sha256"]
    else:
        raise D105Phase1BundleError("asset kind drift")
    try:
        bundle = deserialize_rxid_metabias4_bundle(wire)
    except RXIDMetaBias4BundleError as error:
        raise D105Phase1BundleError("asset wire deserialization failed") from error
    _validate_common_manifest(manifest, bundle, wire)
    revocation = _validate_persisted_d102_revocation(
        root,
        manifest,
        bundle_wire_sha256=_sha256_bytes(wire),
        bundle_content_root_sha256=bundle.content_root_sha256,
        component_manifest_sha256=component_manifest_sha_for_revocation,
    )
    if formal:
        _validate_formal_authority_package(
            root,
            manifest,
            bundle=bundle,
            validated_bundle_id_sha256=validated_id,
            validator_receipt_sha256=validator_receipt,
            d102_revocation_manifest_sha256=revocation["manifest_sha256"],
        )
    if require_formal_phase2_eligible and not formal:
        raise D105Phase1BundleError("formal Phase2 eligibility is required")
    return D105Phase1Asset(
        bundle=bundle,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        formal_phase2_eligible=formal,
        validated_bundle_id_sha256=validated_id,
        validator_receipt_sha256=validator_receipt,
    )


def _authority_identity(
    component: D105Phase1Asset, *, validated_bundle_id_sha256: str
) -> dict[str, str]:
    """Bind all component identities which must be inside the authority message."""

    manifest = _thaw(component.manifest)
    if manifest["status"] != COMPONENT_STATUS:
        raise D105Phase1BundleError(
            "source-held component is incomplete and cannot receive formal seal"
        )
    if set(manifest["formal_phase2_eligibility_missing"]) != {
        "independent_review_p0_0_p1_0",
        "independent_phase2_authority_seal",
    }:
        raise D105Phase1BundleError("component formal prerequisites are not closed")
    validated = _require_sha256(validated_bundle_id_sha256, "validated_bundle_id_sha256")
    if validated == manifest["bundle_content_root_sha256"]:
        raise D105Phase1BundleError("validated bundle ID may not equal content root")
    validator = compute_d105_bundle_validator_receipt(
        validated_bundle_id_sha256=validated,
        expected_content_root_sha256=manifest["bundle_content_root_sha256"],
        checkpoint_sha256=manifest["checkpoint_sha256"],
        runtime_sha256=manifest["runtime_sha256"],
        method_lock_sha256=manifest["method_lock_sha256"],
        receipt_root_sha256=manifest["bundle_receipt_root_sha256"],
    )
    return {
        "candidate_id": CANDIDATE_ID,
        "component_manifest_sha256": component.manifest_sha256,
        "bundle_wire_sha256": manifest["bundle_wire_sha256"],
        "bundle_content_root_sha256": manifest["bundle_content_root_sha256"],
        "bundle_receipt_root_sha256": manifest["bundle_receipt_root_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "runtime_sha256": manifest["runtime_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
        "d105_candidate_runtime_manifest_sha256": manifest[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": manifest[
            "d105_candidate_method_lock_sha256"
        ],
        "strict_tap_receipt_sha256": manifest["strict_tap_receipt_sha256"],
        "source_held_gate_receipt_sha256": manifest[
            "source_held_gate_receipt_sha256"
        ],
        "validated_bundle_id_sha256": validated,
        "validator_receipt_sha256": validator,
    }


def _validate_formal_authority_package(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    bundle: RXIDMetaBias4Bundle,
    validated_bundle_id_sha256: str,
    validator_receipt_sha256: str,
    d102_revocation_manifest_sha256: str,
) -> None:
    """Verify copied envelope, detached signature and independent review."""

    component_manifest_path = root / COMPONENT_MANIFEST_NAME
    component_payload = component_manifest_path.read_bytes()
    try:
        component_value = json.loads(component_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105Phase1BundleError("formal component manifest JSON drift") from error
    if (
        type(component_value) is not dict
        or component_payload != _canonical_bytes(component_value) + b"\n"
    ):
        raise D105Phase1BundleError("formal component manifest canonical drift")
    component_asset = D105Phase1Asset(
        bundle=bundle,
        manifest=component_value,
        manifest_sha256=_sha256_bytes(component_payload),
        formal_phase2_eligible=False,
        validated_bundle_id_sha256=None,
        validator_receipt_sha256=None,
    )
    identity = _authority_identity(
        component_asset, validated_bundle_id_sha256=validated_bundle_id_sha256
    )
    if identity["validator_receipt_sha256"] != validator_receipt_sha256:
        raise D105Phase1BundleError("formal validator receipt binding drift")
    try:
        review = load_independent_review_receipt(
            root / INDEPENDENT_REVIEW_RECEIPT_NAME, identity=identity
        )
        authority = load_signed_d105_authority_envelope(
            root / AUTHORITY_ENVELOPE_NAME,
            root / AUTHORITY_SIGNATURE_NAME,
            identity=identity,
            independent_review_receipt_sha256=review["receipt_sha256"],
            d102_revocation_manifest_sha256=d102_revocation_manifest_sha256,
        )
    except D105AuthorityError as error:
        raise D105Phase1BundleError("formal D105 authority package validation failed") from error
    envelope = authority["envelope"]
    expected = {
        "authority_envelope_sha256": authority["envelope_sha256"],
        "authority_signature_sha256": authority["signature_sha256"],
        "independent_review_receipt_sha256": review["receipt_sha256"],
        "authority_nonce_sha256": _sha256_bytes(str(envelope["nonce"]).encode("ascii")),
        "authority_nonce_ledger_identity_sha256": envelope[
            "nonce_ledger_identity_sha256"
        ],
        "authority_run_id": envelope["run_id"],
        "authority_git_commit": envelope["git_commit"],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise D105Phase1BundleError("formal authority envelope/manifest drift")


def seal_d105_phase1_component(
    component_dir: str | Path,
    authority_envelope: str | Path,
    authority_signature: str | Path,
    independent_review_receipt: str | Path,
    d102_revocation_manifest: str | Path,
    d102_revocation_signature: str | Path,
    nonce_ledger_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Copy a passing component into a signed, one-time formal Phase2 asset."""

    component = load_d105_phase1_asset(
        component_dir, require_formal_phase2_eligible=False
    )
    if component.formal_phase2_eligible:
        raise D105Phase1BundleError("authority seal input must be an unsealed component")
    revocation = _load_d102_revocation(
        d102_revocation_manifest, d102_revocation_signature
    )
    manifest = _thaw(component.manifest)
    if (
        revocation["manifest_sha256"] != manifest["d102_revocation_manifest_sha256"]
        or revocation["signature_sha256"] != manifest[
            "d102_revocation_signature_sha256"
        ]
    ):
        raise D105Phase1BundleError("formal seal D102 revocation SHA256 drift")
    envelope_preview = _load_json(authority_envelope, name="D105 authority envelope")
    try:
        identity = _authority_identity(
            component,
            validated_bundle_id_sha256=_require_sha256(
                envelope_preview.get("validated_bundle_id_sha256"),
                "validated_bundle_id_sha256",
            ),
        )
        review = load_independent_review_receipt(
            independent_review_receipt, identity=identity
        )
        authority = load_signed_d105_authority_envelope(
            authority_envelope,
            authority_signature,
            identity=identity,
            independent_review_receipt_sha256=review["receipt_sha256"],
            d102_revocation_manifest_sha256=revocation["manifest_sha256"],
            nonce_guard=set(),
        )
    except D105AuthorityError as error:
        raise D105Phase1BundleError("formal D105 authority seal validation failed") from error
    envelope = authority["envelope"]
    root = Path(output_dir)
    if root.name != envelope["run_id"]:
        raise D105Phase1BundleError("formal asset output directory must equal authority run_id")
    if root.exists() or root.is_symlink():
        raise D105Phase1BundleError(f"output already exists: {root}")
    try:
        consume_authority_nonce_once(
            nonce_ledger_dir,
            envelope=envelope,
            envelope_sha256=authority["envelope_sha256"],
        )
    except D105AuthorityError as error:
        raise D105Phase1BundleError("formal D105 authority nonce validation failed") from error
    source_root = Path(component_dir)
    component_manifest_bytes = (source_root / MANIFEST_NAME).read_bytes()
    wire = (source_root / BUNDLE_WIRE_NAME).read_bytes()
    gate_bytes = (source_root / HELD_GATE_NAME).read_bytes()
    manifest.update(
        {
            "artifact_kind": "D105_PHASE1_FORMAL_SEALED",
            "status": FORMAL_STATUS,
            "formal_phase2_eligible": True,
            "formal_phase2_eligibility_missing": [],
            "component_manifest_sha256": component.manifest_sha256,
            "authority_envelope_sha256": authority["envelope_sha256"],
            "authority_signature_sha256": authority["signature_sha256"],
            "independent_review_receipt_sha256": review["receipt_sha256"],
            "authority_nonce_sha256": _sha256_bytes(
                str(envelope["nonce"]).encode("ascii")
            ),
            "authority_nonce_ledger_identity_sha256": envelope[
                "nonce_ledger_identity_sha256"
            ],
            "authority_run_id": envelope["run_id"],
            "authority_git_commit": envelope["git_commit"],
            "validated_bundle_id_sha256": identity["validated_bundle_id_sha256"],
            "validator_receipt_sha256": identity["validator_receipt_sha256"],
        }
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest_bytes = _canonical_bytes(manifest) + b"\n"
    _write_new(root / BUNDLE_WIRE_NAME, wire)
    _write_new(root / HELD_GATE_NAME, gate_bytes)
    _write_new(root / COMPONENT_MANIFEST_NAME, component_manifest_bytes)
    _write_new(root / D102_REVOCATION_MANIFEST_NAME, revocation["manifest_bytes"])
    _write_new(root / D102_REVOCATION_SIGNATURE_NAME, revocation["signature"])
    _write_new(root / INDEPENDENT_REVIEW_RECEIPT_NAME, review["receipt_bytes"])
    _write_new(root / AUTHORITY_ENVELOPE_NAME, authority["envelope_bytes"])
    _write_new(root / AUTHORITY_SIGNATURE_NAME, authority["signature"])
    _write_new(root / MANIFEST_NAME, manifest_bytes)
    _write_new(
        root / SEAL_NAME,
        f"{_sha256_bytes(manifest_bytes)}  {MANIFEST_NAME}\n".encode("ascii"),
    )
    asset = load_d105_phase1_asset(root, require_formal_phase2_eligible=True)
    return {
        "bundle_root": str(root),
        "manifest_sha256": asset.manifest_sha256,
        "bundle_wire_sha256": _sha256_bytes(wire),
        "bundle_content_root_sha256": asset.bundle.content_root_sha256,
        "validated_bundle_id_sha256": asset.validated_bundle_id_sha256,
        "validator_receipt_sha256": asset.validator_receipt_sha256,
        "authority_envelope_sha256": authority["envelope_sha256"],
        "d102_revocation_manifest_sha256": revocation["manifest_sha256"],
        "formal_phase2_eligible": True,
        "status": FORMAL_STATUS,
    }


def make_d105_phase1_runtime_handle(asset: D105Phase1Asset) -> D105CBRCBundleHandle:
    """Return a typed D105 runtime handle only from a formal asset."""

    if type(asset) is not D105Phase1Asset or not asset.formal_phase2_eligible:
        raise D105Phase1BundleError(
            "only a formal D105 Phase1 asset may create a runtime handle"
        )
    try:
        return make_d105_cbrc_bundle_handle(
            asset.bundle,
            validated_bundle_id_sha256=str(asset.validated_bundle_id_sha256),
            validator_receipt_sha256=str(asset.validator_receipt_sha256),
            expected_content_root_sha256=asset.bundle.content_root_sha256,
        )
    except ValueError as error:
        raise D105Phase1BundleError(
            "typed D105 runtime handle construction failed"
        ) from error


def validate_d105_phase1_asset(
    bundle_dir: str | Path, *, require_formal_phase2_eligible: bool = False
) -> dict[str, Any]:
    """Return a small validation receipt with no source-row exposure."""

    asset = load_d105_phase1_asset(
        bundle_dir, require_formal_phase2_eligible=require_formal_phase2_eligible
    )
    manifest = _thaw(asset.manifest)
    return {
        "schema": SCHEMA + ".validation_receipt.v1",
        "artifact_kind": manifest["artifact_kind"],
        "manifest_sha256": asset.manifest_sha256,
        "bundle_content_root_sha256": asset.bundle.content_root_sha256,
        "checkpoint_sha256": asset.bundle.checkpoint_sha256,
        "runtime_sha256": asset.bundle.runtime_sha256,
        "method_lock_sha256": asset.bundle.method_lock_sha256,
        "d105_candidate_method_lock_sha256": manifest[
            "d105_candidate_method_lock_sha256"
        ],
        "d105_candidate_runtime_manifest_sha256": manifest[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "numeric_state_bytes": asset.bundle.numeric_state_bytes,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "raw_iq_retained": False,
        "source_replay_access": False,
        "formal_phase2_eligible": asset.formal_phase2_eligible,
        "validated_bundle_id_sha256": asset.validated_bundle_id_sha256,
    }


__all__ = [
    "AUTHORITY_SEAL_SCHEMA",
    "BUNDLE_WIRE_NAME",
    "CANDIDATE_ID",
    "COMPONENT_MANIFEST_NAME",
    "COMPONENT_STATUS",
    "D105Phase1Asset",
    "D105Phase1BundleError",
    "DIAGNOSTIC_STATUS",
    "FORMAL_PREREQUISITES",
    "FORMAL_STATUS",
    "MANIFEST_NAME",
    "PROTOCOL_SCHEMA",
    "SCHEMA",
    "SEAL_NAME",
    "SOURCE_HELD_GATE_SCHEMA",
    "SOURCE_HELD_PREDICTION_SCHEMA",
    "SOURCE_HELD_SCORE_SCHEMA",
    "SOURCE_HELD_TRUTH_OPEN_SCHEMA",
    "STRICT_TAP_MEMBERS",
    "STRICT_TAP_SCHEMA",
    "D105_STRICT_TAP_FORWARD_BATCH_CAPACITY",
    "D105_STRICT_TAP_FORWARD_BATCH_POLICY",
    "StrictTapRows",
    "build_d105_phase1_component",
    "build_d105_source_access_receipt",
    "compute_d105_source_held_prediction_commit",
    "compute_d105_source_held_tx_prediction_commit",
    "derive_d105_source_held_gate",
    "execute_d105_source_held_predictions",
    "export_d105_phase1_strict_tap",
    "load_d105_phase1_asset",
    "load_d105_candidate_method_lock",
    "load_d105_candidate_runtime_manifest",
    "load_d105_strict_tap_rows",
    "make_d105_phase1_runtime_handle",
    "open_d105_source_held_truth",
    "score_d105_source_held_truth",
    "seal_d105_phase1_component",
    "sha256_file",
    "validate_d105_phase1_asset",
]
