from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch


FCR_BUNDLE_SCHEMA = "cvs.phase1.adv3b02_fcr.bundle.v1"
FCR_FEATURE_SCHEMA = "ADV3B02:FCR:z_f_id:unit_l2:160:v1"
FCR_PHYSICAL_BASIS_ID = "fixed_response_basis:pa_conjugate_memory4:v1"
FCR_INPUT_NORMALIZATION_VERSION = "adv3b02_input_iq:v1"
FCR_FISHER_GATE_ID = "FisherIdentifiabilityGate:v1"
FCR_NUISANCE_SCHEMA_VERSION = "structured_nuisance:16_8_6_3:v1"


def _unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def export_fcr_bundle(model) -> dict[str, Any]:
    """Export the small, serializable identity contract for an FCR state dict."""

    state_model = _unwrap_model(model)
    if not bool(getattr(state_model, "use_fcr", False)):
        raise ValueError("fcr_bundle is defined only for a use_fcr=True model")
    config = getattr(state_model, "fcr_config", None)
    fcr = getattr(state_model, "fcr", None)
    if config is None or fcr is None:
        raise ValueError("use_fcr=True requires fcr_config and the FCR module")
    fisher_gate = getattr(fcr, "fisher_gate", None)
    fisher_eps = float(getattr(fisher_gate, "eps", 1e-8))
    trainable_fisher = 0 if fisher_gate is None else sum(
        int(parameter.numel()) for parameter in fisher_gate.parameters()
    )
    bundle = {
        "bundle_schema": FCR_BUNDLE_SCHEMA,
        "feature_schema": FCR_FEATURE_SCHEMA,
        "fcr_config": asdict(config),
        "physical_basis": {
            "identifier": FCR_PHYSICAL_BASIS_ID,
            "terms": ["s", "conjugate_s", "s_abs2", "delay1_abs2"],
            "trainable": False,
        },
        "input_normalization": {
            "version": FCR_INPUT_NORMALIZATION_VERSION,
            "description": (
                "WiSig per-record RMS-power IQ normalization without centering, "
                "followed by the deterministic bounded FCR gain/phase/CFO canonicalizer"
            ),
            "iq_layout": "float32:[batch,2,input_len]",
        },
        "fisher_gate": {
            "identifier": FCR_FISHER_GATE_ID,
            "deterministic": trainable_fisher == 0,
            "trainable_parameters": trainable_fisher,
            "eps": fisher_eps,
        },
        "nuisance_schema": {
            "order": ["channel", "receiver", "sync", "gain"],
            "dimensions": {
                "channel": int(config.channel_dim),
                "receiver": int(config.receiver_dim),
                "sync": int(config.sync_dim),
                "gain": int(config.gain_dim),
            },
            "version": FCR_NUISANCE_SCHEMA_VERSION,
        },
        "routing": {
            "fingerprint_excitation": "content.s_hat.detach()",
            "version": "task9_fingerprint_excitation_detach:v1",
            "single_view_inference": True,
        },
        "model_identity": {
            "candidate": "ADV3B02-FCR",
            "identity_dimension": 160,
            "feature_normalization": "unit_l2",
        },
    }
    return bundle


def validate_fcr_bundle_for_model(payload: Mapping[str, Any], model) -> None:
    """Reject incompatible identity metadata before loading candidate weights."""

    state_model = _unwrap_model(model)
    if not bool(getattr(state_model, "use_fcr", False)):
        return
    bundle = payload.get("fcr_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("FCR checkpoint requires a mapping fcr_bundle")
    expected = export_fcr_bundle(state_model)
    for key in (
        "bundle_schema",
        "feature_schema",
        "fcr_config",
        "physical_basis",
        "input_normalization",
        "fisher_gate",
        "nuisance_schema",
        "routing",
        "model_identity",
    ):
        if bundle.get(key) != expected[key]:
            raise ValueError(f"incompatible fcr_bundle {key}")


def load_fcr_checkpoint_strict(
    path: str | os.PathLike[str],
    model,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Validate FCR identity first, then restore the complete model strictly."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError("checkpoint must contain a model state mapping")
    validate_fcr_bundle_for_model(payload, model)
    _unwrap_model(model).load_state_dict(payload["model"], strict=True)
    return payload


class AveragedModelState:
    """EMA/SWA/SWAD-style online weight averaging."""

    def __init__(self, mode: str, decay: float = 0.999):
        self.mode = str(mode)
        self.decay = float(decay)
        self.n = 0
        self.avg: Dict[str, torch.Tensor] = {}
        self.non_float: Dict[str, torch.Tensor] = {}
        self.epochs: List[int] = []

    def update(self, model, epoch: int, *, ema: bool = False) -> None:
        state = getattr(model, "_orig_mod", model).state_dict()
        with torch.no_grad():
            for k, v in state.items():
                vv = v.detach()
                if torch.is_floating_point(vv):
                    vf = vv.float().clone()
                    if k not in self.avg:
                        self.avg[k] = vf
                    elif ema:
                        self.avg[k].mul_(float(self.decay)).add_(vf, alpha=1.0 - float(self.decay))
                    else:
                        self.avg[k].mul_(float(self.n) / float(self.n + 1)).add_(vf, alpha=1.0 / float(self.n + 1))
                else:
                    self.non_float[k] = vv.clone()
        self.n += 1
        self.epochs.append(int(epoch))

    def has_state(self) -> bool:
        return self.n > 0 and len(self.avg) > 0

    def averaged_state_dict(self, model) -> Dict[str, torch.Tensor]:
        ref_state = getattr(model, "_orig_mod", model).state_dict()
        out = {}
        for k, v in ref_state.items():
            if k in self.avg:
                out[k] = self.avg[k].to(device=v.device, dtype=v.dtype)
            elif k in self.non_float:
                out[k] = self.non_float[k].to(device=v.device, dtype=v.dtype)
            else:
                out[k] = v.detach().clone()
        return out

    def cpu_state_dict(self, model) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.averaged_state_dict(model).items()}


def save_checkpoint(path: str, *, model, optimizer, scheduler, scaler, epoch: int, args, split_info, stats: dict):
    parent = os.path.dirname(os.path.abspath(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    state_model = _unwrap_model(model)
    payload = {
        "model": state_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "args": vars(args),
        "split_info": split_info,
        "stats": stats,
    }
    if bool(getattr(state_model, "use_fcr", False)):
        payload["fcr_bundle"] = export_fcr_bundle(state_model)
    torch.save(payload, path)


def derive_checkpoint_path(base_path: str, suffix: str) -> str:
    """Derive a checkpoint path when user does not provide one explicitly.

    Example:
      best_model.pth + test_overall -> best_model_test_overall.pth
    """
    base_path = str(base_path).strip() or "best_model.pth"
    root, ext = os.path.splitext(base_path)
    if ext == "":
        ext = ".pth"
    return f"{root}_{suffix}{ext}"


def default_is_path(p: str, default_name: str) -> bool:
    return str(p).strip() == default_name

