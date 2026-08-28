"""Real-checkpoint, support-only smoke for the H6 cached deployment path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_structured_late_block_runner import _load_frozen_checkpoint  # noqa: E402
from cvsrffi.target_only_progressive_adapt import (  # noqa: E402
    ProgressiveTrainabilityPolicy,
    SFTAPFTConfig,
    TargetPrototypeHead,
    _extract_joint_embedding,
    _forward_aux,
    _source_classifier_weight,
    _target_prototypes,
    build_h6_prefix_cache,
    ensure_time_adapter,
    fit_sf_tapft,
    forward_h6_prefix_cache,
)
from cvsrffi.target_only_progressive_runner import _load_target_support  # noqa: E402


def _norm_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and (name.endswith("t3.norm.weight") or name.endswith("t3.norm.bias"))
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    model = _load_frozen_checkpoint(args.checkpoint, device=device)
    support = _load_target_support(args.support)
    ensure_time_adapter(model, rank=16)
    policy = ProgressiveTrainabilityPolicy(
        "p1_head_norm", norm_rules=(("t3", "weight_bias"),)
    )
    policy.apply(model, "A")
    values = support.received_iq.to(device=device, dtype=next(model.parameters()).dtype)
    labels = support.labels.to(device=device)
    with torch.no_grad():
        initial = _extract_joint_embedding(_forward_aux(model, values), int(values.size(0)))
        class_ids = support.class_ids
        prototypes = _target_prototypes(initial, labels, class_ids)
        head = TargetPrototypeHead.from_source_and_target(
            source_weights=_source_classifier_weight(model).to(initial),
            target_prototypes=prototypes,
            source_class_ids=tuple(range(_source_classifier_weight(model).size(0))),
            target_class_ids=class_ids,
            rho=0.5,
            scale=8.0,
        ).to(device=device, dtype=initial.dtype)

    full_embedding = _extract_joint_embedding(_forward_aux(model, values), int(values.size(0)))
    full_logits = head(full_embedding)
    full_logits.square().mean().backward()
    full_grad = _norm_gradients(model)
    for parameter in model.parameters():
        parameter.grad = None
    for parameter in head.parameters():
        parameter.grad = None

    cache32 = build_h6_prefix_cache(model, values, storage_dtype=torch.float32)
    cached32_logits = head(forward_h6_prefix_cache(model, cache32))
    cached32_logits.square().mean().backward()
    cached32_grad = _norm_gradients(model)
    logit_delta = float(torch.max(torch.abs(full_logits.detach() - cached32_logits.detach())))
    gradient_delta = max(
        float(torch.max(torch.abs(full_grad[name] - cached32_grad[name])))
        for name in full_grad
    )
    prediction_equal = bool(
        torch.equal(full_logits.argmax(dim=1), cached32_logits.argmax(dim=1))
    )
    for parameter in model.parameters():
        parameter.grad = None
    for parameter in head.parameters():
        parameter.grad = None
    cache16_storage = build_h6_prefix_cache(
        model,
        values,
        storage_dtype=torch.float16,
    )
    cache16 = cache16_storage.materialize_once(
        device=values.device,
        dtype=values.dtype,
    )
    with torch.no_grad():
        cached16_logits = head(forward_h6_prefix_cache(model, cache16))
    fp16_finite = bool(torch.isfinite(cached16_logits).all())

    one_step = fit_sf_tapft(
        model,
        support,
        SFTAPFTConfig(
            adapter_rank=16,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            phase_steps=(1, 0, 0),
            scheduler_reference_steps=1,
            validation_steps=(),
            prefix_cache_dtype="float32",
            checkpoint_average_top_k=1,
            mixed_precision=device.type == "cuda",
            seed=392002,
        ),
        checkpoint_selection_mode="final_step",
    )
    payload = {
        "status": "SMOKE_PASS",
        "support_count": len(support.physical_ids),
        "class_ids": list(support.class_ids),
        "fp32_max_abs_logit_delta": logit_delta,
        "fp32_max_abs_gradient_delta": gradient_delta,
        "fp32_prediction_equal": prediction_equal,
        "fp16_finite": fp16_finite,
        "prefix_cache_build_forward_steps": (
            one_step.audit.prefix_cache_build_forward_steps
        ),
        "backbone_train_forward_steps": one_step.audit.backbone_train_forward_steps,
        "cached_suffix_forward_steps": one_step.audit.cached_suffix_forward_steps,
        "query_opened": one_step.audit.query_opened,
    }
    if (
        logit_delta >= 1.0e-5
        or gradient_delta >= 1.0e-5
        or not prediction_equal
        or not fp16_finite
        or one_step.audit.query_opened
    ):
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
