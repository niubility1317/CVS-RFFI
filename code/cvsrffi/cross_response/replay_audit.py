"""Disposable source replay instrumentation, not real-entrypoint acceptance.

Callers supply legal source data and the actual training/evaluation callbacks.
No checkpoint is loaded here and no audit branch is returned for promotion.
"""
from __future__ import annotations

import copy
import itertools
import random
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


def freeze(value):
    """Detach and own all storage; preserve None and tensor dtype/shape."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "state_dict"):
        return freeze(value.state_dict())
    if isinstance(value, Mapping):
        return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(freeze(v) for v in value)
    return copy.deepcopy(value)


def capture_rng_state(augmentation_rng=None):
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "cpu": torch.get_rng_state(),
              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if augmentation_rng is not None:
        if isinstance(augmentation_rng, torch.Generator):
            result["augmentation"] = augmentation_rng.get_state()
        elif isinstance(augmentation_rng, np.random.Generator):
            result["augmentation"] = augmentation_rng.bit_generator.state
        else:
            result["augmentation"] = augmentation_rng.getstate()
    return freeze(result)


def restore_rng_state(state, augmentation_rng=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if augmentation_rng is not None and "augmentation" in state:
        if isinstance(augmentation_rng, torch.Generator):
            augmentation_rng.set_state(state["augmentation"])
        elif isinstance(augmentation_rng, np.random.Generator):
            augmentation_rng.bit_generator.state = copy.deepcopy(state["augmentation"])
        else:
            augmentation_rng.setstate(state["augmentation"])


def _state(obj):
    return freeze(obj.state_dict() if hasattr(obj, "state_dict") else obj)


def snapshot_training_state(model, optimizer, *, scaler=None, ema=None,
                            prototypes=None, pseudo_state=None,
                            augmentation_rng=None, extra_state=None):
    return {"model": _state(model), "optimizer": _state(optimizer),
            "module_training": {n: m.training for n, m in model.named_modules()},
            "parameter_requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
            "scaler": _state(scaler), "ema": _state(ema),
            "prototypes": _state(prototypes), "pseudo_state": _state(pseudo_state),
            "rng": capture_rng_state(augmentation_rng), "extra_state": _state(extra_state)}


def capture_step_snapshot(step_id, *, inputs, batch_ticket, logits, base_losses,
                          model, optimizer, scaler=None, amp_skipped=None,
                          gradients_are_unscaled=True, **persistent):
    """Capture AFTER main unscale, BEFORE step; capture state again after step.

    Callers must pass each base objective, not just the total. Explicit stages
    permit snapshots at the actual point of execution rather than recomputation.
    """
    if not gradients_are_unscaled:
        raise ValueError("capture requires unscaled main gradients")
    gradients = {n: freeze(p.grad) for n, p in model.named_parameters()}
    return {"step_id": int(step_id), "inputs": freeze(inputs),
            "batch_ticket": freeze(batch_ticket), "logits": freeze(logits),
            "base_losses": freeze(base_losses), "unscaled_gradients": gradients,
            "amp": {"scale": float(scaler.get_scale()) if scaler else 1.,
                    "skipped": amp_skipped,
                    "finite": all(g is None or bool(torch.isfinite(g).all())
                                  for g in gradients.values())},
            "state": snapshot_training_state(model, optimizer, scaler=scaler, **persistent)}


def first_divergence(left, right, *, atol=0., rtol=0.):
    """Deterministic depth-first comparison; left mapping order is stage order.

    Nonfinite values are discrepancies even when both branches contain NaN.
    Relative error uses abs(left), with infinity for nonzero error at zero.
    """
    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be nonnegative")

    def walk(a, b, path):
        result = {"path": path, "reason": "value", "max_abs": None, "max_rel": None}
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                return dict(result, reason="keys", left_keys=list(a), right_keys=list(b))
            for key in a:
                found = walk(a[key], b[key], f"{path}.{key}" if path else str(key))
                if found:
                    return found
            return None
        if isinstance(a, (tuple, list)) and isinstance(b, type(a)):
            if len(a) != len(b):
                return dict(result, reason="length")
            for i, (x, y) in enumerate(zip(a, b)):
                found = walk(x, y, f"{path}[{i}]")
                if found:
                    return found
            return None
        if isinstance(a, (torch.Tensor, np.ndarray)):
            if type(a) is not type(b):
                return dict(result, reason="type")
            x, y = torch.as_tensor(a).cpu(), torch.as_tensor(b).cpu()
            if x.shape != y.shape or x.dtype != y.dtype:
                return dict(result, reason="shape_or_dtype")
            if x.numel() == 0:
                return None
            finite = torch.isfinite(x) & torch.isfinite(y)
            if not bool(finite.all()):
                return dict(result, reason="nonfinite")
            if torch.equal(x, y):
                return None
            if not x.is_floating_point() and not x.is_complex():
                # IDs, counters and RNG bytes are exact discrete state. Never
                # round uint64/int64 through float64 or relax with atol.
                pairs = list(zip(x.reshape(-1).tolist(), y.reshape(-1).tolist()))
                errors = [abs(int(u)-int(v)) for u, v in pairs]
                relative = [d/abs(int(u)) if u else (float("inf") if d else 0.)
                            for (u, _), d in zip(pairs, errors)]
                return dict(result, max_abs=max(errors), max_rel=max(relative))
            # Float64 error analysis avoids overflow in low precision or ints.
            x = x.to(torch.complex128 if x.is_complex() else torch.float64)
            y = y.to(x.dtype)
            delta = (x-y).abs()
            if bool((delta <= atol + rtol*x.abs()).all()):
                return None
            relative = torch.where(x.abs() > 0, delta/x.abs(),
                                   torch.where(delta == 0, 0., float("inf")))
            return dict(result, max_abs=float(delta.max()), max_rel=float(relative.max()))
        if type(a) is not type(b):
            return dict(result, reason="type")
        if isinstance(a, (float, int, np.number)) and not isinstance(a, bool):
            return walk(np.asarray(a), np.asarray(b), path)
        return None if a == b else result

    result = walk(left, right, "")
    if result:
        result["stage"] = result["path"].split(".")[0]
        result["step_id"] = left.get("step_id") if isinstance(left, Mapping) else None
        result["atol"], result["rtol"] = atol, rtol
    return result


def weighted_gradient_audit(losses, named_parameters, weights, *, shared_parameter_names):
    """Autograd reachability is grad-is-not-None, including numerical zero.

    Only jointly reachable explicitly selected representation parameters enter
    norms and cosines. Classifier-only and other parameters remain disclosed.
    Does not populate or modify parameter .grad buffers.
    """
    params = dict(named_parameters)
    selected = set(shared_parameter_names)
    if not selected <= set(params):
        raise ValueError("unknown shared parameter")
    active = [(n, p) for n, p in params.items() if p.requires_grad]
    grads = {}
    for key, loss in losses.items():
        values = (torch.autograd.grad(loss * weights[key], [p for _, p in active],
                                     allow_unused=True, retain_graph=True)
                  if loss.requires_grad and active else [None] * len(active))
        grads[key] = dict(zip((n for n, _ in active), values))
    common = sorted(n for n in selected if all(g.get(n) is not None for g in grads.values()))
    norms, vectors = {}, {}
    for key, mapping in grads.items():
        vector = torch.cat([mapping[n].detach().reshape(-1).double().cpu() for n in common]) if common else torch.empty(0, dtype=torch.float64)
        vectors[key] = vector
        norms[key] = float(vector.norm())
    pairs = {}
    for a, b in itertools.combinations(grads, 2):
        denom = norms[a]*norms[b]
        pairs[f"{a}/{b}"] = {"norm_ratio": norms[a]/norms[b] if norms[b] else None,
                              "cosine": float(torch.dot(vectors[a], vectors[b]))/denom if denom else None}
    return {"common_parameter_names": common, "weighted_norms": norms, "pairs": pairs,
            "reachability": {key: {n: ("none" if g.get(n) is None else
                            "zero" if not bool(torch.count_nonzero(g[n])) else "nonzero")
                            for n in params} for key, g in grads.items()},
            "excluded_parameter_names": sorted(set(params)-set(common))}


def summarize_gradient_audits(reports):
    """Keep mean step ratios distinct from ratio of epoch mean norms."""
    reports = list(reports)
    if not reports:
        return {"audited_steps": 0, "pairs": {}}
    pairs = {}
    for name in reports[0]["pairs"]:
        a, b = name.split("/")
        ratios = [r["pairs"][name]["norm_ratio"] for r in reports
                  if r["pairs"][name]["norm_ratio"] is not None]
        numerator = sum(r["weighted_norms"][a] for r in reports)
        denominator = sum(r["weighted_norms"][b] for r in reports)
        pairs[name] = {"mean_step_norm_ratio": sum(ratios)/len(ratios) if ratios else None,
                       "valid_ratio_steps": len(ratios),
                       "ratio_of_mean_norms": numerator/denominator if denominator else None}
    return {"audited_steps": len(reports), "pairs": pairs}


def grouped_source_risk(logits, labels, receivers, conditions):
    """Full-class inference CE/accuracy/margin, grouped by source RX/condition."""
    logits = logits.detach().float()
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or logits.shape[1] < 2 or not len(labels):
        raise ValueError("nonempty full-class logits required")
    if len(receivers) != len(labels) or len(conditions) != len(labels):
        raise ValueError("source metadata length mismatch")
    ce = F.cross_entropy(logits, labels, reduction="none")
    competitors = logits.clone().scatter_(1, labels[:, None], -float("inf"))
    margin = logits.gather(1, labels[:, None]).squeeze(1)-competitors.max(1).values
    correct = logits.argmax(1).eq(labels).float()
    def summarize(ids):
        return {"n": len(ids), "ce": float(ce[ids].mean()),
                "accuracy": float(correct[ids].mean()), "margin": float(margin[ids].mean())}
    groups = {}
    for rx, condition in sorted(set(zip(map(str, receivers), map(str, conditions)))):
        ids = [i for i, (r, c) in enumerate(zip(receivers, conditions)) if str(r) == rx and str(c) == condition]
        groups[f"rx={rx}/condition={condition}"] = summarize(ids)
    return {"overall": summarize(list(range(len(labels)))), "groups": groups,
            "worst_group_ce": max(g["ce"] for g in groups.values()),
            "worst_group_accuracy": min(g["accuracy"] for g in groups.values())}


def disposable_counterfactual(model, optimizer, *, train_step, evaluate,
                              train_physical_ids, eval_physical_ids,
                              train_role, eval_role, persistent_state=None,
                              augmentation_rng=None, probe=None):
    """Clone A/B from identical AdamW and all supplied persistent state.

    train_step(model, optimizer, state, include_response) executes ONE actual
    update; evaluate(model, state) returns grouped_source_risk-compatible data.
    State must contain scaler/EMA/prototypes/pseudo state used by the callback.
    Inputs/tickets should be owned by that state, never mutable callback closures.
    Optional probe(model,state) computes next-base orthogonality and other losses
    and gradients on the same fixed source probe batch; run in isolated copies.
    Physical IDs refer to raw recordings: augmentation views must share IDs.
    The caller owns source-role/provenance validation; this checks its declaration.
    """
    if train_role != "source_train" or eval_role != "source_validation":
        raise ValueError("counterfactual requires source_train/source_validation")
    if len(train_physical_ids) == 0 or len(eval_physical_ids) == 0:
        raise ValueError("physical ID sets must be nonempty")
    if set(train_physical_ids) & set(eval_physical_ids):
        raise ValueError("train/eval physical IDs overlap")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("actual AdamW update required")
    caller_rng = capture_rng_state(augmentation_rng)
    # Copy as ONE graph so copied optimizer parameters alias copied model.
    template = copy.deepcopy((model, optimizer, persistent_state, augmentation_rng))
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    results = {}
    try:
        for branch, response in (("initial", None), ("base", False), ("base_response", True)):
            branch_model, branch_optimizer, state, aug_rng = copy.deepcopy(template)
            restore_rng_state(caller_rng, aug_rng)
            # Give callbacks access to their copied augmentation generator.
            if isinstance(state, dict) and aug_rng is not None:
                state["augmentation_rng"] = aug_rng
            evidence = None if response is None else train_step(branch_model, branch_optimizer, state, response)
            final_parameters = {n: p.detach().cpu().clone() for n, p in branch_model.named_parameters()}
            deltas = {n: final_parameters[n]-initial[n] for n in initial}
            # Evaluation consumes its own copy: no BN/prototype/RNG mutation can
            # alter the recorded post-update training state or the other branch.
            post = snapshot_training_state(branch_model, branch_optimizer, extra_state=state, augmentation_rng=aug_rng)
            probe_template = copy.deepcopy((branch_model, branch_optimizer, state, aug_rng)) if probe is not None else None
            restore_rng_state(caller_rng, aug_rng)
            branch_model.eval()
            with torch.no_grad():
                risk = evaluate(branch_model, state)
            probe_result = None
            if probe is not None:
                # Use the pristine post-update training state, before evaluation.
                probe_model, _, probe_state, probe_rng = copy.deepcopy(probe_template)
                restore_rng_state(caller_rng, probe_rng)
                probe_result = freeze(probe(probe_model, probe_state))
            results[branch] = {"risk": freeze(risk), "parameter_delta": deltas,
                               "delta_norm": float(torch.cat([d.reshape(-1).double() for d in deltas.values()]).norm()),
                               "post_state": post, "step_evidence": freeze(evidence),
                               "indirect_coupling_probe": probe_result}
    finally:
        restore_rng_state(caller_rng, augmentation_rng)
    results["metadata"] = {"scope": "disposable_source_one_step_utility",
                           "train_unique_physical_ids": len(set(train_physical_ids)),
                           "eval_unique_physical_ids": len(set(eval_physical_ids)),
                           "branches_promoted": False}
    return results
