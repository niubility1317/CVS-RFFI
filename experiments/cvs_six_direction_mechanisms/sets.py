"""Conservative set pseudo-label source diagnostic with hidden U truth.

Every teacher set is constructed using probabilities alone. Source labels are
joined only afterwards for L/V diagnostics. This module never updates identity.
"""
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from . import design as d

RECIPE = dict(classes=6, cumulative_mass=.95, teacher_views=["clean", "fixed_gaussian_noise_std0.005"],
              second_teacher_noise_std=.005, strong_noise_std=.015,
              set_composition="union of independently computed minimal cumulative prefixes",
              teacher_target="detached arithmetic mean of two fixed views",
              participation="union set size <=3; identical hard/soft/set mask",
              loss_denominator="entire role batch count, including zeroed nonparticipants",
              confidence_gate=False, parameter_updates=0,
              noise_seed_offsets=dict(L_s=4600, V=4601, U_s=4602))
CRITERIA = dict(L_fold2_minimum_ambiguous_small_sets=20,
                L_fold2_minimum_ambiguous_fraction=.01,
                L_fold2_minimum_top1_errors=10,
                L_fold2_small_set_truth_coverage=.95,
                L_fold2_ambiguous_top1_error_truth_inclusion=.5,
                L_fold2_maximum_full_set_fraction=.1)


def _check_probabilities(probabilities):
    if probabilities.ndim != 2 or probabilities.shape[1] != 6 or not len(probabilities):
        raise ValueError("Nonempty six-class probability tensor required")
    q = probabilities.detach()
    if not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("Finite nonnegative teacher probabilities required")
    if not torch.allclose(q.sum(1), torch.ones(len(q), dtype=q.dtype, device=q.device), atol=1e-5):
        raise ValueError("Teacher probabilities must sum to one")
    return q


def prefix_sets(probabilities, mass=.95):
    """Smallest stable probability-ordered prefix reaching the fixed mass."""
    q = _check_probabilities(probabilities)
    if not 0 < mass <= 1:
        raise ValueError("Cumulative mass must be in (0,1]")
    values, indices = torch.sort(q, dim=1, descending=True, stable=True)
    # Include the first item crossing the threshold, not just items below it.
    included = values.cumsum(1) - values < mass
    mask = torch.zeros_like(q, dtype=torch.bool).scatter(1, indices, included)
    return mask.detach()


def teacher_sets(clean, second):
    if clean.shape != second.shape:
        raise ValueError("Teacher views must describe the same physical packets/classes")
    a = prefix_sets(clean, RECIPE["cumulative_mass"])
    b = prefix_sets(second, RECIPE["cumulative_mass"])
    return (a | b).detach(), a, b


def class_set_mask(sets, classes=6):
    """Strict helper for explicit class-index sets, useful at the training boundary."""
    if classes != 6 or not isinstance(sets, (tuple, list)) or not sets:
        raise ValueError("Nonempty registered six-class index sets required")
    result = torch.zeros(len(sets), classes, dtype=torch.bool)
    for row, indices in enumerate(sets):
        if not isinstance(indices, (tuple, list)) or not indices:
            raise ValueError("Empty class sets are forbidden")
        if any(type(index) is not int or index < 0 or index >= classes for index in indices):
            raise ValueError("Class-set index out of range or not an integer")
        result[row, indices] = True
    return result


def set_loss(logits, class_mask):
    """Unreduced exact -log sum_{c in S} softmax(logits)_c."""
    if logits.ndim != 2 or logits.shape[1] != 6 or not len(logits) or not torch.isfinite(logits).all():
        raise ValueError("Finite nonempty six-class student logits required")
    if class_mask.dtype != torch.bool or class_mask.shape != logits.shape:
        raise ValueError("Class sets must be a boolean tensor matching logits")
    if not class_mask.any(1).all():
        raise ValueError("Empty class sets are forbidden")
    mask = class_mask.detach().to(logits.device)
    return torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~mask, -torch.inf), 1)


def matched_losses(logits, clean, second):
    """Same teacher views, packets, strong logits, participation, and denominator."""
    clean = _check_probabilities(clean).to(logits.device)
    second = _check_probabilities(second).to(logits.device)
    mask, a, b = teacher_sets(clean, second)
    target = ((clean + second)/2).detach()
    participate = (mask.sum(1) <= 3).detach()
    losses = dict(hard=F.cross_entropy(logits, target.argmax(1), reduction="none"),
                  soft=F.kl_div(F.log_softmax(logits, 1), target, reduction="none").sum(1),
                  set=set_loss(logits, mask))
    return dict(losses={k: v*participate for k, v in losses.items()},
                unmasked_losses=losses, teacher_target=target, class_mask=mask,
                view_masks=[a, b], participation=participate,
                mean_losses={k: (v*participate).mean() for k, v in losses.items()})


@torch.no_grad()
def _fixed_view_logits(model, cache, std, seed, device):
    generator = torch.Generator().manual_seed(seed)
    parts = []
    for start in range(0, len(cache["ids"]), d.BATCH_SIZE):
        x = cache["x"][start:start+d.BATCH_SIZE]
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype) * std
        logits = model((x + noise).to(device))
        if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
            raise ValueError("Source fixed-view teacher logits differ")
        parts.append(logits.detach().cpu())
    return torch.cat(parts)


def _rate(count, total):
    return count/total if total else None


def _summary(q, second, mask, a, b, selection, y=None):
    if not selection.any():
        return dict(count=0)
    size = mask.sum(1); small = size <= 3; ambiguous = (size >= 2) & small
    prediction = q.argmax(1); confidence = q.max(1).values
    count = int(selection.sum())
    selected = small & selection; ambiguous_selected = ambiguous & selection
    intersection = (a & b).sum(1).float(); union = (a | b).sum(1).float()
    result = dict(count=count, set_size_counts=torch.bincount(size[selection], minlength=7)[1:].tolist(),
                  mean_set_size=float(size[selection].float().mean()),
                  full_set_count=int(((size == 6) & selection).sum()),
                  full_set_fraction=float((size[selection] == 6).float().mean()),
                  small_set_count=int(selected.sum()), small_set_fraction=int(selected.sum())/count,
                  ambiguous_small_set_count=int(ambiguous_selected.sum()),
                  ambiguous_small_set_fraction=int(ambiguous_selected.sum())/count,
                  low_confidence_count=int(((confidence < .95) & selection).sum()),
                  ambiguous_low_confidence_count=int((ambiguous_selected & (confidence < .95)).sum()),
                  teacher_top1_agreement=float((prediction[selection] == second[selection].argmax(1)).float().mean()),
                  teacher_prefix_agreement=float((a[selection] == b[selection]).all(1).float().mean()),
                  mean_prefix_Jaccard=float((intersection[selection]/union[selection]).mean()))
    if y is None:
        result.update(truth_coverage=None, small_set_truth_coverage=None, top1_errors=None,
                      reason="U TX truth hidden; set size/stability are prediction-only quantities")
        return result
    if y.dtype != torch.long or y.shape != (len(q),) or y.min() < 0 or y.max() >= 6:
        raise ValueError("Legal labeled source TX values required after set construction")
    included = mask[torch.arange(len(y)), y]
    wrong = prediction != y
    wrong_ambiguous = wrong & ambiguous_selected
    result.update(truth_coverage=float(included[selection].float().mean()),
                  false_exclusion_count=int((~included & selection).sum()),
                  small_set_truth_coverage=_rate(int((included & selected).sum()), int(selected.sum())),
                  small_set_false_exclusion_count=int((~included & selected).sum()),
                  top1_errors=int((wrong & selection).sum()),
                  small_set_top1_errors=int((wrong & selected).sum()),
                  ambiguous_top1_errors=int(wrong_ambiguous.sum()),
                  ambiguous_top1_error_truth_inclusion=_rate(int((included & wrong_ambiguous).sum()), int(wrong_ambiguous.sum())),
                  top1_wrong_included_in_small_set=int((included & wrong & selected).sum()),
                  ambiguous_truth_coverage=_rate(int((included & ambiguous_selected).sum()), int(ambiguous_selected.sum())))
    return result


def _groups(cache, q, second, mask, a, b, y):
    def summarize(indices):
        return _summary(q, second, mask, a, b, indices, y)
    rx = cache["rx"]; day = cache["day"]
    result = dict(RX={str(r): summarize(rx == r) for r in rx.unique().tolist()},
                  day={str(v): summarize(day == v) for v in day.unique().tolist()},
                  RX_day={f"RX{r}_day{v}": summarize((rx == r) & (day == v))
                          for r in rx.unique().tolist() for v in day.unique().tolist()})
    if y is not None:
        result["TX"] = {str(tx): summarize(y == tx) for tx in range(6)}
        result["RX_day_TX"] = {f"RX{r}_day{v}_TX{tx}": summarize((rx == r) & (day == v) & (y == tx))
                               for r in rx.unique().tolist() for v in day.unique().tolist() for tx in range(6)}
    return result


def diagnose(ctx, output: Path, device):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); role_results = {}; audit = {}; forward_samples = 0
    for name, cache in [("L_s", ctx["L"]), ("V", ctx["V"]), ("U_s", ctx["U"])]:
        if name == "U_s":
            from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
            validate_unlabeled(cache)
        q = _check_probabilities(ctx["probabilities"][name]).cpu()
        second_logits = _fixed_view_logits(ctx["model"], cache, .005, ctx["seed"] + RECIPE["noise_seed_offsets"][name], device)
        second = second_logits.softmax(1).detach()
        # Construct every set without looking at the lawful L/V labels.
        mask, a, b = teacher_sets(q, second)
        strong_logits = _fixed_view_logits(ctx["model"], cache, .015, ctx["seed"] + RECIPE["noise_seed_offsets"][name]+100, device)
        losses = matched_losses(strong_logits, q, second)
        if mask.requires_grad or any(v.requires_grad for v in (q, second, losses["teacher_target"])):
            raise ValueError("Teacher probabilities and class sets must be detached")
        y = None if name == "U_s" else cache["y"]
        summary = _summary(q, second, mask, a, b, torch.ones(len(q), dtype=torch.bool), y)
        result = dict(summary=summary, groups=_groups(cache, q, second, mask, a, b, y),
                      mean_matched_losses={k: float(v) for k, v in losses["mean_losses"].items()},
                      teacher_forward_samples=len(q), strong_forward_samples=len(q),
                      loss_denominator=len(q), participating_samples=int(losses["participation"].sum()),
                      performance_gate=name == "L_s", U_truth_read=False)
        if name == "L_s":
            result["L_fold2"] = _summary(q, second, mask, a, b, ctx["folds"] == 2, y)
        role_results[name] = result
        audit[name] = dict(ids=cache["ids"], RX=cache["rx"], day=cache["day"],
                           clean_probabilities=q, second_teacher_logits=second_logits,
                           second_teacher_probabilities=second, strong_logits=strong_logits,
                           class_mask=mask, teacher_view_masks=[a, b],
                           matched_losses=losses["losses"], unmasked_losses=losses["unmasked_losses"],
                           teacher_target=losses["teacher_target"], participation=losses["participation"])
        if y is not None:
            audit[name]["legal_source_TX_after_set_construction"] = y
        forward_samples += 2*len(q)
    health = role_results["L_s"]["L_fold2"]
    passed = bool(health["ambiguous_small_set_count"] >= 20 and health["ambiguous_small_set_fraction"] >= .01
                  and health["top1_errors"] >= 10 and (health["small_set_truth_coverage"] or 0) >= .95
                  and (health["ambiguous_top1_error_truth_inclusion"] or 0) >= .5
                  and health["full_set_fraction"] <= .1)
    torch.save(dict(recipe=RECIPE, criteria=CRITERIA, roles=audit, L_folds=ctx["folds"]), output/"audit.pt")
    return dict(direction="sets", recipe=RECIPE, roles=role_results,
                applicability=dict(passed=passed, criteria=CRITERIA,
                                   reason="Legal held-source labels support nontrivial conservative small sets" if passed else "Held-source ambiguous-set count, coverage, or error-inclusion condition failed"),
                identity_updates=0, identity_backward_calls=0, auxiliary_parameters=0,
                deployment_parameters=0, target_access=False, U_truth_read=False,
                source_V_fit=False, extra_identity_forward_samples=forward_samples,
                elapsed_seconds=time.monotonic()-started, audit_artifact="audit.pt",
                caveat="Source truth coverage does not guarantee target coverage; set loss can change class proportions within the chosen set.")
