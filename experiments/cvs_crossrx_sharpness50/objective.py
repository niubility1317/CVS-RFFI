"""Feature-only source-L cross-RX sharpness transfer, with fixed constants.

No data reads, model/head construction, optimizer updates, second-order inner
gradient or parameter scan. Source/layer provenance is asserted by the caller.
"""
from contextlib import contextmanager
from collections.abc import Sequence
import sys

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches, physical_ids
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng, _rng_snapshot, _equal

P = 'id_backbone.encoder.core.id_backbone.'
Q = 'id_backbone.encoder.sequence_pool.'
SOURCE_RX = (1, 3, 4, 6, 8)
RHO = .05
COEFFICIENT = .1
EPS = 1e-12
MODES = ('ZERO', 'CLEAN_CE', 'GLOBAL_GAP', 'CROSS_GAP')


def select_names(model):
    """Exact primary sinc/time_fuse/t1/t2 plus all SIG_TIME except gain_raw."""
    if not isinstance(model, nn.Module):
        raise ValueError('An existing identity model is required')
    prefixes = tuple(P + group + '.' for group in ('sinc', 'time_fuse', 't1', 't2'))
    names = [name for name, parameter in model.named_parameters()
             if name.startswith(prefixes) or (name.startswith(Q) and name != Q + 'gain_raw')]
    if not names:
        raise ValueError('Declared primary identity/SIG_TIME feature scope is absent')
    return names


def _metadata(value, count, device, name):
    if (not isinstance(value, torch.Tensor) or value.shape != (count,)
            or value.dtype != torch.long or value.device != device):
        raise ValueError(f'{name} must be one visible long metadata item per packet on its device')
    return value.detach()


def _source(labels, receivers, identifiers, name):
    if not bool(((labels >= 0) & (labels <= 5)).all()) or set(receivers.tolist()) != set(SOURCE_RX):
        raise ValueError(f'{name} requires visible labels0..5 and all five source RX1/3/4/6/8')
    ids = physical_ids(identifiers, len(labels))
    if any(identity[1] != int(rx) or identity[2] not in (1, 2, 3)
           for identity, rx in zip(ids, receivers.tolist())):
        raise ValueError(f'{name} complete physical IDs disagree with source RX/day metadata')
    return ids


def _norm(values):
    result = torch.sqrt(sum(value.detach().double().square().sum() for value in values))
    if not bool(torch.isfinite(result)):
        raise ValueError('Selected feature gradient/perturbation norm is nonfinite')
    return result


def _delta(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, create_graph=False, allow_unused=True)
    for gradient in gradients:
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise ValueError('Current visible-L feature gradient is nonfinite')
    values = tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, gradients))
    norm = _norm(values)
    denominator = norm.clamp_min(EPS)
    delta = tuple((RHO * value / denominator).to(dtype=p.dtype).detach() for p, value in zip(parameters, values))
    actual = _norm(delta)
    return delta, float(norm), float(actual), sum(gradient is None for gradient in gradients)


@contextmanager
def _evaluation(model):
    # Backup is solely for preservation/readback, not a perturbation copy of
    # unselected parameters. Functional branch states detach those parameters.
    params = [(p, p.detach().clone(), p._version, p.grad,
               None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    buffers = [(b, b.detach().clone(), b._version) for b in model.buffers()]
    flags = [(module, module.training) for module in model.modules()]
    with _preserve_rng(), fresh_parameter_caches(model) as calls:
        try:
            model.eval()
            yield calls
        finally:
            active_exception = sys.exc_info()[0] is not None
            changed = False
            with torch.no_grad():
                for p, value, version, grad, grad_value in params:
                    changed |= p._version != version or p.grad is not grad
                    if not torch.equal(p.detach(), value):
                        changed = True;p.copy_(value)
                    if p.grad is not grad:p.grad = grad
                    if grad is not None and not torch.equal(grad, grad_value):
                        changed = True;grad.copy_(grad_value)
                for buffer, value, version in buffers:
                    changed |= buffer._version != version
                    if not torch.equal(buffer, value):
                        changed = True;buffer.copy_(value)
            for module, flag in flags:module.training = flag
            if changed and not active_exception:
                raise RuntimeError('Temporary feature evaluation altered live tensors/gradients')


def objective(model, current_logits, current_labels, current_rx, held_x, held_labels, held_rx,
              current_ids, held_ids, selected_names, default_weight_name, mode, *, role='L_s'):
    """Fixed first-order feature transfer on current and disjoint held source L.

    Current logits retain their original main graph. All five RX must be
    present; a future training caller may preregister skipping an entire missing-
    RX call, but this function never changes the five-group denominator.
    Each current RX CE mean supplies detached delta=.05*g/max(||g||,1e-12),
    treating unused gradients as zero. GLOBAL_GAP uses the full current CE
    gradient for all five deltas. For equal inner diagnostic cost every mode
    computes the five local gradients and that one global gradient.

    Evaluate held clean once, and perturb each group's OTHER-RX packets once.
    Thus every mode performs six forwards and 5*heldN IQ observations. CROSS/
    GLOBAL loss=.1/5*sum ReLU(perturbedCE(otherRX)-cleanCE(otherRX)); clean
    remains attached. CLEAN_CE uses .1/5*sum clean means on the same masks.
    ZERO is selected-connected clean CE*0, never current_logits*0.

    Functional states keep only selected parameters attached (p+detached delta
    for perturbations), detach every other parameter including current preW and
    all gains, and clone buffers independently per call. No new head/parameter
    is created. Main head/backbone joint training belongs to the caller.
    """
    if role != 'L_s' or mode not in MODES:
        raise ValueError('Only fixed modes and visible source L_s are permitted')
    if (not isinstance(current_logits, torch.Tensor) or current_logits.ndim != 2
            or current_logits.shape[1] != 6 or len(current_logits) < 5
            or not current_logits.is_floating_point() or not current_logits.requires_grad
            or not bool(torch.isfinite(current_logits).all())):
        raise ValueError('Current attached finite source logits[N,6] required')
    if (not isinstance(held_x, torch.Tensor) or held_x.layout != torch.strided
            or held_x.ndim != 3 or held_x.shape[1:] != (2, 256) or len(held_x) < 5
            or not held_x.is_floating_point() or not bool(torch.isfinite(held_x).all())
            or held_x.device != current_logits.device):
        raise ValueError('Held finite real source IQ[M,2,256] on the current device required')
    current_labels = _metadata(current_labels, len(current_logits), current_logits.device, 'current_labels')
    current_rx = _metadata(current_rx, len(current_logits), current_logits.device, 'current_rx')
    held_labels = _metadata(held_labels, len(held_x), held_x.device, 'held_labels')
    held_rx = _metadata(held_rx, len(held_x), held_x.device, 'held_rx')
    current = _source(current_labels, current_rx, current_ids, 'current L')
    held = _source(held_labels, held_rx, held_ids, 'held L')
    if {identity[-1] for identity in current} & {identity[-1] for identity in held}:
        raise ValueError('Held L contains a physical packet from the complete current batch')
    names = select_names(model)
    if (not isinstance(selected_names, Sequence) or isinstance(selected_names, (str, bytes))
            or list(selected_names) != names):
        raise ValueError('selected_names must equal the exact fixed primary/SIG_TIME scope and order')
    named = dict(model.named_parameters())
    if default_weight_name not in named or named[default_weight_name].shape != (6, 160) or default_weight_name in names:
        raise ValueError('The unchanged native default W6x160 must be outside the feature scope')
    parameters = [named[name] for name in names]
    if (not all(p.requires_grad and p.is_floating_point() and p.device == held_x.device
                and bool(torch.isfinite(p).all()) for p in parameters)
            or not bool(torch.isfinite(named[default_weight_name]).all())):
        raise ValueError('Selected live feature parameters must be finite, trainable and on the source device')
    rng_before = _rng_snapshot()
    cross_delta, inner = {}, {}
    with torch.enable_grad():
        # These autograd calls never accumulate into live p.grad and retain the
        # original current L graph for the normal main loss backward.
        for receiver in SOURCE_RX:
            mask = current_rx == receiver
            ce = F.cross_entropy(current_logits[mask], current_labels[mask])
            delta, norm, actual, missing = _delta(ce, parameters)
            cross_delta[str(receiver)] = delta
            inner[str(receiver)] = dict(inner_CE=float(ce.detach()), inner_gradient_norm=norm,
                                        actual_perturb_norm=actual, inner_none_gradient_count=missing)
        global_ce = F.cross_entropy(current_logits, current_labels)
        global_delta, global_norm, global_actual, global_none = _delta(global_ce, parameters)
        selected_delta = ({str(receiver): global_delta for receiver in SOURCE_RX}
                          if mode == 'GLOBAL_GAP' else cross_delta)
        detached = {name: parameter.detach() for name, parameter in named.items()}
        base_buffers = {name: value.detach() for name, value in model.named_buffers()}
        fixed_w = named[default_weight_name].detach()
        def forward(x, delta=None):
            state = dict(detached)
            for i, name in enumerate(names):
                state[name] = named[name] if delta is None else named[name] + delta[i]
            state[default_weight_name] = fixed_w
            result = torch.func.functional_call(model,
                (state, {name: value.clone() for name, value in base_buffers.items()}),
                (x.detach(),), strict=True)
            if not isinstance(result, torch.Tensor) or result.shape != (len(x), 6) or not bool(torch.isfinite(result).all()):
                raise ValueError('Native default head functional logits must be finite [heldN,6]')
            return result
        by_rx, perturbed_ce, gaps, clean_means = {}, {}, [], []
        with _evaluation(model) as calls:
            clean_ce = F.cross_entropy(forward(held_x), held_labels, reduction='none')
            for receiver in SOURCE_RX:
                key = str(receiver);mask = held_rx != receiver
                pert_ce = F.cross_entropy(forward(held_x[mask], selected_delta[key]), held_labels[mask], reduction='none')
                clean_mean, pert_mean = clean_ce[mask].mean(), pert_ce.mean()
                gap = pert_mean - clean_mean
                gaps.append(gap);clean_means.append(clean_mean);perturbed_ce[key] = pert_ce
                facts = dict(inner[key])
                if mode == 'GLOBAL_GAP':
                    facts.update(inner_CE=float(global_ce.detach()), inner_gradient_norm=global_norm,
                                 actual_perturb_norm=global_actual, inner_none_gradient_count=global_none)
                by_rx[key] = dict(facts, current_count=int((current_rx == receiver).sum()),
                                  held_other_count=int(mask.sum()), clean_CE=float(clean_mean.detach()),
                                  pert_CE=float(pert_mean.detach()), gap=float(gap.detach()),
                                  positive_gap=float(F.relu(gap).detach()))
            gap_loss = COEFFICIENT / 5 * torch.stack(gaps).relu().sum()
            clean_loss = COEFFICIENT / 5 * torch.stack(clean_means).sum()
            weighted_loss = (clean_ce * 0.).sum() if mode == 'ZERO' else clean_loss if mode == 'CLEAN_CE' else gap_loss
            if calls[0] != 6 or not bool(torch.isfinite(weighted_loss)):
                raise RuntimeError('Fixed six-forward feature-transfer budget/finite loss differs')
    if not _equal(rng_before, _rng_snapshot()):
        raise RuntimeError('Feature-transfer evaluation did not restore initialized RNG states')
    summary = dict(mode=mode, role=role, rho=RHO, coefficient=COEFFICIENT, eps=EPS,
        selected_count=len(names), selected_names=names, current_count=len(current), held_count=len(held),
        current_RX_counts={str(rx):int((current_rx == rx).sum()) for rx in SOURCE_RX},
        held_RX_counts={str(rx):int((held_rx == rx).sum()) for rx in SOURCE_RX},
        by_RX=by_rx, loss=float(weighted_loss.detach()), raw_gap_loss=float(gap_loss.detach()),
        clean_CE_loss=float(clean_loss.detach()), global_inner_gradient_norm=global_norm,
        inner_gradient_calls=6, functional_forwards=6, extra_IQ_forwards=5*len(held),
        perturbation_rule='rho*g.detach()/max(global_selected_L2_norm,1e-12); None gradients zero',
        clean_gradient_stopped=False, perturbation_second_order=False,
        outer_graph='Only selected live parameters; all other parameters including preW/gains detached',
        default_weight_name=default_weight_name, auxiliary_W_gain_gradients_None=True,
        main_graph_retained=True, live_parameters_buffers_gradients_unchanged=True,
        evaluation_flags_RNG_private_caches_restored=True, buffers_cloned_per_forward=True,
        full_current_physical_batch_excluded=True, previously_trained_source_L=True,
        never_trained_holdout=False, optimizer_updates=0, head_fit_updates=0,
        new_parameters=0, extra_head=False, target_access=False, U_truth_read=False,
        boundary='Caller asserts true visible source/layer provenance; fixed source transfer prototype, not efficacy or DG evidence')
    return dict(loss=weighted_loss, weighted_loss=weighted_loss, summary=summary,
                tensors=dict(clean_CE=clean_ce, perturbed_CE_by_RX=perturbed_ce,
                             deltas_by_RX=selected_delta))
