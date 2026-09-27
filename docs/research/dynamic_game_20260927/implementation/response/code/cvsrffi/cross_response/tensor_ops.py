"""Equal-weight complete-grid algebra; zero interaction does not imply disentanglement."""
import torch
import math


def _grid(x):
    if x.ndim != 3 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError('expected finite complete [TX,RX,coordinate] grid')
    return x


def double_center(x):
    x = _grid(x)
    return x - x.mean(1, keepdim=True) - x.mean(0, keepdim=True) + x.mean((0, 1), keepdim=True)


def grid_decomposition(x):
    x = _grid(x)
    grand = x.mean((0, 1), keepdim=True)
    return dict(grand=grand, tx=x.mean(1, keepdim=True)-grand,
                rx=x.mean(0, keepdim=True)-grand, interaction=double_center(x))


def arithmetic_completion(x):
    """Leave-row/column-corner completion, algebraically the cross residual."""
    x = _grid(x)
    p, q, _ = x.shape
    if min(p, q) < 2:
        raise ValueError('completion requires at least 2x2')
    row, col, total = x.sum(1, keepdim=True), x.sum(0, keepdim=True), x.sum((0, 1), keepdim=True)
    return (row-x)/(q-1)+(col-x)/(p-1)-(total-row-col+x)/((p-1)*(q-1))


def identity_interaction_loss(grid):
    """Separate Ux control: mean cell squared Euclidean interaction norm."""
    if min(grid.shape[:2]) < 2:
        raise ValueError('identity interaction requires at least 2x2')
    return double_center(grid).square().sum(-1).mean()


def response_loss(prediction, target, interaction_weight=0.0):
    if prediction.shape != target.shape or interaction_weight < 0:
        raise ValueError('matched grids and nonnegative interaction weight required')
    _grid(prediction)
    target = _grid(target).detach()
    mse = (prediction-target).square().mean()
    available = min(prediction.shape[:2]) >= 2
    if interaction_weight and not available:
        raise ValueError('interaction weighting needs independent query rectangle')
    true, pred = grid_decomposition(target), grid_decomposition(prediction)
    interaction_error = (true['interaction']-pred['interaction']).square().mean() if available else None
    loss = mse + interaction_weight * interaction_error if interaction_weight else mse
    return loss, dict(total_error=mse.detach(), interaction_available=available,
                      interaction_error=None if interaction_error is None else interaction_error.detach(),
                      true_interaction=true['interaction'].detach() if available else None,
                      predicted_interaction=pred['interaction'].detach() if available else None,
                      tx_main_effect_error=(true['tx']-pred['tx']).square().mean().detach(),
                      rx_main_effect_error=(true['rx']-pred['rx']).square().mean().detach())


def unit_identity_geometry(features, *, min_norm, labels=None, logits=None):
    """Audit received-record geometry, normalizing each record before K means.

    Incomplete/near-zero grids have no normalized interaction estimate. Invalid
    records are never replaced by epsilon-generated unit vectors or zero losses.
    All returned diagnostics are detached; use the separate candidate loss below
    for differentiable unit-record interaction.
    """
    if features.ndim != 4 or min(features.shape) < 1 or not torch.isfinite(features).all():
        raise ValueError('expected finite [TX,RX,K,feature] records')
    if not math.isfinite(float(min_norm)) or min_norm <= 0:
        raise ValueError('min_norm must be explicitly positive')
    z = features.detach()
    if z.dtype in (torch.float16, torch.bfloat16):
        z = z.float()
    norms = z.norm(dim=-1)
    valid = norms > min_norm
    raw_grid = z.mean(2)
    raw_interaction = double_center(raw_grid).square().sum(-1).mean()
    raw_energy = raw_grid.square().sum(-1).mean()
    available = bool(valid.all() and min(z.shape[:2]) >= 2)
    result = dict(available=available, valid_record_mask=valid,
                  invalid_records=int((~valid).sum()), min_norm=float(min_norm),
                  raw_norm_mean=float(norms.mean()), raw_norm_p50=float(norms.quantile(.5)),
                  raw_norm_p95=float(norms.quantile(.95)), raw_interaction=float(raw_interaction),
                  raw_interaction_per_160_dimensions=float(raw_interaction * 160 / z.shape[-1]),
                  raw_relative_interaction=(float(raw_interaction/raw_energy)
                                            if float(raw_energy) > min_norm**2 else None),
                  normalized_grid=None, unit_interaction=None,
                  unavailable_reason=None if available else 'invalid_record_norm_or_incomplete_rectangle')
    if available:
        normalized_grid = (z / norms.unsqueeze(-1)).mean(2)
        result.update(normalized_grid=normalized_grid,
                      unit_interaction=float(identity_interaction_loss(normalized_grid)))
    if labels is None and logits is not None:
        raise ValueError('logits diagnostics require true legal source labels')
    if labels is not None:
        flat = z.reshape(-1, z.shape[-1])
        labels = torch.as_tensor(labels, device=z.device).reshape(-1)
        if len(labels) != len(flat) or labels.dtype.is_floating_point or (labels < 0).any():
            raise ValueError('one nonnegative integer source label required per record')
        flat_valid = valid.reshape(-1)
        unit = flat[flat_valid] / norms.reshape(-1)[flat_valid, None]
        valid_labels = labels[flat_valid]
        prototypes, intra, invalid_classes = {}, {}, []
        for label in labels.unique().tolist():
            selected = unit[valid_labels == label]
            if not len(selected):
                invalid_classes.append(int(label))
                continue
            center = selected.mean(0)
            norm = center.norm()
            if norm <= min_norm:
                invalid_classes.append(int(label))
                continue
            center = center / norm
            prototypes[int(label)] = center
            angles = torch.rad2deg(torch.acos((selected @ center).clamp(-1, 1)))
            intra[int(label)] = dict(count=len(selected), angle_p95_deg=float(angles.quantile(.95)),
                                     angle_max_deg=float(angles.max()))
        classes = sorted(prototypes)
        inter = {(a, b): float(torch.rad2deg(torch.acos(torch.dot(prototypes[a], prototypes[b]).clamp(-1, 1))))
                 for i, a in enumerate(classes) for b in classes[i+1:]}
        result.update(inter_class_angles_deg=inter, intra_class_tail=intra,
                      unavailable_class_prototypes=invalid_classes)
        if logits is not None:
            logits = torch.as_tensor(logits, device=z.device).detach()
            if logits.numel() == 0 or logits.numel() % len(flat):
                raise ValueError('logits must align with source records')
            logits = logits.reshape(len(flat), -1)
            if not torch.isfinite(logits).all() or labels.max() >= logits.shape[-1]:
                raise ValueError('invalid source logits/class axis')
            predicted = logits.argmax(-1)
            pairs = {}
            for truth, prediction in zip(labels.tolist(), predicted.tolist()):
                if truth != prediction:
                    key = (int(truth), int(prediction))
                    pairs[key] = pairs.get(key, 0) + 1
            result['error_class_pairs'] = pairs
            result['classification_records'] = len(labels)
    return result


def normalized_identity_interaction_loss(features, *, min_norm):
    """Separate optional candidate; never changes the existing raw Ux objective."""
    diagnostics = unit_identity_geometry(features, min_norm=min_norm)
    if not diagnostics['available']:
        return None, diagnostics
    work = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
    unit_records = work / work.norm(dim=-1, keepdim=True)
    return identity_interaction_loss(unit_records.mean(2)), diagnostics


def decomposed_response_losses(prediction, target, *, alpha, beta,
                               interaction_reliability, weight_bounds):
    """Return three route-specific scalars, not a summed training objective.

    Caller differentiates predictor_full only for predictor/readout parameters,
    identity only for identity parameters and domain only for domain parameters.
    Shared parameters are excluded by the caller's explicit parameter routing.
    """
    if prediction.shape != target.shape or min(prediction.shape[:2]) < 2:
        raise ValueError('matched complete independent query grids required')
    _grid(prediction)
    target = _grid(target).detach()
    if prediction.dtype in (torch.float16, torch.bfloat16):
        prediction = prediction.float()
        target = target.float()
    if any(not math.isfinite(float(v)) or v < 0 for v in (alpha, beta)):
        raise ValueError('alpha/beta must be explicit finite nonnegative coefficients')
    if len(weight_bounds) != 2:
        raise ValueError('explicit reliability lower and upper bounds required')
    lower, upper = map(float, weight_bounds)
    if not all(math.isfinite(v) for v in (lower, upper)) or not 0 <= lower <= upper <= 1:
        raise ValueError('reliability bounds must lie in [0,1]')
    reliability = torch.as_tensor(interaction_reliability, dtype=prediction.dtype,
                                  device=prediction.device).detach()
    if reliability.numel() != 1 or not torch.isfinite(reliability).all():
        raise ValueError('one finite source-derived interaction reliability required')
    weight = reliability.reshape(()).clamp(lower, upper)
    error = prediction - target
    parts = grid_decomposition(error)
    energies = {name: value.square().mean() for name, value in parts.items()}
    losses = dict(predictor_full=error.square().mean(),
                  identity=energies['tx'] + float(alpha)*weight*energies['interaction'],
                  domain=energies['rx'] + float(beta)*weight*energies['interaction'])
    reconstruction = sum(parts.values())
    diagnostics = dict(error_components={name: value.detach() for name, value in parts.items()},
                       component_energies={name: value.detach() for name, value in energies.items()},
                       reconstruction_max_error=(reconstruction-error).abs().max().detach(),
                       energy_identity_error=(sum(energies.values())-losses['predictor_full']).abs().detach(),
                       interaction_reliability_raw=reliability.reshape(()),
                       interaction_reliability=weight, weight_bounds=(lower, upper))
    return losses, diagnostics
