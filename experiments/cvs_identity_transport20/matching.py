"""Stopped visible-label matching of two six-class actual feature cohorts.

No head, prototype, model, optimizer or persistent state is constructed. Input
labels/provenance are caller assertions; this primitive cannot authorize data.
"""
import math

import torch
from torch.nn import functional as F

CLASSES = 6
FEATURE_DIM = 160
TEMPERATURE = .1
PROBABILITY = .9
WEIGHT = .1
ITERATIONS = 80
RMS_FLOOR = 1e-8
APPROXIMATE_RESIDUAL = 1e-5


def _features(a, b):
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or a.shape != (CLASSES, FEATURE_DIM) or b.shape != a.shape
            or a.dtype not in (torch.float32, torch.float64) or b.dtype != a.dtype
            or a.device != b.device or a.device.type == 'meta'
            or a.layout != torch.strided or b.layout != torch.strided):
        raise ValueError('a/b must be same-device/dtype dense FP32 or FP64 tensors[6,160]')
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError('Actual cohort features must be finite')


def _labels(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type == 'meta' or value.shape != (CLASSES,) or value.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(name + ' must be integer labels[6]')
        value = value.detach().cpu().tolist()
    if (not isinstance(value, (list, tuple)) or len(value) != CLASSES
            or any(type(label) is not int for label in value) or set(value) != set(range(CLASSES))):
        raise ValueError(name + ' must contain every visible class0..5 exactly once')
    return list(value)


def _relative_cost(a, b):
    """Return graph-attached FP64(C, mean(B²), rawD); no RMS detach.

    Independent common shifts on a/b add only row/column terms to rawD,
    removed by double-centering. A common positive feature scale multiplies B
    and its RMS by the same factor squared, so C is invariant while the floor
    is inactive. The absolute floor explicitly limits that scale claim.
    """
    _features(a, b)
    raw = (a.double()[:, None, :] - b.double()[None, :, :]).square().mean(dim=2)
    centered = raw - raw.mean(dim=1, keepdim=True) - raw.mean(dim=0, keepdim=True) + raw.mean()
    square_rms = centered.square().mean()
    # clamp before sqrt: at exact collapse sqrt(0)'s infinite derivative must
    # not be multiplied by a downstream zero derivative and produce NaN.
    denominator = square_rms.clamp_min(RMS_FLOOR ** 2).sqrt()
    relative = centered / denominator
    if any(not bool(torch.isfinite(value).all()) for value in (raw, centered, square_rms, relative)):
        raise ValueError('FP64 centered-distance calculation overflowed')
    return relative, square_rms, raw


def match(a, b, mode, labels_a, labels_b):
    """Return attached stopped loss and compact JSON diagnostics.

    ROW uses independent row/column log-softmax of the same -C/.1. BI uses
    exactly80 log-domain row/column scaling iterations with uniform1/6
    marginals. Its finite plan is explicitly approximate: residual>1e-5 is
    disclosed, never repaired by extra iterations or treated as a hidden gate.
    Correct BI masses are conditional row/column probabilities, not unchecked
    6Pdiag. The latter is separately recorded and equals conditional mass only
    in the exact uniform-marginal limit.

    Loss=.1*(mean relu(log(.9)-logqrow)+mean relu(log(.9)-logqcol))/2.
    Full six-row denominators include stopped terms. Zero/collapsed features
    have correct mass1/6, finite positive loss and zero feature derivative;
    this objective alone cannot escape exact feature collapse. Outputs retain
    gradients through BOTH feature endpoints, including the cost RMS. Internal
    FP64 calculation returns loss in the original FP32/FP64 input dtype.
    """
    if mode not in ('ROW', 'BI'):
        raise ValueError('Matching mode must be ROW or BI')
    la, lb = _labels(labels_a, 'labels_a'), _labels(labels_b, 'labels_b')
    relative, square_rms, raw = _relative_cost(a, b)
    logits = -relative / TEMPERATURE
    raw_balanced = None
    row_residual = column_residual = residual_max = None
    row_control_column = column_control_row = None
    if mode == 'BI':
        u = torch.zeros(CLASSES, device=a.device, dtype=torch.float64)
        v = torch.zeros_like(u)
        log_marginal = -math.log(CLASSES)
        for _ in range(ITERATIONS):
            u = log_marginal - torch.logsumexp(logits + v[None, :], dim=1)
            v = log_marginal - torch.logsumexp(logits + u[:, None], dim=0)
        log_plan = logits + u[:, None] + v[None, :]
        log_row = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
        log_column = log_plan - torch.logsumexp(log_plan, dim=0, keepdim=True)
        plan = log_plan.exp()
        row_residual = float((plan.sum(dim=1) - 1. / CLASSES).abs().max().detach())
        column_residual = float((plan.sum(dim=0) - 1. / CLASSES).abs().max().detach())
        residual_max = max(row_residual, column_residual)
    else:
        log_row = F.log_softmax(logits, dim=1)
        log_column = F.log_softmax(logits, dim=0)
        row_control_column = float((log_row.exp().sum(dim=0) / CLASSES - 1. / CLASSES).abs().max().detach())
        column_control_row = float((log_column.exp().sum(dim=1) / CLASSES - 1. / CLASSES).abs().max().detach())
    row_index = torch.arange(CLASSES, device=a.device)
    row_target = torch.tensor([lb.index(label) for label in la], device=a.device)
    column_target = torch.tensor([la.index(label) for label in lb], device=a.device)
    correct_row = log_row[row_index, row_target]
    correct_column = log_column[column_target, row_index]
    log_stop = math.log(PROBABILITY)
    hrow = F.relu(log_stop - correct_row)
    hcolumn = F.relu(log_stop - correct_column)
    loss_double = WEIGHT * (hrow.mean() + hcolumn.mean()) / 2.
    loss = loss_double.to(dtype=a.dtype)
    if mode == 'BI':
        raw_balanced = (CLASSES * plan[row_index, row_target]).detach()
    outputs = (correct_row, correct_column, hrow, hcolumn, loss_double, loss)
    if any(not bool(torch.isfinite(value).all()) for value in outputs):
        raise ValueError('Log-domain matching probabilities/loss became nonfinite')
    raw_norm = torch.linalg.vector_norm(raw).detach()
    if not bool(torch.isfinite(raw_norm)):
        raise ValueError('FP64 raw-distance summary norm overflowed')
    summary = dict(mode=mode, loss=float(loss.detach()), iterations=ITERATIONS if mode == 'BI' else 0,
                   diagmass={'row': correct_row.detach().exp().tolist(),
                             'column': correct_column.detach().exp().tolist()},
                   lowmass={'row': int((correct_row < log_stop).sum().detach()),
                            'column': int((correct_column < log_stop).sum().detach())},
                   argmaxerrors={'row': int((log_row.argmax(dim=1) != row_target).sum().detach()),
                                 'column': int((log_column.argmax(dim=0) != column_target).sum().detach())},
                   balanced_plan=mode == 'BI',
                   row_marginal_residual=row_residual, column_marginal_residual=column_residual,
                   marginal_residual_max=residual_max,
                   approximate_plan=residual_max > APPROXIMATE_RESIDUAL if mode == 'BI' else None,
                   row_control_column_marginal_residual=row_control_column,
                   column_control_row_marginal_residual=column_control_row,
                   raw_balanced_diagmass=raw_balanced.tolist() if raw_balanced is not None else None,
                   raw_balanced_mass_above_one_count=int((raw_balanced > 1.).sum()) if raw_balanced is not None else None,
                   costRMS=math.sqrt(float(square_rms.detach())),
                   rawDnorm=float(raw_norm),
                   relative_cost_RMS=float(relative.detach().square().mean().sqrt()),
                   floor_active=bool(square_rms.detach() <= RMS_FLOOR ** 2),
                   temperature=TEMPERATURE, probability=PROBABILITY, weight=WEIGHT, RMS_floor=RMS_FLOOR,
                   marginal_residual_reporting_threshold=APPROXIMATE_RESIDUAL,
                   internal_dtype='float64', loss_dtype=str(a.dtype),
                   cost_formula='double_center(mean_feature_squared_distance)/differentiable_RMS',
                   mass_formula='conditional row/column correct-class probabilities',
                   scale_invariance_requires_inactive_RMS_floor=True,
                   finite_iterations_are_not_exact_OT=True, no_new_head_parameters_or_state=True,
                   caller_establishes_visible_L_provenance=True)
    return loss, summary
