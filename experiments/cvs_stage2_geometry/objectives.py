"""Exact finite one-sided identity risks from report section5.2.

These functions contain no CE, coefficient, ramp, optimizer, or model forward.
The trainer owns clean CE and the preregistered external weights/schedule.
"""
import math

import torch

from experiments.cvs_stage2_decouple.common import margins


M_REQ = math.log(95.)


def _check_margin(value):
    if value <= 0 or not math.isfinite(float(value)):
        raise ValueError('Finite positive required margin expected')


def keep_per_sample(student_logits, parent_margin, labels, m_req=M_REQ):
    _check_margin(m_req)
    student = margins(student_logits, labels)
    parent = parent_margin.detach().to(student)
    if parent.shape != student.shape or not bool(torch.isfinite(parent).all()):
        raise ValueError('Parent raw-logit margin must be finite and match the batch')
    target = parent.clamp_max(m_req)
    violation = ((target - student) / m_req).clamp_min(0.)
    return .5 * violation.square() * (parent > 0).to(student.dtype)


def lt_per_sample(clean_logits, augmented_logits, labels, m_req=M_REQ):
    _check_margin(m_req)
    if clean_logits.shape != augmented_logits.shape:
        raise ValueError('Paired clean/action logits must have matching shapes')
    target = margins(clean_logits, labels).detach().clamp_max(m_req)
    augmented = margins(augmented_logits, labels)
    return .5 * ((target - augmented) / m_req).clamp_min(0.).square()


def keep_loss(student_logits, parent_margin, labels, m_req=M_REQ):
    return keep_per_sample(student_logits, parent_margin, labels, m_req).mean()


def lt_loss(clean_logits, augmented_logits, labels, m_req=M_REQ):
    return lt_per_sample(clean_logits, augmented_logits, labels, m_req).mean()


def margin_risk(clean_margin, augmented_margin, m_req=M_REQ):
    """Same LT risk for frozen preparation diagnostics with cached margins."""
    _check_margin(m_req)
    if clean_margin.shape != augmented_margin.shape:
        raise ValueError('Paired margin shapes differ')
    return .5 * ((clean_margin.detach().clamp_max(m_req) - augmented_margin) / m_req).clamp_min(0.).square()


def contract():
    return dict(required_margin=M_REQ, required_margin_expression='log(95)',
                classes=6, explanatory_softmax_epsilon=.05, calibrated_probability_guarantee=False,
                margin='raw true logit minus maximum competing logit',
                lt='.5 * relu((min(stopgrad(clean_margin),m_req)-aug_margin)/m_req)^2',
                keep='1[parent_margin>0] * .5 * relu((min(parent_margin,m_req)-student_margin)/m_req)^2',
                parent_reference_detached=True, clean_lt_reference_detached=True,
                reduction='mean over full batch, including zero/ineligible items',
                auxiliary_ce=False, weights_and_ramp='trainer responsibility')
