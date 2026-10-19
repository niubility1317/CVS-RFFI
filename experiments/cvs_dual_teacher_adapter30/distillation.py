"""Two separate same-IQ teacher KL terms and all-L smoothed true-label CE.

All scale arguments are positive multipliers on native logits. Temperatures
are explicit; neither teacher is selected by student confidence/correctness.
"""
import math
import torch
from torch.nn import functional as F


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(name+' must be an explicit finite positive scalar')
    return float(value)


def _weight(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(name+' must be finite and nonnegative')
    return float(value)


def _inputs(student, teacher, labels):
    if (not torch.is_tensor(student) or not torch.is_tensor(teacher)
            or not torch.is_tensor(labels) or labels.dtype != torch.long or labels.ndim != 1
            or not len(labels) or student.shape != (len(labels), 6) or teacher.shape != student.shape
            or not student.is_floating_point() or not teacher.is_floating_point()
            or student.dtype != teacher.dtype or student.device != teacher.device
            or labels.device != student.device or not bool(((labels >= 0) & (labels < 6)).all())
            or not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all())):
        raise ValueError('Aligned finite native six-class logits and visible true L labels required')


def masked_teacher_kl(student, teacher, labels, *, temperature, teacher_scale,
                      student_scale, teacher_correct_mask=True, temperature_compensation=True):
    """Mean_all_N[mask * KL(teacher || student)] with optional T-squared factor."""
    _inputs(student, teacher, labels)
    temperature = _positive(temperature, 'temperature')
    teacher_scale = _positive(teacher_scale, 'teacher_scale multiplier')
    student_scale = _positive(student_scale, 'student_scale multiplier')
    if type(teacher_correct_mask) is not bool or type(temperature_compensation) is not bool:
        raise ValueError('Correct-teacher masking and T-squared policy must be explicit booleans')
    target = teacher.detach()
    teacher_correct = target.argmax(1) == labels
    accepted = teacher_correct if teacher_correct_mask else torch.ones_like(teacher_correct)
    log_target = F.log_softmax(target * teacher_scale / temperature, dim=1)
    log_student = F.log_softmax(student * student_scale / temperature, dim=1)
    per_packet = (log_target.exp() * (log_target - log_student)).sum(1)
    factor = temperature**2 if temperature_compensation else 1.
    value = (per_packet * accepted.to(per_packet.dtype)).sum() * factor / len(labels)
    return value, dict(KL=float(value.detach()), teacher_correct=int(teacher_correct.sum()),
        accepted=int(accepted.sum()), denominator=len(labels), temperature=temperature,
        teacher_scale_multiplier=teacher_scale, student_scale_multiplier=student_scale,
        temperature_compensation=temperature_compensation, temperature_factor=factor,
        teacher_correct_mask=teacher_correct_mask, direction='teacher||student',
        teacher_stop_gradient=True, student_confidence_gate=False, student_correctness_gate=False)


def objective(student_clean, student_As, labels, teacher_clean, teacher_As, *,
              clean_temperature, satellite_temperature,
              clean_teacher_scale, clean_student_scale,
              satellite_teacher_scale, satellite_student_scale,
              clean_KD_weight, satellite_KD_weight,
              clean_CE_weight=.5, satellite_CE_weight=.5,
              teacher_correct_mask=True, temperature_compensation=True):
    """0.5 CE(clean)+0.5 CE(As)+w_clean KL(Tc||S)+w_As KL(Ts||S).

    Tc and clean student logits must originate from the identical clean IQ;
    Ts and As student logits must originate from the identical full As IQ.
    Callers preserve those two pairings; this tensor function never mixes them.
    """
    _inputs(student_clean, teacher_clean, labels)
    wc = _weight(clean_KD_weight, 'clean_KD_weight')
    ws = _weight(satellite_KD_weight, 'satellite_KD_weight')
    cc = _weight(clean_CE_weight, 'clean_CE_weight')
    cs = _weight(satellite_CE_weight, 'satellite_CE_weight')
    if cc + cs != 1.:
        raise ValueError('True-label CE weights must sum to one')
    if student_As is None and (cs != 0. or ws != 0.):
        raise ValueError('A satellite student forward is required for nonzero satellite CE or KD')
    if student_As is not None:
        _inputs(student_As, teacher_As, labels)
        if student_clean.dtype != student_As.dtype or student_clean.device != student_As.device:
            raise ValueError('Both student slots must use the same native numerical policy')
    else:
        # Both teachers still run on their own views as a measured shadow.
        _inputs(student_clean, teacher_As, labels)
    clean_CE = F.cross_entropy(student_clean, labels, label_smoothing=.01)
    As_CE = student_clean.sum() * 0. if student_As is None else F.cross_entropy(student_As, labels, label_smoothing=.01)
    clean_KL, clean = masked_teacher_kl(student_clean, teacher_clean, labels,
        temperature=clean_temperature, teacher_scale=clean_teacher_scale,
        student_scale=clean_student_scale, teacher_correct_mask=teacher_correct_mask,
        temperature_compensation=temperature_compensation)
    if student_As is not None:
        satellite_KL, satellite = masked_teacher_kl(student_As, teacher_As, labels,
            temperature=satellite_temperature, teacher_scale=satellite_teacher_scale,
            student_scale=satellite_student_scale, teacher_correct_mask=teacher_correct_mask,
            temperature_compensation=temperature_compensation)
        satellite['student_forward'] = True
    else:
        satellite_KL = student_clean.sum() * 0.
        satellite = dict(KL=0., teacher_correct=int((teacher_As.detach().argmax(1)==labels).sum()),
            accepted=0, denominator=len(labels), student_forward=False, shadow_only=True,
            temperature=_positive(satellite_temperature, 'satellite_temperature'),
            teacher_scale_multiplier=_positive(satellite_teacher_scale, 'satellite_teacher_scale'),
            student_scale_multiplier=_positive(satellite_student_scale, 'satellite_student_scale'))
    total = cc * clean_CE + cs * As_CE + wc * clean_KL + ws * satellite_KL
    return total, dict(clean_CE=float(clean_CE.detach()), As_CE=float(As_CE.detach()),
        clean_KL=float(clean_KL.detach()), satellite_KL=float(satellite_KL.detach()),
        total_L=float(total.detach()), clean_CE_weight=cc, satellite_CE_weight=cs,
        clean_KD_weight=wc, satellite_KD_weight=ws, CE_smoothing=.01,
        CE_temperature=1., denominator=len(labels), clean=clean, satellite=satellite,
        mixed_teacher_target=False, raw_logit_MSE=False, feature_keep_loss=False)
