"""Finite input/output changes, with no autograd, F, labels, or identity claims."""
import math
import torch
from . import design as d


def _iq(value):
    if (not isinstance(value, torch.Tensor) or value.ndim != 3 or not len(value)
            or value.shape[1:] != (2, 256) or value.device.type != 'cpu'
            or value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all()):
        raise ValueError('Nonempty finite CPU float IQ[B,2,256] required')
    return value


def power(value):
    value = _iq(value).to(torch.float64)
    return value.square().sum(1).mean(1)


def perturbations(value, seed, epsilons=d.EPSILONS):
    _iq(value)
    if (len(epsilons) != 2 or any(type(e) not in (int,float) or not math.isfinite(e) or e <= 0 for e in epsilons)
            or epsilons[0] >= epsilons[1] or type(seed) is not int or seed < 0):
        raise ValueError('Two increasing finite positive steps and deterministic integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    random = torch.randn(value.shape, generator=generator, dtype=value.dtype)
    random = random / power(random).sqrt().to(value.dtype)[:,None,None]
    random = random * power(value).sqrt().to(value.dtype)[:,None,None]
    phase = torch.stack((-value[:,1], value[:,0]), 1)
    return {name:{e:direction*e for e in epsilons} for name,direction in (('random',random),('phase',phase))}


def response_fields(plus, minus, center):
    _iq(plus); _iq(minus); _iq(center)
    if plus.shape != minus.shape or plus.shape != center.shape:
        raise ValueError('Matched plus/minus/center outputs required')
    plus, minus, center = (v.to(torch.float64) for v in (plus,minus,center))
    return (plus-minus)*.5, (plus+minus)*.5-center


def _finite(values):
    if any(v.ndim != 1 or not torch.isfinite(v).all() or (v < 0).any() for v in values.values()):
        raise FloatingPointError('Nonnegative finite per-example metrics required')
    return values


def center_metrics(value, clean, teacher):
    if value.shape != clean.shape or value.shape != teacher.shape:
        raise ValueError('Matched center, clean, and reference outputs required')
    return _finite(dict(paired_clean_nmse=power(value-clean)/power(clean).clamp_min(d.ABSOLUTE_FLOOR),
        teacher_output_nmse=power(value-teacher)/power(teacher).clamp_min(d.ABSOLUTE_FLOOR)))


def response_metrics(student_odd, student_even, teacher_odd, teacher_even, delta, clean):
    values = (student_odd,student_even,teacher_odd,teacher_even,delta,clean)
    for value in values:_iq(value)
    if any(value.shape != clean.shape for value in values):
        raise ValueError('Matched perturbation and output batches required')
    t_power, s_power, input_power = power(teacher_odd), power(student_odd), power(delta)
    floor = (d.FLOOR_FRACTION*input_power).clamp_min(d.ABSOLUTE_FLOOR)
    reference = torch.maximum(t_power, floor)
    error = power(student_odd-teacher_odd)
    return _finite(dict(relative_response_nmse=error/reference,
        response_error_clean_nmse=error/power(clean).clamp_min(d.ABSOLUTE_FLOOR),
        student_response_gain=s_power/input_power.clamp_min(d.ABSOLUTE_FLOOR),
        teacher_response_gain=t_power/input_power.clamp_min(d.ABSOLUTE_FLOOR),
        student_even_to_odd=power(student_even)/torch.maximum(s_power,floor),
        teacher_even_to_odd=power(teacher_even)/reference,
        teacher_floor_active=(t_power < floor).to(torch.float64)))


def step_disagreement(low_odd, high_odd, low_step, high_step, input_iq):
    if not (math.isfinite(low_step) and math.isfinite(high_step) and 0 < low_step < high_step):
        raise ValueError('Two ordered positive steps required')
    low, high = _iq(low_odd)/low_step, _iq(high_odd)/high_step
    if low.shape != high.shape or low.shape != input_iq.shape:
        raise ValueError('Matched finite-step batches required')
    denominator = torch.maximum(power(low), (d.FLOOR_FRACTION*power(input_iq)).clamp_min(d.ABSOLUTE_FLOOR))
    return _finite(dict(disagreement=power(low-high)/denominator))['disagreement']
