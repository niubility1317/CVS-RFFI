"""Matched extra full-restorer calls for source output or finite-response fitting."""
import torch
from . import design as d


def _native(value):
    if (not isinstance(value,torch.Tensor) or value.ndim!=3 or not len(value)
            or value.shape[1:]!=(2,256) or value.dtype not in (torch.float32,torch.float64)
            or not torch.isfinite(value).all()):
        raise ValueError('Finite native floating IQ[B,2,256] required')
    return value


def _power(value):
    return value.to(torch.float64).square().sum(1).mean(1)


def probe_spec(update):
    if type(update) is not int or update<0:
        raise ValueError('Nonnegative integer source update required')
    return d.PROBE_CYCLE[update%len(d.PROBE_CYCLE)]


def make_delta(value,update):
    value=_native(value).detach();probe,epsilon=probe_spec(update)
    if probe=='phase':
        direction=torch.stack((-value[:,1],value[:,0]),1)
    else:
        generator=torch.Generator(device='cpu').manual_seed(d.PROBE_SEED+update)
        direction=torch.randn(value.shape,generator=generator,dtype=value.dtype,device='cpu').to(value.device)
        ratio=(_power(value)/_power(direction).clamp_min(d.PROBE_ABSOLUTE_FLOOR)).sqrt().to(value.dtype)
        direction=direction*ratio[:,None,None]
    return direction*epsilon,dict(probe=probe,epsilon=epsilon,count=len(value))


def pair_loss(student_plus,student_minus,teacher_plus,teacher_minus,delta,arm):
    if arm not in d.ARMS:
        raise ValueError('Registered probe objective required')
    values=(student_plus,student_minus,teacher_plus,teacher_minus,delta)
    for value in values:_native(value)
    if any(v.shape!=student_plus.shape or v.device!=student_plus.device for v in values):
        raise ValueError('Matched paired source outputs and perturbations required')
    sp,sm=student_plus.to(torch.float64),student_minus.to(torch.float64)
    tp,tm=teacher_plus.detach().to(torch.float64),teacher_minus.detach().to(torch.float64)
    if arm=='pair_aug_control':
        loss=.5*(_power(sp-tp)/_power(tp).clamp_min(d.PROBE_ABSOLUTE_FLOOR)
                 +_power(sm-tm)/_power(tm).clamp_min(d.PROBE_ABSOLUTE_FLOOR))
    else:
        so,to=(sp-sm)*.5,(tp-tm)*.5
        floor=(d.PROBE_FLOOR_FRACTION*_power(delta.detach())).clamp_min(d.PROBE_ABSOLUTE_FLOOR)
        loss=_power(so-to)/torch.maximum(_power(to),floor)
    if not torch.isfinite(loss).all():
        raise FloatingPointError('Nonfinite source probe objective')
    return loss.to(student_plus.dtype)


def paired_probe(student,teacher,received,update,arm):
    if arm not in d.ARMS:
        raise ValueError('Registered probe arm required before forward')
    probe_spec(update);received=_native(received).detach()[:d.PROBE_COUNT]
    delta,info=make_delta(received,update)
    plus,minus=received+delta,received-delta
    with torch.no_grad():
        teacher_plus,teacher_minus=teacher(plus),teacher(minus)
    student_plus,student_minus=student(plus),student(minus)
    return pair_loss(student_plus,student_minus,teacher_plus,teacher_minus,delta,arm),info
