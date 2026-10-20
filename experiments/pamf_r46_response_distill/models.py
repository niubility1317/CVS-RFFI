import torch
from . import design as d
from experiments.pamf_r46_conditional_factor import models as baseline
from experiments.pamf_r46_conditional_factor.models import spectral_error


class CompactSuccessor(baseline.CompactSuccessor):
    def __init__(self, arm):
        if arm not in d.ARMS:
            raise ValueError('Registered response-distill arm required')
        super().__init__('factor625_stage')
        self.arm = arm


def compact_from_parent(parent, arm):
    if arm not in d.ARMS:
        raise ValueError('Registered response-distill arm required before initialization')
    model = baseline.compact_from_parent(parent, 'factor625_stage')
    model.arm = arm
    return model


def balanced_complex_error(restored, clean):
    if restored.shape != clean.shape or restored.ndim != 3 or restored.shape[1:] != (2,256):
        raise ValueError('Matched native IQ2x256 required')
    target = clean.detach()
    reference = torch.fft.fft(torch.complex(target[:,0], target[:,1]), norm='ortho')
    estimate = torch.fft.fft(torch.complex(restored[:,0], restored[:,1]), norm='ortho')
    power = reference.abs().square()
    order = power.argsort(1)
    error = (estimate-reference).abs().square()
    global_power = power.mean(1)
    groups = []
    for q in range(4):
        take = order[:,q*64:(q+1)*64]
        denominator = torch.maximum(power.gather(1,take).mean(1), .01*global_power).clamp_min(1e-4)
        groups.append(error.gather(1,take).mean(1)/denominator)
    return torch.stack(groups).mean(0)


def paired_outputs(model, teacher, received, clean):
    if model.arm not in d.ARMS:
        raise ValueError('Registered arm required')
    with torch.no_grad():
        desired = teacher.stage_outputs(received)[0]
    outputs = model.stage_outputs(received)[0]
    identity = model(clean.detach()) if d.CLEAN_WEIGHT[model.arm] else None
    return outputs, desired, identity


def objective(student, teacher, clean, epoch, arm, identity_output=None):
    if arm not in d.ARMS or len(student) != 3 or len(teacher) != 3 or any(v.shape != clean.shape for v in (*student,*teacher)):
        raise ValueError('Registered three-stage paired source IQ required')
    target = clean.detach()
    energy = target.square().mean((1,2)).clamp_min(1e-4)
    paired = (student[-1]-target).square().mean((1,2))/energy
    teacher_error = torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(student,teacher)]).mean(0)
    spectral = spectral_error(student[-1], target)
    complex_error = balanced_complex_error(student[-1], target)
    if d.CLEAN_WEIGHT[arm]:
        if identity_output is None or identity_output.shape != clean.shape:
            raise ValueError('Actual additional clean forward required')
        identity = (identity_output-target).square().mean((1,2))/energy
    else:
        if identity_output is not None:
            raise ValueError('Complex-only arm must not execute identity branch')
        identity = paired.new_zeros(paired.shape)
    loss = paired+.05*spectral+d.teacher_weight(epoch)*teacher_error+d.COMPLEX_WEIGHT*complex_error+d.CLEAN_WEIGHT[arm]*identity
    return loss, dict(iq_nmse=paired, spectral_error=spectral, teacher_stage_nmse=teacher_error,
                      balanced_complex_nmse=complex_error, clean_identity_nmse=identity)
