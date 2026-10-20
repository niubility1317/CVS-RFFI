import torch
from . import design as d
from experiments.pamf_r46_conditional_factor import models as architecture
from experiments.pamf_r46_complex_fidelity.models import balanced_complex_error,spectral_error


class CompactSuccessor(architecture.CompactSuccessor):
    def __init__(self,arm):
        if arm not in d.ARMS:raise ValueError('Registered moment-fidelity arm required')
        super().__init__('factor625_stage');self.arm=arm


def compact_from_parent(parent,arm):
    if arm not in d.ARMS:raise ValueError('Registered moment arm required before initialization')
    model=architecture.compact_from_parent(parent,'factor625_stage');model.arm=arm
    return model


def moment_descriptor(value):
    """R29 arithmetic, native shapes; training checks aggregate finite loss/gradients.

    Avoid per-descriptor GPU-to-host finite checks. No fit, extra network or labels.
    """
    if value.ndim!=3 or value.shape[1:]!=(2,256) or value.dtype not in (torch.float32,torch.float64):
        raise ValueError('Native floating IQ[B,2,256] required')
    z=torch.complex(value[:,0],value[:,1]);z=z-z.mean(1,keepdim=True)
    power=z.abs().square().mean(1);den=power.clamp_min(d.POWER_FLOOR).square()
    m20=z.square().mean(1);m40=z.pow(4).mean(1);m42=z.abs().pow(4).mean(1)
    c40=m40-3*m20.square();c42=m42-m20.abs().square()-2*power.square()
    features=[c40.abs()/den,c42/den,m42/den]
    for lag in d.LAGS:
        a,b=z[:,lag:],z[:,:-lag];a=a-a.mean(1,keepdim=True);b=b-b.mean(1,keepdim=True)
        pa=a.abs().square().mean(1);pb=b.abs().square().mean(1)
        cumulant=(a.square()*a.conj()*b.conj()).mean(1)-2*pa*(a*b.conj()).mean(1)-a.square().mean(1)*(a.conj()*b.conj()).mean(1)
        normalized=cumulant/(pa.clamp_min(d.POWER_FLOOR).pow(3)*pb.clamp_min(d.POWER_FLOOR)).sqrt()
        features.extend((normalized.real,normalized.imag))
    return torch.stack(features,1)


def paired_outputs(model,teacher,received,clean):
    if model.arm not in d.ARMS:raise ValueError('Registered moment arm required')
    with torch.no_grad():desired=teacher.stage_outputs(received)[0]
    return model.stage_outputs(received)[0],desired,model(clean.detach())


def objective(student,teacher,clean,epoch,arm,identity_output=None):
    if arm not in d.ARMS or len(student)!=3 or len(teacher)!=3 or any(v.shape!=clean.shape for v in (*student,*teacher)):
        raise ValueError('Registered three-stage paired source IQ required')
    if identity_output is None or identity_output.shape!=clean.shape:
        raise ValueError('Actual additional clean forward required for both moment arms')
    target=clean.detach();energy=target.square().mean((1,2)).clamp_min(1e-4)
    paired=(student[-1]-target).square().mean((1,2))/energy
    teacher_error=torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(student,teacher)]).mean(0)
    spectral=spectral_error(student[-1],target);complex_error=balanced_complex_error(student[-1],target)
    identity=(identity_output-target).square().mean((1,2))/energy
    base=paired+.05*spectral+d.teacher_weight(epoch)*teacher_error+d.COMPLEX_WEIGHT*complex_error+d.CLEAN_WEIGHT[arm]*identity
    with torch.no_grad():
        clean_descriptor=moment_descriptor(target)
        parent_descriptor=moment_descriptor(teacher[-1].detach())
    estimate=moment_descriptor(student[-1])
    clean_moment=(estimate-clean_descriptor).square().mean(1)
    teacher_moment=(estimate-parent_descriptor).square().mean(1)
    selected=clean_moment if d.MOMENT_REFERENCE[arm]=='paired_clean' else teacher_moment
    weighted=d.MOMENT_WEIGHT*selected
    return base+weighted,dict(iq_nmse=paired,spectral_error=spectral,teacher_stage_nmse=teacher_error,
        balanced_complex_nmse=complex_error,clean_identity_nmse=identity,base_loss=base,
        clean_moment_mse=clean_moment,teacher_moment_mse=teacher_moment,moment_objective=selected,
        weighted_moment_objective=weighted)
