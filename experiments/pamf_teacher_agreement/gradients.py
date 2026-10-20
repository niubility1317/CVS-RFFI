import torch
from . import design as d
from experiments.pamf_gradient_balance.gradients import merge_gradients as capped_merge


def teacher_agreement(teachers,arm):
    if arm not in d.ARMS or teachers.ndim!=4 or teachers.shape[0]!=3 or teachers.shape[2:]!=(2,256) or teachers.dtype!=torch.float32 or not torch.isfinite(teachers).all():
        raise ValueError('Three matching finite FP32 teacher IQ gradients required')
    pairs=((0,1),(0,2),(1,2));dots=[];norms=[]
    dimensions=(1,2) if arm==d.ARMS[0] else (1,)
    for a,b in pairs:
        dots.append((teachers[a]*teachers[b]).sum(dimensions))
        norms.append((teachers[a].square().sum(dimensions)*teachers[b].square().sum(dimensions)).sqrt())
    dots=torch.stack(dots);norms=torch.stack(norms);keep=(dots>=0).all(0)
    raw=teachers.mean(0);mask=keep[:,None,None] if arm==d.ARMS[0] else keep[:,None,:]
    agreed=raw*mask
    valid=norms>d.COSINE_DENOMINATOR_MIN
    cosine=torch.where(valid,dots/norms.clamp_min(d.COSINE_DENOMINATOR_MIN),torch.zeros_like(dots))
    active=raw.square().sum((1,2) if arm==d.ARMS[0] else (1,))>0
    after=(teachers*agreed[None]).sum((2,3))
    stats=dict(units=keep.numel(),agreed=int(keep.sum()),active_agreed=int((keep&active).sum()),
        pair_conflicts=[int((value<0).sum()) for value in dots],pair_valid=[int(value.sum()) for value in valid],
        pair_cosine_sum=[float(value.sum()) for value in cosine],raw_norm2=float(raw.square().sum()),agreed_norm2=float(agreed.square().sum()),
        teacher_negative_after=int((after<0).sum()),teacher_dot_after_min=float(after.min()))
    return agreed.detach(),stats


def output_gradient(estimate,paired,per_teacher,arm):
    if (paired.shape!=(len(estimate),) or not isinstance(per_teacher,(tuple,list)) or len(per_teacher)!=3 or
        any(value.shape!=paired.shape for value in per_teacher)):raise ValueError('One waveform and three separate teacher scalars per IQ required')
    wave=torch.autograd.grad(paired.sum(),estimate,retain_graph=True)[0]
    teachers=torch.stack([torch.autograd.grad(per_teacher[i].sum(),estimate,retain_graph=i<2)[0] for i in range(3)])
    semantic,agreement=teacher_agreement(teachers,arm)
    merged,stats=capped_merge(wave,semantic,'per_frame_cap');stats['agreement']=agreement
    return merged.detach()/len(estimate),stats
