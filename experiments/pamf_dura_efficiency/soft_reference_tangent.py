"""IQ feasibility and direction-preserving step loss for a shifted F reference."""
import math
import torch

ALPHA=.0033130025035573727


def minimum_shift(jacobian,offset,*,dimension=5):
    if dimension not in (5,10) or jacobian.ndim!=4 or jacobian.shape[1:]!=(dimension,2,256) or offset.shape!=(len(jacobian),dimension):
        raise ValueError('Paired source rank5 or explicit source10 IQ Jacobian and reference offset required')
    j=jacobian.detach().double().flatten(2);d=offset.detach().double()
    gram=j@j.transpose(1,2);values,vectors=torch.linalg.eigh(gram)
    defined=values>values[:,-1:].clamp_min(0)*1e-10
    reciprocal=torch.where(defined,values.clamp_min(1e-30).reciprocal(),torch.zeros_like(values))
    coefficient=(vectors@((vectors.transpose(1,2)@d[:,:,None])*reciprocal[:,:,None])).squeeze(2)
    shift=(j.transpose(1,2)@coefficient[:,:,None]).reshape(-1,2,256)
    residual=(j@shift.flatten(1)[:,:,None]).squeeze(2)-d
    norm=d.norm(dim=1);relative=torch.where(norm>1e-12,residual.norm(dim=1)/norm.clamp_min(1e-12),residual.norm(dim=1))
    return shift.to(jacobian),dict(rank=defined.sum(1),relative_residual=relative)


def prototype_tangent_loss(restored,clean,jacobian,offset,alpha=ALPHA,*,dimension=5):
    if (restored.ndim!=3 or restored.shape[1:]!=(2,256) or clean.shape!=restored.shape
        or dimension not in (5,10) or jacobian.shape!=(len(restored),dimension,2,256) or offset.shape!=(len(restored),dimension)
        or not math.isfinite(alpha) or alpha<=0):raise ValueError('Matched source IQ/J/offset and positive fixed alpha required')
    delta=restored-clean.detach();j=jacobian.detach();d=offset.detach()
    residual=(j*delta[:,None]).sum((2,3))-d;raw=residual.square().sum(1)
    with torch.no_grad():
        gradient=2*alpha*(residual.detach()[:,:,None,None]*j).sum(1);norm=gradient.flatten(1).norm(dim=1)
        IQnorm=2*delta.detach().flatten(1).norm(dim=1)/(512*clean.detach().square().mean((1,2)).clamp_min(1e-4))
        weight=torch.clamp(IQnorm/norm.clamp_min(1e-12),max=1)
        pressure=norm/IQnorm.clamp_min(1e-12)
    # The detached scalar bounds only this step's IQ-output gradient. With a
    # nonzero target offset there is no claim that its loss value <= IQ NMSE.
    return raw*weight,dict(weight=weight,IQ_gradient_norm=IQnorm,raw_gradient_norm=norm,
        raw_pressure=pressure,pressure_defined=IQnorm>1e-12,bounded_pressure=(norm*weight)/IQnorm.clamp_min(1e-12))


def calibration_decision(feasibility,views):
    if set(views)!={'1','20'}:raise ValueError('Both fixed cached source views required')
    f=feasibility
    feasible=(f['linear_residual_max']<=1e-3 and f['shift_NMSE_mean']<=.001 and f['shift_NMSE_p95']<=.005
        and f['nonlinear_relative_error_median']<=.5 and f['nonlinear_relative_error_p95']<=1 and f['choices_preserved']==512)
    rows=[]
    for view,row in views.items():
        if row['count']!=512 or set(row['RX_mean_cosine'])!=set(map(str,[1,3,4,6,8])):raise ValueError('Full512/five-sourceRX required')
        passed=(row['defined']==512 and row['defined_IQ']==512 and row['mean_cosine']>=.5 and row['mean_cosine']>=row['mean_IQ_cosine']+.1
            and row['nonnegative']>=384 and all(value is not None and value>=0 for value in row['RX_mean_cosine'].values())
            and row['maximum_balanced_pressure']<=1+2e-5)
        rows.append(dict(view=view,passed=bool(passed)))
    return dict(feasible=bool(feasible),views=rows,eligible=bool(feasible and all(r['passed'] for r in rows)),alpha=ALPHA,automatic_training=False)
