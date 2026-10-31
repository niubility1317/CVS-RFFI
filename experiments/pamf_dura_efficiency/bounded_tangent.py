"""Cap a fixed local feature metric relative to paired-source IQ reconstruction."""
import math
import torch


@torch.no_grad()
def cap_metric(jacobian,energy,weight):
    if (jacobian.ndim!=4 or jacobian.shape[1:]!=(5,2,256) or energy.shape!=(len(jacobian),)
            or not math.isfinite(weight) or weight<=0 or not torch.isfinite(jacobian).all()
            or not torch.isfinite(energy).all() or not (energy>0).all()):
        raise ValueError('Finite fixed5x2x256 source metric, positive IQ energy and weight required')
    flat=jacobian.detach().flatten(2).double();e=energy.detach().to(flat)
    gram=flat@flat.transpose(1,2)
    if not torch.isfinite(gram).all():raise FloatingPointError('Source metric Gram overflow')
    values,basis=torch.linalg.eigh(gram)
    if (values[:,0]<-1e-8*values[:,-1].clamp_min(1)).any():raise FloatingPointError('Invalid source PSD eigensystem')
    values=values.clamp_min(0);threshold=1/(weight*e[:,None])
    scale=(threshold/torch.maximum(values,threshold)).sqrt()
    factor=(scale[:,:,None]*(basis.transpose(1,2)@flat)).reshape_as(jacobian).to(jacobian.dtype)
    relative=weight*e[:,None]*values
    check=factor.flatten(2).double();bounded_max=torch.linalg.eigvalsh(check@check.transpose(1,2))[:,-1]*weight*e
    return factor,dict(raw_relative_eigenvalues=relative,raw_trace=values.sum(1),clipped_record=(relative>1).any(1),
        clipped_directions=(relative>1).sum(1),bounded_relative_max=bounded_max)


def eligibility(clipped_fraction,maximum_relative_eigenvalue,rows):
    if set(rows)!={'1','20'}:raise ValueError('Both original source views required')
    physical=clipped_fraction>=.05 and maximum_relative_eigenvalue<=1+2e-5
    checks=[]
    for view,row in rows.items():
        if row['count']!=512 or set(row['rx_bounded_mean'])!=set(map(str,[1,3,4,6,8])):raise ValueError('All source512/RX required')
        passed=(row['defined']==512 and row['raw_pressure_p95']>1 and row['bounded_mean_cosine']>=.5
            and row['bounded_mean_cosine']>=row['raw_mean_cosine']-.02 and row['bounded_nonnegative']>=384
            and all(value is not None and value>=0 for value in row['rx_bounded_mean'].values())
            and row['maximum_bounded_pressure']<=1+2e-5)
        checks.append(dict(view=view,passed=bool(passed)))
    return dict(eligible=bool(physical and all(row['passed'] for row in checks)),full_source_pressure_qualified=bool(physical),
        views=checks,automatic_training=False)
