import torch
from . import design as d


def complex_iq(value):
    if (value.ndim!=3 or value.shape[1]!=2 or value.shape[2]<=max(d.LAGS)+1 or value.dtype not in (torch.float32,torch.float64)
            or not torch.isfinite(value).all()):raise ValueError('Finite floating complex IQ[B,2,N],N>maxlag+1 required')
    return torch.complex(value[:,0],value[:,1])


def descriptor(value):
    z=complex_iq(value);z=z-z.mean(1,keepdim=True)
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
    result=torch.stack(features,1)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite normalized moment descriptor')
    return result,power.lt(d.POWER_FLOOR).to(result.dtype)


def measure(restored,clean,parent):
    if restored.shape!=clean.shape or restored.shape!=parent.shape:raise ValueError('Matched source IQ required')
    rz,cz,pz=(complex_iq(v) for v in (restored,clean,parent))
    r,rf=descriptor(restored);c,cf=descriptor(clean);p,pf=descriptor(parent)
    ce=(r-c).square();pe=(r-p).square()
    result=dict(paired_clean_iq_nmse=(rz-cz).abs().square().mean(1)/cz.abs().square().mean(1).clamp_min(d.POWER_FLOOR),
        teacher_iq_nmse=(rz-pz).abs().square().mean(1)/pz.abs().square().mean(1).clamp_min(d.POWER_FLOOR),
        paired_clean_moment_mse=ce.mean(1),teacher_moment_mse=pe.mean(1),
        clean_energy_floor_active=cf,teacher_energy_floor_active=pf,restored_energy_floor_active=rf)
    for i,name in enumerate(d.FEATURES):result['paired_'+name+'_squared_error']=ce[:,i];result['teacher_'+name+'_squared_error']=pe[:,i]
    if any(not torch.isfinite(v).all() for v in result.values()):raise FloatingPointError('Nonfinite source moment comparison')
    return result
