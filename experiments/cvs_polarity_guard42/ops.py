"""Exact coordinate reflection;invertibility does not establish RF identity invariance."""
import torch
def conjugate_iq(x):
 if not torch.is_tensor(x) or x.ndim!=3 or x.shape[0]<1 or x.shape[1:]!=(2,256) or x.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(x).all()):raise ValueError('Finite Bx2x256 real floating IQ required')
 y=x.clone();y[:,1].neg_();return y

def invariant_summary(x,y):
 if x.shape!=y.shape or not torch.equal(conjugate_iq(y),x) or not torch.equal(y[:,0],x[:,0]) or not torch.equal(y[:,1],-x[:,1]):raise ValueError('Exact same-packet reflection/inverse required')
 p0=x.square().sum(1);p1=y.square().sum(1)
 if not torch.equal(p0,p1):raise ValueError('Reflection must preserve every sample magnitude')
 return dict(inverse_exact=True,sample_magnitude_exact=True,maximum_sample_power_error=float((p0-p1).abs().max()),relative_IQ_L2_mean=float(((y-x).square().sum((1,2))/(x.square().sum((1,2)).clamp_min(1e-30))).sqrt().mean()),boundary_samples_added=0,FIR_delay=0,new_noise=False,post_normalization=False)
