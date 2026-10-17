"""Bounded finite-window allpass view. No normalization or hardware inversion."""
import numpy as np
from .design import LENGTH, CONTEXT, MAX_ALPHA
def _coefficients(a,batch):
    v=np.asarray(a)
    if v.dtype.kind not in "fiu" or np.iscomplexobj(v):
        raise TypeError("Real numeric coefficients required")
    v=np.asarray(v,dtype=np.float64)
    if v.ndim==0:v=np.full(batch,float(v),dtype=np.float64)
    if v.shape!=(batch,) or not np.isfinite(v).all() or np.any(np.abs(v)>MAX_ALPHA):
        raise ValueError("One bounded finite coefficient per packet")
    return v
def apply_allpass(x,a):
    """H=(a+z^-1)/(1+a*z^-1), followed by known one-sample alignment.

    Reflection is an explicit finite-record approximation, not true RX history.
    No RNG, external context, TX labels, clipping, post-RMS or resampling.
    """
    x=np.asarray(x)
    if x.dtype!=np.float32 or x.ndim!=3 or x.shape[0]<1 or x.shape[1:]!=(2,LENGTH):
        raise ValueError("Float32 Bx2x256 IQ required")
    if not np.isfinite(x).all() or np.any(np.sum(x.astype(np.float64)**2,axis=(1,2))<=0):
        raise ValueError("Finite positive-energy packets required")
    a=_coefficients(a,len(x))
    if np.all(a==0):return x.copy()
    extended=np.pad(x.astype(np.float64),((0,0),(0,0),(CONTEXT,CONTEXT)),mode="reflect")
    y=np.empty_like(extended);previous_x=extended[:,:,0].copy();previous_y=previous_x.copy()
    aa=a[:,None]
    for n in range(extended.shape[-1]):
        current=aa*extended[:,:,n]+previous_x-aa*previous_y
        y[:,:,n]=current;previous_x=extended[:,:,n];previous_y=current
    result=y[:,:,CONTEXT+1:CONTEXT+1+LENGTH]
    if result.shape!=x.shape or not np.isfinite(result).all():raise FloatingPointError("Invalid finite allpass output")
    return np.ascontiguousarray(result,dtype=np.float32)
def response(a,w):
    a=float(_coefficients(a,1)[0]);w=np.asarray(w,dtype=np.float64)
    if w.ndim!=1 or not np.isfinite(w).all():raise ValueError("Finite frequency grid required")
    # Pure numerical delay removed, as in apply_allpass. Not a causal deployment module.
    return (1+a*np.exp(1j*w))/(1+a*np.exp(-1j*w))
