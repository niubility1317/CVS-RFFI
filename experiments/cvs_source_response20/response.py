"""Bounded finite IQ response, paired with a norm-matched direction control.

Source contrasts describe RX-associated observations, not identified hardware.
This operator adds a random response; it never inverts the recorded TX/RX.
"""
import numpy as np

RADIUS=.06
TAPS=33
ROTATION_SEED=20261016011
SESSION_SEED=20261016023


def directions(predicted):
    p=np.asarray(predicted,dtype=float)
    if p.shape!=(18,5,8) or not np.isfinite(p).all():
        raise ValueError('Need fixed 18-fold source prediction array')
    # Average over all six excluded TX values; each bank entry is TX-balanced.
    values=p.reshape(6,3,5,8).mean(0).reshape(15,8)
    if np.any(np.linalg.norm(values,axis=1)<1e-12):
        raise ValueError('No nonzero source direction in every fixed bank entry')
    return values


def unit_kernel(coordinates):
    c=np.asarray(coordinates,dtype=float)
    if c.shape!=(8,) or not np.isfinite(c).all():raise ValueError('Need eight finite coordinates')
    # DCT evaluated at all 256 frequency-bin centers; no periodic IQ filtering.
    basis=np.cos(np.pi*(np.arange(256)[:,None]+.5)*np.arange(1,9)[None,:]/256)
    profile=basis@c
    impulse=np.fft.ifft(np.fft.ifftshift(profile))
    offsets=np.arange(-16,17)
    kernel=impulse[offsets%256]*np.hanning(TAPS)
    kernel[16]=0 # no budget spent on a scalar gain removed by RMS restoration
    norm=float(np.linalg.norm(kernel))
    if norm<1e-12:raise ValueError('Degenerate finite response')
    return kernel/norm


def bank(predicted):
    values=directions(predicted)
    rng=np.random.default_rng(ROTATION_SEED)
    q,r=np.linalg.qr(rng.normal(size=(8,8)))
    rotation=q*np.where(np.diag(r)<0,-1,1)[None,:]
    physical=[];control=[]
    for value in values:
        a=unit_kernel(value);b=unit_kernel(rotation@value)
        # Equal impulse L2 norm and common proven response bound in each pair.
        scale=RADIUS/max(float(np.abs(a).sum()),float(np.abs(b).sum()))
        physical.append(a*scale);control.append(b*scale)
    return dict(empirical=np.array(physical),rotated=np.array(control),rotation=rotation,
        source_scope='L-derived TX-balanced RX/day-associated contrasts; not pure hardware',
        radius=RADIUS,taps=TAPS,normalization='pair-matched impulse L2; common delta L1 <= .06')


def session(epoch,rx,day,eq,view=0):
    if (not all(isinstance(v,(int,np.integer)) for v in (epoch,rx,day,eq,view))
        or epoch<1 or rx not in (1,3,4,6,8) or day not in (1,2,3) or eq<0 or view<0):
        raise ValueError('Invalid source observation session')
    rng=np.random.default_rng(np.random.SeedSequence([SESSION_SEED,epoch,rx,day,eq,view]))
    return int(rng.integers(15)),float(rng.uniform(-1,1))


def kernels(response_bank,kind,sessions):
    if kind not in ('empirical','rotated'):raise ValueError('Unregistered response kind')
    values=np.asarray(response_bank[kind],dtype=complex)
    if values.shape!=(15,33) or not np.isfinite(values).all():raise ValueError('Invalid finite bank')
    result=[]
    for item in sessions:
        if not isinstance(item,tuple) or len(item)!=5:raise ValueError('Session needs epoch/RX/day/eq/view only')
        index,strength=session(*item);h=values[index]*strength
        h=h.copy();h[16]+=1;result.append(h)
    return np.array(result)


def apply_numpy(iq,h,boundary='reflect',restore_rms=True):
    x=np.asarray(iq,dtype=float);h=np.asarray(h,dtype=complex)
    if (x.ndim!=3 or x.shape[1:]!=(2,256) or h.shape!=(len(x),33)
        or not np.isfinite(x).all() or not np.isfinite(h).all() or boundary not in ('reflect','edge')):
        raise ValueError('Finite Bx2x256 IQ, Bx33 kernels, reflect/edge required')
    z=x[:,0]+1j*x[:,1]
    padded=np.pad(z,((0,0),(16,16)),mode=boundary)
    windows=np.lib.stride_tricks.sliding_window_view(padded,33,axis=1)
    y=np.sum(windows*h[:,None,::-1],axis=-1)
    if restore_rms:
        old=np.mean(np.abs(z)**2,axis=1);new=np.mean(np.abs(y)**2,axis=1)
        scale=np.sqrt(np.divide(old,new,out=np.ones_like(old),where=new>0));y*=scale[:,None]
    return np.stack((y.real,y.imag),axis=1)


def apply_torch(iq,h,boundary='reflect',restore_rms=True):
    import torch
    import torch.nn.functional as F
    if (iq.ndim!=3 or tuple(iq.shape[1:])!=(2,256) or boundary not in ('reflect','edge')
        or not torch.isfinite(iq).all()):raise ValueError('Finite Bx2x256 IQ required')
    h=np.asarray(h,dtype=complex)
    if h.shape!=(len(iq),33) or not np.isfinite(h).all():raise ValueError('Need finite Bx33 kernels')
    # Explicit lists avoid a NumPy ABI bridge in the verified local torch runtime.
    hr=torch.tensor(h.real[:,::-1].tolist(),device=iq.device,dtype=iq.dtype)[:,None,:]
    hi=torch.tensor(h.imag[:,::-1].tolist(),device=iq.device,dtype=iq.dtype)[:,None,:]
    window=F.pad(iq,(16,16),mode='reflect' if boundary=='reflect' else 'replicate').unfold(-1,33,1)
    real=(window[:,0]*hr-window[:,1]*hi).sum(-1)
    imag=(window[:,0]*hi+window[:,1]*hr).sum(-1)
    y=torch.stack((real,imag),dim=1)
    if not restore_rms:return y
    old=iq.square().sum(1).mean(1);new=y.square().sum(1).mean(1)
    scale=torch.sqrt(torch.where(new>0,old/new.clamp_min(torch.finfo(iq.dtype).tiny),torch.ones_like(new)))
    return y*scale[:,None,None]
