"""Per-record fixed phase-orbit moments of the unchanged frozen F.

G, when requested, consumes the original record once before the C4 action.
There is no new observation, label, cross-record normalization or fitted state.
"""
import numpy as np
from .features import tensor,fft96
from .signature_features import capture
from .milkr.source_bridge import BLOCKS,blocks


def rotate(iq,quarter):
    x=np.asarray(iq,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Invalid phase-orbit IQ')
    if quarter not in (0,1,2,3):raise ValueError('C4 quarter required')
    if quarter==0:return x
    if quarter==1:return np.stack((-x[:,1],x[:,0]),axis=1)
    if quarter==2:return -x
    return np.stack((x[:,1],-x[:,0]),axis=1)


def rotate_eighth(iq,eighth):
    if eighth not in range(8):raise ValueError('C8 eighth required')
    if eighth%2==0:return rotate(iq,eighth//2)
    x=rotate(iq,(eighth-1)//2);scale=np.float32(np.sqrt(.5))
    return np.stack((scale*(x[:,0]-x[:,1]),scale*(x[:,0]+x[:,1])),axis=1)


def moments(phases):
    if len(phases)!=4:raise ValueError('Exactly four per-record phases required')
    return _moments(phases)


def spacetime_moments(phases):
    if len(phases)!=16:raise ValueError('Exactly sixteen per-record spacetime views required')
    return _moments(phases)


def eighth_moments(phases):
    if len(phases)!=8:raise ValueError('Exactly eight per-record phases required')
    return _moments(phases)


def _moments(phases):
    normalized=[blocks(f) for f in phases];values=[]
    for index in range(8):
        a=np.stack([f[index] for f in normalized])
        values.append(np.concatenate((a.mean(0),a.std(0)),axis=1))
    result=dict(zip(BLOCKS[:5],values[:5]));result['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
    return result


def stack_views(phases):
    return {key:np.stack([f[key] for f in phases],axis=1) for key in phases[0]}


def extract(reader,raw,device,engine=None,batch=128,*,aggregate=moments,return_phase_logits=False,time_shifts=(0,),phase_eighths=False):
    import torch
    if time_shifts not in ((0,),(0,64,128,192)):raise ValueError('Unregistered cyclic time orbit')
    if phase_eighths and time_shifts!=(0,):raise ValueError('C8 is a phase-only orbit')
    output=[];logits=[]
    with torch.inference_mode():
        for start in range(0,len(raw),batch):
            base=np.asarray(raw[start:start+batch],np.float32)
            if engine is not None:base=np.asarray(engine(tensor(base,device))[0].cpu().tolist(),np.float32)
            spectral=fft96(base);phases=[];phase_logits=[]
            for shift in time_shifts:
                shifted=base if shift==0 else np.roll(base,shift,axis=2)
                if shift!=0:spectral=fft96(shifted)
                for quarter in range(8 if phase_eighths else 4):
                    phase=rotate_eighth(shifted,quarter) if phase_eighths else rotate(shifted,quarter)
                    f,l=capture(reader,tensor(phase,device))
                    phases.append(dict({key:np.asarray(value.cpu().tolist(),np.float32) for key,value in f.items()},fft=spectral))
                    if shift==0 and (not phase_eighths or quarter%2==0):
                        if return_phase_logits:phase_logits.append(np.asarray(l.cpu().tolist(),np.float32))
                        elif quarter==0:logits.append(np.asarray(l.cpu().tolist(),np.float32))
            output.append(aggregate(phases))
            if return_phase_logits:logits.append(np.stack(phase_logits,axis=1))
    return {key:np.concatenate([value[key] for value in output]) for key in output[0]},np.concatenate(logits)
