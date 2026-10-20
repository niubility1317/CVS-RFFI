import numpy as np
from . import design as d


def select_positions(rx):
    rx=np.asarray(rx);selected=[]
    if rx.shape!=(1800,) or set(rx.tolist())!=set(d.RECEIVERS):raise ValueError('Original1800 sourceV/fiveRX required')
    for receiver in d.RECEIVERS:
        positions=np.flatnonzero(rx==receiver)
        if len(positions)<d.PER_RECEIVER:raise ValueError('36sourcepositionsperRX required')
        selected.extend(positions[np.linspace(0,len(positions)-1,d.PER_RECEIVER,dtype=np.int64)].tolist())
    result=np.asarray(selected,np.int64)
    if len(result)!=d.COUNT or len(set(result.tolist()))!=d.COUNT:raise ValueError('Fixed180unique sourcepositions required')
    return result


def complex_iq(x):return x[:,0].astype(np.float64)+1j*x[:,1].astype(np.float64)
def iq(z):return np.stack((z.real,z.imag),1).astype(np.float32)


def counterfactuals(received,parent,clean):
    values=(received,parent,clean)
    if any(x.shape!=received.shape or x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in values):
        raise ValueError('Matching finite source FP32 Bx2x256 received/R03/clean required')
    restored=complex_iq(parent);reference=complex_iq(clean)
    tail=parent.copy();tail[:,:,232:]=clean[:,:,232:]
    prefix=parent.copy();prefix[:,:,:232]=clean[:,:,:232]
    spectrum=np.fft.fft(restored,norm='ortho');target=np.fft.fft(reference,norm='ortho')
    order=np.argsort(np.abs(target)**2,axis=-1,kind='stable');spectral={}
    for name,index in (('clean_low64',order[:,:64]),('clean_high64',order[:,-64:])):
        replaced=spectrum.copy();np.put_along_axis(replaced,index,np.take_along_axis(target,index,axis=-1),axis=-1)
        spectral[name]=iq(np.fft.ifft(replaced,norm='ortho'))
    gain=np.sum(restored.conj()*reference,-1)/np.maximum(np.sum(np.abs(restored)**2,-1),1e-12)
    result=dict(identity=received.copy(),parent_R03=parent.copy(),clean_tail24=tail,clean_prefix232=prefix,
        **spectral,clean_gain=iq(gain[:,None]*restored),clean_reference=clean.copy())
    if tuple(result)!=d.READOUTS or any(not np.isfinite(x).all() for x in result.values()):raise ValueError('Finite registeredcounterfactuals required')
    return result
