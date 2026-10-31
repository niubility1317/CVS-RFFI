"""Source CCA plus shared native directions orthogonal to both CCA maps."""
from pathlib import Path
import numpy as np
from .orbit_cca import OrbitCCA,blocks
from .source_bridge import BLOCKS,DIMS

SCHEMA='source_cca_complement_skip_v1'


def complement_basis(raw_projection,restored_projection,x,y,width):
    maps=np.concatenate((raw_projection,restored_projection),axis=1)
    u,s,_=np.linalg.svd(maps,full_matrices=False)
    rank=int((s>np.finfo(np.float64).eps*max(maps.shape)*s[0]).sum())
    span=u[:,:rank];joined=np.concatenate((x,y));mean=joined.mean(0)
    centered=joined-mean;residual=centered-centered@span@span.T
    values,vectors=np.linalg.eigh(residual.T@residual/len(residual))
    # Use the original variance scale: a fully removed subspace leaves roundoff,
    # whose own maximum eigenvalue cannot establish a positive residual rank.
    tolerance=np.finfo(np.float64).eps*max(centered.shape)*float(np.sum(centered**2)/len(centered))
    if values[-width]<=tolerance:
        raise FloatingPointError('Insufficient positive source residual directions')
    projection=vectors[:,-width:][:,::-1].copy()
    return mean,projection


class ResidualSkip:
    def __init__(self,cca,parameters):
        self.cca=cca;self.parameters=tuple(parameters);self.rank=64
        if not isinstance(cca,OrbitCCA) or len(self.parameters)!=8 or self.parameters[1] is not None or self.parameters[3] is not None:raise ValueError('Wrong residual-skip schema')
        for index,(d,p) in enumerate(zip(DIMS,self.parameters)):
            if index in (1,3):continue
            if p is None or len(p)!=2 or tuple(a.shape for a in p)!=((2*d,),(2*d,d-64)):raise ValueError('Wrong residual projection dimensions')
            for a in p:
                if a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid residual projection')
                a.flags.writeable=False

    @classmethod
    def fit(cls,raw,restored):
        cca=OrbitCCA.fit(raw,restored);parameters=[]
        for index,(x,y,d,(_,_,px,py)) in enumerate(zip(blocks(raw),blocks(restored),DIMS,cca.parameters)):
            # C4 shares FFT and its frequency embedding across phases. Their
            # std halves are identically zero; retain their original CCA only.
            parameters.append(None if index in (1,3) else complement_basis(px,py,x,y,d-64))
        return cls(cca,parameters)

    def transform(self,features,*,restored):
        aligned=self.cca.transform(features,restored=restored)
        aligned_blocks=[aligned[key] for key in BLOCKS[:5]]+[aligned['sig_views'][:,i*112:(i+1)*112]*np.sqrt(3) for i in range(3)]
        values=[]
        for x,a,p in zip(blocks(features),aligned_blocks,self.parameters):
            if p is None:value=a
            else:
                mean,projection=p;native=(x-mean)@projection;norm=np.linalg.norm(native,axis=1,keepdims=True)
                if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid residual feature norm')
                value=np.concatenate((a[:,:64],native/norm),axis=1)/np.sqrt(2)
            value.flags.writeable=False;values.append(value)
        out=dict(zip(BLOCKS[:5],values[:5]));out['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
        return out

    def save(self,path):
        if Path(path).exists():raise FileExistsError('Frozen residual-skip encoder exists')
        payload=dict(schema=np.asarray(SCHEMA))
        for name,cca,native in zip(BLOCKS,self.cca.parameters,self.parameters):
            for key,a in zip(('mean_raw','mean_restored','raw_projection','restored_projection'),cca):payload[name+'_'+key]=a
            if native is not None:
                for key,a in zip(('native_mean','native_projection'),native):payload[name+'_'+key]=a
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!=SCHEMA:raise ValueError('Wrong residual-skip checkpoint')
            cca=[tuple(np.array(z[name+'_'+key],dtype=np.float64,copy=True) for key in ('mean_raw','mean_restored','raw_projection','restored_projection')) for name in BLOCKS]
            native=[None if name in ('fft','f_emb') else tuple(np.array(z[name+'_'+key],dtype=np.float64,copy=True) for key in ('native_mean','native_projection')) for name in BLOCKS]
        return cls(OrbitCCA(cca),native)
