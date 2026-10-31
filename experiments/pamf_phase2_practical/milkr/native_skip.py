"""Frozen equal-energy CCA/native-PCA skip, trained on legal source pairs only."""
from pathlib import Path
import numpy as np
from .orbit_cca import OrbitCCA,blocks
from .source_bridge import BLOCKS,DIMS

SCHEMA='source_orbit_native_skip_v1'


class NativeSkip:
    def __init__(self,cca,parameters):
        if not isinstance(cca,OrbitCCA) or len(parameters)!=8:raise ValueError('Wrong native-skip schema')
        self.cca=cca;self.parameters=tuple(parameters);self.rank=64
        for d,p in zip(DIMS,self.parameters):
            if len(p)!=2 or tuple(a.shape for a in p)!=((2*d,),(2*d,d-64)):raise ValueError('Wrong native projection dimensions')
            for a in p:
                if a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid native projection')
                a.flags.writeable=False

    @classmethod
    def fit(cls,raw,restored):
        cca=OrbitCCA.fit(raw,restored);parameters=[]
        for x,y,d in zip(blocks(raw),blocks(restored),DIMS):
            # The SAME native basis/center is used for both observed paths.
            joined=np.concatenate((x,y));mean=joined.mean(0);centered=joined-mean
            _,vectors=np.linalg.eigh(centered.T@centered/len(joined))
            parameters.append((mean,vectors[:,-(d-64):][:,::-1].copy()))
        return cls(cca,parameters)

    def transform(self,features,*,restored):
        aligned=self.cca.transform(features,restored=restored)
        aligned_blocks=[aligned[key] for key in BLOCKS[:5]]+[aligned['sig_views'][:,i*112:(i+1)*112]*np.sqrt(3) for i in range(3)]
        values=[]
        for x,a,d,(mean,projection) in zip(blocks(features),aligned_blocks,DIMS,self.parameters):
            native=(x-mean)@projection;norm=np.linalg.norm(native,axis=1,keepdims=True)
            if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid native feature norm')
            # CCA keeps its fixed 64 coordinates; native PCA fills the rest.
            value=np.concatenate((a[:,:64],native/norm),axis=1)/np.sqrt(2)
            if value.shape!=(len(x),d):raise ValueError('Wrong native-skip output')
            value.flags.writeable=False;values.append(value)
        out=dict(zip(BLOCKS[:5],values[:5]));out['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
        return out

    def save(self,path):
        if Path(path).exists():raise FileExistsError('Frozen native-skip encoder exists')
        payload=dict(schema=np.asarray(SCHEMA))
        for name,cca,native in zip(BLOCKS,self.cca.parameters,self.parameters):
            for key,a in zip(('mean_raw','mean_restored','raw_projection','restored_projection'),cca):payload[name+'_'+key]=a
            for key,a in zip(('native_mean','native_projection'),native):payload[name+'_'+key]=a
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!=SCHEMA:raise ValueError('Wrong native-skip checkpoint')
            cca=[tuple(np.array(z[name+'_'+key],dtype=np.float64,copy=True) for key in ('mean_raw','mean_restored','raw_projection','restored_projection')) for name in BLOCKS]
            native=[tuple(np.array(z[name+'_'+key],dtype=np.float64,copy=True) for key in ('native_mean','native_projection')) for name in BLOCKS]
        return cls(OrbitCCA(cca),native)
