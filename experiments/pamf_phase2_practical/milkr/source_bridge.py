"""Frozen source-pair low-rank residual map from raw to restored features.

Fit is a source-stage operation only. Runtime load/transform accepts checkpoint
parameters and current features, never source examples or query statistics.
"""
from pathlib import Path
import numpy as np
from . import signature_kernel as sig

BLOCKS=('z_id','fft','t_emb','f_emb','pa_local','sig0','sig1','sig2')
DIMS=(160,96,160,160,160,112,112,112)


def blocks(features):
    x=sig.normalize(features)
    return [x[k] for k in BLOCKS[:5]]+[x['sig_views'][:,i*112:(i+1)*112]*np.sqrt(3) for i in range(3)]


class FeatureBridge:
    def __init__(self,parameters,rank=8):
        self.rank=int(rank);self.parameters=tuple(parameters)
        if len(self.parameters)!=8 or self.rank!=8:raise ValueError('Wrong frozen source bridge schema')
        for d,p in zip(DIMS,self.parameters):
            if len(p)!=4 or tuple(a.shape for a in p)!=((d,),(d,),(d,8),(8,d)):
                raise ValueError('Wrong bridge parameter dimensions')
            for a in p:
                if a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid bridge parameters')
                a.flags.writeable=False

    @classmethod
    def fit(cls,raw,restored,*,rank=8,ridge_fraction=.01):
        if rank!=8 or ridge_fraction!=.01:raise ValueError('Unregistered source bridge hyperparameters')
        X,Y=blocks(raw),blocks(restored);params=[]
        if len(X[0])<2 or any(len(x)!=len(X[0]) for x in X+Y):raise ValueError('Source pair alignment differs')
        for x,y in zip(X,Y):
            mx=x.mean(0);delta=y-x;md=delta.mean(0);xc=x-mx
            covariance=xc.T@xc/len(x)
            ridge=ridge_fraction*max(float(np.trace(covariance)/x.shape[1]),np.finfo(np.float64).eps)
            A=np.linalg.solve(covariance+ridge*np.eye(x.shape[1]),xc.T@(delta-md)/len(x))
            U,S,V=np.linalg.svd(A,full_matrices=False)
            params.append((mx,md,U[:,:rank]*S[:rank],V[:rank]))
        return cls(params,rank)

    def transform(self,features):
        values=[]
        for x,(mx,md,left,right) in zip(blocks(features),self.parameters):
            value=x+md+(x-mx)@left@right
            norm=np.linalg.norm(value,axis=1,keepdims=True)
            if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid bridged feature norm')
            value=value/norm;value.flags.writeable=False;values.append(value)
        out={name:value for name,value in zip(BLOCKS[:5],values[:5])}
        out['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
        return out

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError('Frozen source checkpoint already exists')
        payload={'schema':np.asarray('raw_to_restored_source_bridge_v1'),'rank':np.asarray(self.rank,dtype=np.int64)}
        for name,p in zip(BLOCKS,self.parameters):
            for label,a in zip(('mean_raw','mean_delta','left','right'),p):payload[name+'_'+label]=a
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!='raw_to_restored_source_bridge_v1':raise ValueError('Wrong bridge checkpoint')
            params=[tuple(np.array(z[name+'_'+label],dtype=np.float64,copy=True) for label in ('mean_raw','mean_delta','left','right')) for name in BLOCKS]
            return cls(params,int(z['rank'].item()))
