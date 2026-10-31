"""Source-only regularized CCA, separate raw/restored encoders in one space."""
from pathlib import Path
import numpy as np
from .source_bridge import BLOCKS,DIMS,blocks

RANKS=(16,32,64,96)


def inverse_root(covariance):
    ridge=.01*max(float(np.trace(covariance)/len(covariance)),np.finfo(np.float64).eps)
    values,vectors=np.linalg.eigh(covariance+ridge*np.eye(len(covariance)))
    return (vectors/np.sqrt(values))@vectors.T


class PairedCCA:
    def __init__(self,parameters,rank):
        self.parameters=tuple(parameters);self.rank=int(rank)
        if self.rank not in RANKS or len(self.parameters)!=8:raise ValueError('Unregistered CCA rank/schema')
        for d,p in zip(DIMS,self.parameters):
            if len(p)!=4 or tuple(a.shape for a in p)!=((d,),(d,),(d,self.rank),(d,self.rank)):
                raise ValueError('Wrong CCA parameter dimensions')
            for a in p:
                if a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid CCA parameter')
                a.flags.writeable=False

    @classmethod
    def fit(cls,raw,restored,rank):
        if rank not in RANKS:raise ValueError('Unregistered CCA rank')
        X,Y=blocks(raw),blocks(restored);parameters=[]
        if len(X[0])<2 or any(len(a)!=len(X[0]) for a in X+Y):raise ValueError('Source pair alignment differs')
        for x,y in zip(X,Y):
            mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my
            wx=inverse_root(xc.T@xc/len(x));wy=inverse_root(yc.T@yc/len(y))
            u,_,vt=np.linalg.svd(wx@(xc.T@yc/len(x))@wy,full_matrices=False)
            parameters.append((mx,my,wx@u[:,:rank],wy@vt.T[:,:rank]))
        return cls(parameters,rank)

    def transform(self,features,*,restored):
        if not isinstance(restored,bool):raise ValueError('Explicit raw/restored path required')
        values=[]
        for x,d,(mx,my,px,py) in zip(blocks(features),DIMS,self.parameters):
            projected=(x-(my if restored else mx))@(py if restored else px)
            norm=np.linalg.norm(projected,axis=1,keepdims=True)
            if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid CCA feature norm')
            value=np.zeros((len(x),d),np.float64);value[:,:self.rank]=projected/norm;value.flags.writeable=False;values.append(value)
        out=dict(zip(BLOCKS[:5],values[:5]));out['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
        return out

    def save(self,path):
        if Path(path).exists():raise FileExistsError('Frozen CCA checkpoint exists')
        payload=dict(schema=np.asarray('source_paired_cca_v1'),rank=np.asarray(self.rank,np.int64))
        for name,p in zip(BLOCKS,self.parameters):
            for key,a in zip(('mean_raw','mean_restored','raw_projection','restored_projection'),p):payload[name+'_'+key]=a
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!='source_paired_cca_v1':raise ValueError('Wrong CCA checkpoint')
            parameters=[tuple(np.array(z[name+'_'+key],dtype=np.float64,copy=True) for key in ('mean_raw','mean_restored','raw_projection','restored_projection')) for name in BLOCKS]
            return cls(parameters,int(z['rank'].item()))
