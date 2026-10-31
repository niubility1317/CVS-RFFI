"""Source-trained paired encoder for C4 mean/std inputs, rank fixed at 64."""
from pathlib import Path
import numpy as np
from .source_bridge import BLOCKS,DIMS
from .paired_cca import inverse_root


def blocks(features):
    if set(features)!=set(BLOCKS[:5])|{'sig_views'}:raise ValueError('Exact orbit feature schema required')
    x=[features[key] for key in BLOCKS[:5]]+[features['sig_views'][:,i*224:(i+1)*224] for i in range(3)]
    result=[]
    for value,d in zip(x,DIMS):
        value=np.asarray(value,np.float64)
        if value.shape!=(len(x[0]),2*d) or not np.isfinite(value).all():raise ValueError('Wrong orbit input dimensions')
        norm=np.linalg.norm(value,axis=1,keepdims=True)
        if (norm<=1e-12).any():raise ValueError('Zero orbit feature')
        result.append(value/norm)
    if features['sig_views'].shape[1]!=672:raise ValueError('Wrong orbit signature dimensions')
    return result


class OrbitCCA:
    def __init__(self,parameters,rank=64):
        self.parameters=tuple(parameters);self.rank=int(rank)
        if self.rank!=64 or len(self.parameters)!=8:raise ValueError('Unregistered orbit CCA schema')
        for d,p in zip(DIMS,self.parameters):
            if len(p)!=4 or tuple(a.shape for a in p)!=((2*d,),(2*d,),(2*d,64),(2*d,64)):raise ValueError('Wrong orbit CCA parameter dimensions')
            for a in p:
                if a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid orbit encoder parameter')
                a.flags.writeable=False

    @classmethod
    def fit(cls,raw,restored):
        X,Y=blocks(raw),blocks(restored);parameters=[]
        if len(X[0])<2 or any(len(x)!=len(X[0]) for x in X+Y):raise ValueError('Source pair alignment differs')
        for x,y in zip(X,Y):
            mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my
            wx=inverse_root(xc.T@xc/len(x));wy=inverse_root(yc.T@yc/len(y))
            u,_,vt=np.linalg.svd(wx@(xc.T@yc/len(x))@wy,full_matrices=False)
            parameters.append((mx,my,wx@u[:,:64],wy@vt.T[:,:64]))
        return cls(parameters)

    def transform(self,features,*,restored):
        if not isinstance(restored,bool):raise ValueError('Explicit raw/restored path required')
        values=[]
        for x,d,(mx,my,px,py) in zip(blocks(features),DIMS,self.parameters):
            projected=(x-(my if restored else mx))@(py if restored else px)
            norm=np.linalg.norm(projected,axis=1,keepdims=True)
            if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid orbit CCA feature norm')
            value=np.zeros((len(x),d),np.float64);value[:,:64]=projected/norm;value.flags.writeable=False;values.append(value)
        out=dict(zip(BLOCKS[:5],values[:5]));out['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
        return out

    def save(self,path):
        if Path(path).exists():raise FileExistsError('Frozen orbit encoder exists')
        payload=dict(schema=np.asarray('source_phase_orbit_cca_v1'),rank=np.asarray(self.rank,np.int64))
        for name,p in zip(BLOCKS,self.parameters):
            for key,a in zip(('mean_raw','mean_restored','raw_projection','restored_projection'),p):payload[name+'_'+key]=a
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!='source_phase_orbit_cca_v1':raise ValueError('Wrong orbit CCA checkpoint')
            parameters=[tuple(np.array(z[name+'_'+key],dtype=np.float64,copy=True) for key in ('mean_raw','mean_restored','raw_projection','restored_projection')) for name in BLOCKS]
            return cls(parameters,int(z['rank'].item()))
