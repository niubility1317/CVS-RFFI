"""Source-L supervised cross-RX episodic diagonal metric on frozen CCA axes."""
from pathlib import Path
import numpy as np
from .orbit_cca import OrbitCCA
from .source_bridge import BLOCKS,DIMS

SCHEMA='source_orbit_episodic_metric_v1'
STEPS=600
LOG_BOUND=np.log(4.)


def coordinates(features):
    values=[features[key][:,:64] for key in BLOCKS[:5]]
    values += [features['sig_views'][:,i*112:i*112+64]*np.sqrt(3) for i in range(3)]
    return np.stack(values,axis=1)


class Episodes:
    """One observation per physical support ID, always two different RXs."""
    def __init__(self,labels,receivers,physical_ids,seed):
        self.y=np.asarray(labels,np.int64);self.rx=np.asarray(receivers,np.int64);self.ids=np.asarray(physical_ids,dtype=str)
        if self.y.ndim!=1 or self.y.shape!=self.rx.shape or self.y.shape!=self.ids.shape:raise ValueError('Source metadata alignment differs')
        self.classes=np.unique(self.y);self.receivers=np.unique(self.rx);self.rng=np.random.default_rng(seed);self.groups={}
        if len(self.classes)<4 or len(self.receivers)<2:raise ValueError('Four source classes and two RXs required')
        roles={}
        for i,(label,rx,physical) in enumerate(zip(self.y,self.rx,self.ids)):
            role=(int(label),int(rx))
            if physical in roles and roles[physical]!=role:raise ValueError('Source physical ID changes label/RX')
            roles[physical]=role;self.groups.setdefault(role,{}).setdefault(physical,[]).append(i)
        for label in self.classes:
            for rx in self.receivers:
                group=self.groups.get((int(label),int(rx)),{})
                if len(group)<20:raise ValueError('Source cell has fewer than 20 physical examples')
                self.groups[int(label),int(rx)]=tuple(tuple(rows) for rows in group.values())

    def sample(self,step):
        k=(1,5,20)[step%3];classes=self.rng.permutation(self.classes)[:4];rx0,rx1=self.rng.permutation(self.receivers)[:2]
        def choose(rx,count):
            result=[]
            for label in classes:
                group=self.groups[int(label),int(rx)]
                for index in self.rng.permutation(len(group))[:count]:
                    rows=group[index];result.append(rows[int(self.rng.integers(len(rows)))])
            return np.asarray(result,np.int64)
        support=choose(rx0,k);held=choose(rx1,8)
        return support,held,k,(step//3)%2,(step//6)%2


def learn(raw,restored,labels,receivers,physical_ids,seed,*,steps=STEPS):
    import torch
    import torch.nn.functional as F
    episodes=Episodes(labels,receivers,physical_ids,seed)
    array=np.stack((coordinates(raw),coordinates(restored)))
    if array.shape!=(2,len(episodes.y),8,64) or not np.isfinite(array).all():raise ValueError('Wrong paired CCA training features')
    bank=torch.tensor(array.tolist(),dtype=torch.float64)
    theta=torch.zeros((8,64),dtype=torch.float64,requires_grad=True);optimizer=torch.optim.Adam([theta],lr=.01)
    losses=[]
    for step in range(steps):
        si,qi,k,sp,qp=episodes.sample(step);scale=torch.exp(.5*LOG_BOUND*torch.tanh(theta))
        support=F.normalize(bank[sp,torch.tensor(si.tolist())]*scale,dim=-1)
        held=F.normalize(bank[qp,torch.tensor(qi.tolist())]*scale,dim=-1)
        prototypes=F.normalize(support.reshape(4,k,8,64).mean(1),dim=-1)
        similarity=(held[:,None]*prototypes[None]).sum(-1).mean(-1)
        loss=F.cross_entropy(similarity/.1,torch.arange(4).repeat_interleave(8))+.01*torch.tanh(theta).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source metric loss')
        optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
    gains=np.exp(LOG_BOUND*np.tanh(np.asarray(theta.detach().tolist(),np.float64)))
    summary=dict(steps=steps,train_observations=len(episodes.y),train_physical_ids=len(set(episodes.ids)),source_TX=episodes.classes.tolist(),source_RX=episodes.receivers.tolist(),
        source_support_K=[1,5,20],ways=4,held_per_class=8,cross_RX=True,physical_support_unique=True,processing_pairs=[[0,0],[1,0],[0,1],[1,1]],
        first_loss=losses[0],last_loss=losses[-1],mean_first50=float(np.mean(losses[:50])),mean_last50=float(np.mean(losses[-50:])),gain_min=float(gains.min()),gain_max=float(gains.max()))
    return gains,summary


class EpisodicMetric:
    def __init__(self,cca,gains,training_summary=None):
        if not isinstance(cca,OrbitCCA):raise ValueError('Expected source CCA encoder')
        gains=np.asarray(gains,np.float64)
        if gains.shape!=(8,64) or not np.isfinite(gains).all() or (gains<.25).any() or (gains>4.).any():raise ValueError('Wrong bounded source metric')
        gains.flags.writeable=False;self.cca=cca;self.gains=gains;self.rank=64;self.training_summary=training_summary

    @classmethod
    def fit(cls,raw,restored,*,labels,receivers,physical_ids,seed):
        cca=OrbitCCA.fit(raw,restored)
        mapped=[cca.transform(features,restored=path) for features,path in ((raw,False),(restored,True))]
        gains,summary=learn(*mapped,labels,receivers,physical_ids,seed)
        return cls(cca,gains,summary)

    def transform(self,features,*,restored):
        aligned=self.cca.transform(features,restored=restored);value=coordinates(aligned)*np.sqrt(self.gains)
        norm=np.linalg.norm(value,axis=2,keepdims=True)
        if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid source metric feature norm')
        value=value/norm;values=[]
        for index,d in enumerate(DIMS):
            block=np.zeros((len(value),d),np.float64);block[:,:64]=value[:,index];block.flags.writeable=False;values.append(block)
        out=dict(zip(BLOCKS[:5],values[:5]));out['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
        return out

    def save(self,path):
        if Path(path).exists():raise FileExistsError('Frozen episodic metric exists')
        payload=dict(schema=np.asarray(SCHEMA),gains=self.gains)
        for name,p in zip(BLOCKS,self.cca.parameters):
            for key,a in zip(('mean_raw','mean_restored','raw_projection','restored_projection'),p):payload[name+'_'+key]=a
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!=SCHEMA:raise ValueError('Wrong source metric checkpoint')
            parameters=[tuple(np.array(z[name+'_'+key],dtype=np.float64,copy=True) for key in ('mean_raw','mean_restored','raw_projection','restored_projection')) for name in BLOCKS]
            return cls(OrbitCCA(parameters),np.array(z['gains'],np.float64,copy=True))
