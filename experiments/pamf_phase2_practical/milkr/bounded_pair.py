"""Source-trained bounded nonlinear residuals on the frozen orbit CCA space.

This module consumes paired source features only during train(). Runtime loads
encoder parameters and transforms each record independently; no source replay.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .orbit_cca import OrbitCCA,blocks
from .source_bridge import BLOCKS,DIMS


BOUND=.25


def tensor(value,device):
    return torch.tensor(np.asarray(value,np.float32).tolist(),dtype=torch.float32,device=device)


def latent(cca,features,restored):
    result=[]
    for value,(mx,my,px,py) in zip(blocks(features),cca.parameters):
        projected=(value-(my if restored else mx))@(py if restored else px)
        norm=np.linalg.norm(projected,axis=1,keepdims=True)
        if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid paired latent norm')
        result.append(projected/norm)
    return result


class BoundedPairEncoder:
    def __init__(self,cca,device='cpu'):
        if not isinstance(cca,OrbitCCA):raise ValueError('Frozen orbit CCA64 required')
        self.cca=cca;self.device=device;self.rank=64
        self.networks=nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(2*d,64),nn.GELU(),nn.Linear(64,64),nn.Tanh()) for d in DIMS]) for _ in range(2)]).to(device)
        for path in self.networks:
            for branch in path:nn.init.zeros_(branch[2].weight);nn.init.zeros_(branch[2].bias)
        self.freeze()

    def freeze(self):
        self.networks.eval()
        for p in self.networks.parameters():p.requires_grad_(False)

    def residuals(self,features,*,restored,batch=512):
        if not isinstance(restored,bool):raise ValueError('Explicit raw/restored path required')
        values=blocks(features);outputs=[[] for _ in values]
        with torch.inference_mode():
            for begin in range(0,len(values[0]),batch):
                for index,(value,net) in enumerate(zip(values,self.networks[int(restored)])):
                    delta=BOUND*net(tensor(value[begin:begin+batch],self.device))/8
                    outputs[index].append(np.asarray(delta.cpu().tolist(),np.float64))
        return [np.concatenate(value) for value in outputs]

    def transform(self,features,*,restored,batch=512):
        if not isinstance(restored,bool):raise ValueError('Explicit raw/restored path required')
        original=latent(self.cca,features,restored);delta=self.residuals(features,restored=restored,batch=batch);values=[]
        for base,residual,d in zip(original,delta,DIMS):
            projected=base+residual;norm=np.linalg.norm(projected,axis=1,keepdims=True)
            if not np.isfinite(norm).all() or (norm<.74999).any():raise FloatingPointError('Invalid bounded residual norm')
            value=np.zeros((len(base),d),np.float64);value[:,:64]=projected/norm;value.flags.writeable=False;values.append(value)
        out=dict(zip(BLOCKS[:5],values[:5]));out['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3)
        return out

    def save(self,path):
        if Path(path).exists():raise FileExistsError('Frozen bounded encoder exists')
        payload=dict(schema=np.asarray('source_bounded_orbit_pair_v1'),rank=np.asarray(64,np.int64),bound=np.asarray(BOUND,np.float64))
        for name,p in zip(BLOCKS,self.cca.parameters):
            for key,value in zip(('mean_raw','mean_restored','raw_projection','restored_projection'),p):payload[name+'_'+key]=value
        for name,value in self.networks.state_dict().items():payload['network_'+name]=np.asarray(value.cpu().tolist(),np.float32)
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path,device='cpu'):
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!='source_bounded_orbit_pair_v1' or int(z['rank'].item())!=64 or float(z['bound'].item())!=BOUND:raise ValueError('Wrong bounded encoder schema')
            params=[tuple(np.array(z[name+'_'+key],dtype=np.float64,copy=True) for key in ('mean_raw','mean_restored','raw_projection','restored_projection')) for name in BLOCKS]
            model=cls(OrbitCCA(params),device)
            state={name:tensor(z['network_'+name],device) for name in model.networks.state_dict()}
            if not all(bool(torch.isfinite(value).all()) for value in state.values()):raise ValueError('Invalid paired neural parameters')
            model.networks.load_state_dict(state,strict=True);model.freeze();return model


def train(cca,raw,restored,*,device,steps=600,seed=2026103101,batch=256):
    """No labels, query handles, F/G models or classification heads accepted."""
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=BoundedPairEncoder(cca,device);X=[[tensor(value,device) for value in blocks(f)] for f in (raw,restored)]
    H=[[tensor(value,device) for value in latent(cca,f,bool(path))] for path,f in enumerate((raw,restored))]
    if len(X[0][0])!=len(X[1][0]) or len(X[0][0])<2:raise ValueError('Paired source alignment differs')
    def outputs(path,indices):
        return [nn.functional.normalize(base[indices]+BOUND*net(value[indices])/8,dim=1) for value,base,net in zip(X[path],H[path],model.networks[path])]
    def alignment(indices):
        a=outputs(0,indices);b=outputs(1,indices)
        pair=torch.stack([((u-v)**2).sum(1).mean() for u,v in zip(a,b)]).mean()
        anchor=torch.stack([((value-base[indices])**2).sum(1).mean() for values,bases in ((a,H[0]),(b,H[1])) for value,base in zip(values,bases)]).mean()
        return pair,anchor
    with torch.no_grad():initial=float(alignment(slice(None))[0].item())
    model.networks.train()
    for parameter in model.networks.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.networks.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator(device=device).manual_seed(seed)
    for step in range(steps):
        indices=torch.randint(len(X[0][0]),(batch,),device=device,generator=generator)
        pair,anchor=alignment(indices);loss=pair+.1*anchor
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite source pair loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    model.freeze()
    with torch.no_grad():final=float(alignment(slice(None))[0].item())
    audit=dict(initial_alignment_loss=initial,final_alignment_loss=final,optimizer_updates=steps,batch=batch,seed=seed,
        lr=.001,weight_decay=.0001,anchor_weight=.1,residual_bound=BOUND,physical_pair_rows=len(X[0][0]),TX_label_reads=0,query_reads=0,F_updates=0,G_updates=0,
        network_parameters=sum(p.numel() for p in model.networks.parameters()),paired_exposures=steps*batch)
    return model,audit
