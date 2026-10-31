"""Bounded source-L cross-RX contrastive residual on correlation-weighted CCA."""
import numpy as np
from .correlation_cca import CorrelationCCA
from .orbit_cca import blocks
from .source_bridge import BLOCKS,DIMS
from .episodic_metric import Episodes,coordinates

STEPS=600
BOUND=.25
TEMPERATURE=.1


def masks(labels,receivers):
    same=labels[:,None]==labels[None,:]
    positive=same&(receivers[:,None]!=receivers[None,:])
    allowed=positive|~same
    if not bool(positive.any(1).all()):raise ValueError('Every source anchor needs a cross-RX positive')
    return positive,allowed


def contrastive_loss(embedding,labels,receivers):
    import torch
    positive,allowed=masks(labels,receivers)
    logits=embedding@embedding.T/TEMPERATURE
    normalizer=torch.logsumexp(logits.masked_fill(~allowed,float('-inf')),dim=1)
    return -((logits-normalizer[:,None])*positive).sum(1).div(positive.sum(1)).mean()


def networks(device):
    from torch import nn
    result=nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(2*d,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh()) for d in DIMS]) for _ in range(2)]).to(device)
    for path in result:
        for branch in path:nn.init.zeros_(branch[2].weight);nn.init.zeros_(branch[2].bias)
    return result


def export(nets):
    return tuple(tuple(tuple(np.asarray(value.detach().cpu().tolist(),np.float32) for value in (net[0].weight,net[0].bias,net[2].weight,net[2].bias)) for net in path) for path in nets)


def learn(cca,raw,restored,labels,receivers,physical_ids,seed,*,steps=STEPS,device='cuda:0'):
    import torch
    import torch.nn.functional as F
    episodes=Episodes(labels,receivers,physical_ids,seed)
    torch.set_num_threads(1);torch.manual_seed(seed)
    def tensor(value):return torch.tensor(np.asarray(value,np.float32).tolist(),dtype=torch.float32,device=device)
    x=[[tensor(value) for value in blocks(f)] for f in (raw,restored)]
    h=[tensor(coordinates(cca.transform(f,restored=bool(path)))) for path,f in enumerate((raw,restored))]
    if any(len(value)!=len(episodes.y) for path in x for value in path):raise ValueError('Source contrastive feature/metadata alignment differs')
    nets=networks(device);optimizer=torch.optim.AdamW(nets.parameters(),lr=.001,weight_decay=.0001)
    losses=[];contrasts=[];pairs=[];anchors=[];exposures=0
    for step in range(steps):
        first,second,_,_,_=episodes.sample(step);index=np.r_[first,second]
        ti=torch.tensor(index.tolist(),dtype=torch.long,device=device);outputs=[]
        for path in (0,1):
            outputs.append(torch.stack([F.normalize(h[path][ti,i]+BOUND*net(value[ti])/8,dim=1) for i,(value,net) in enumerate(zip(x[path],nets[path]))],dim=1))
        embedding=torch.cat(outputs).reshape(2*len(index),-1)/np.sqrt(8)
        label=torch.tensor(episodes.y[index].tolist(),dtype=torch.long,device=device).repeat(2)
        rx=torch.tensor(episodes.rx[index].tolist(),dtype=torch.long,device=device).repeat(2)
        contrast=contrastive_loss(embedding,label,rx)
        pair=(outputs[0]-outputs[1]).square().sum(-1).mean()
        anchor=torch.stack([(value-h[path][ti]).square().sum(-1).mean() for path,value in enumerate(outputs)]).mean()
        loss=contrast+pair+.1*anchor
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite source contrastive residual loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        losses.append(float(loss.detach().item()));contrasts.append(float(contrast.detach().item()));pairs.append(float(pair.detach().item()));anchors.append(float(anchor.detach().item()));exposures+=len(index)
        if (step+1)%100==0:print('SOURCE_CROSSRX_RESIDUAL_STEP',step+1,losses[-1],flush=True)
    width=min(50,steps)
    summary=dict(steps=steps,seed=seed,lr=.001,weight_decay=.0001,temperature=TEMPERATURE,pair_weight=1.,anchor_weight=.1,residual_bound=BOUND,
        train_observations=len(episodes.y),train_physical_ids=len(set(episodes.ids)),source_TX=episodes.classes.tolist(),source_RX=episodes.receivers.tolist(),
        source_support_K=[1,5,20],other_RX_per_class=8,ways=4,physical_records_per_step=[36,52,112],physical_unique_within_step=True,
        paired_exposures=exposures,processing_views=2,positive='same_TX_different_RX',excluded='same_TX_same_RX',
        mean_first50=float(np.mean(losses[:width])),mean_last50=float(np.mean(losses[-width:])),last_contrast=contrasts[-1],last_pair=pairs[-1],last_anchor=anchors[-1],
        network_parameters=sum(p.numel() for p in nets.parameters()),source_L_label_rows=len(episodes.y),target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0)
    return export(nets),summary


class CrossRXResidualCCA(CorrelationCCA):
    def __init__(self,parameters,neural,training_summary=None,correlation_summary=None):
        super().__init__(parameters);self.neural=tuple(tuple(tuple(p) for p in path) for path in neural)
        if len(self.neural)!=2 or any(len(path)!=8 for path in self.neural):raise ValueError('Wrong source residual neural schema')
        for path in self.neural:
            for d,p in zip(DIMS,path):
                if len(p)!=4 or tuple(a.shape for a in p)!=((64,2*d),(64,),(64,64),(64,)):raise ValueError('Wrong source residual parameter dimensions')
                for a in p:
                    if a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid source residual parameters')
                    a.flags.writeable=False
        self.training_summary=training_summary;self.correlation_summary=correlation_summary

    @classmethod
    def fit(cls,raw,restored,*,labels,receivers,physical_ids,seed,steps=STEPS,device='cuda:0'):
        cca=CorrelationCCA.fit(raw,restored);neural,summary=learn(cca,raw,restored,labels,receivers,physical_ids,seed,steps=steps,device=device)
        return cls(cca.parameters,neural,summary,cca.correlation_summary)

    def residuals(self,features,*,restored):
        if not isinstance(restored,bool):raise ValueError('Explicit raw/restored path required')
        return [BOUND*np.tanh(np.tanh(x@p[0].T+p[1])@p[2].T+p[3])/8 for x,p in zip(blocks(features),self.neural[int(restored)])]

    def transform(self,features,*,restored):
        aligned=super().transform(features,restored=restored);base=coordinates(aligned);values=[]
        for i,(delta,d) in enumerate(zip(self.residuals(features,restored=restored),DIMS)):
            changed=base[:,i]+delta;norm=np.linalg.norm(changed,axis=1,keepdims=True)
            if not np.isfinite(norm).all() or (norm<.74999).any():raise FloatingPointError('Invalid source residual norm')
            out=np.zeros((len(changed),d),np.float64);out[:,:64]=changed/norm;out.flags.writeable=False;values.append(out)
        result=dict(zip(BLOCKS[:5],values[:5]));result['sig_views']=np.concatenate(values[5:],axis=1)/np.sqrt(3);return result
