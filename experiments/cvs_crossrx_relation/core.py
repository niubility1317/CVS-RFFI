"""Legal L-only, PID-independent, all-class leave-RX-out relation CE."""
from contextlib import contextmanager
import math
import torch
from torch.nn import functional as F
from experiments.cvs_e20_cross_style.style import _paths

RX=(1,3,4,6,8);DAYS=(1,2,3)

def support(L,li,seed,step):
    if L.get('role') not in (None,'L_s','labeled'):raise ValueError('Relations may only use lawful labeled source L')
    if not all(k in L for k in ('y','rx','day','ids')) or len(L['ids'])!=len(L['y']):raise ValueError('Visible lawful L metadata required')
    if len(set(L['ids']))!=len(L['ids']):raise ValueError('L physical IDs must be unique')
    cells=L.get('_relation_cells')
    if cells is None:
        cells={}
        for tx in range(6):
            for rx in RX:
                for day in DAYS:
                    index=torch.where((L['y']==tx)&(L['rx']==rx)&(L['day']==day))[0]
                    if not len(index):raise ValueError('Missing lawful L TX/RX/day cell')
                    cells[tx,rx,day]=index
        L['_relation_cells']=cells
    query_ids={L['ids'][i] for i in li.tolist()};day=DAYS[((step-448)//4)%3];gen=torch.Generator().manual_seed(seed+step*131071+276444);index=[]
    for tx in range(6):
        for rx in RX:
            eligible=[i for i in cells[tx,rx,day].tolist() if L['ids'][i] not in query_ids]
            if not eligible:raise ValueError('Independent relation support cell exhausted')
            index.append(eligible[int(torch.randint(len(eligible),(1,),generator=gen))])
    si=torch.tensor(index,dtype=torch.long);ids=[L['ids'][i] for i in index]
    if len(ids)!=30 or len(set(ids))!=30 or set(ids)&query_ids:raise ValueError('Support/query physical leakage')
    if L['y'][si].tolist()!=[tx for tx in range(6) for _ in RX] or L['rx'][si].tolist()!=list(RX)*6 or L['day'][si].tolist()!=[day]*30:raise ValueError('Balanced support cell metadata differs')
    return si,dict(support_indices=index,support_IDs=ids,support_day=day,support_query_PID_disjoint=True,L_packets=30,all6_TX=True,source_RX=list(RX),per_TX_RX=1)

@contextmanager
def capture(model,levels,first_only=False):
    values={};handles=[];_,paths=_paths(model.base)
    for level in levels:
        if level=='id_feat_joint':continue
        def hook(module,args,h,key=level):
            if first_only and key in values:return
            if h.ndim!=3 or key in values:raise ValueError('Actual native map capture differs')
            values[key]=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1)
        handles.append(dict(model.base.named_modules())[paths[level]].register_forward_hook(hook))
    try:yield values
    finally:
        for handle in handles:handle.remove()

@torch.no_grad()
def reference(model,x,levels):
    modes=[(m,m.training) for m in model.modules()];before={n:b.detach().clone() for n,b in model.named_buffers()}
    try:
        model.eval()
        with capture(model,levels) as values:out=model.forward_identity_only(x)
        if 'id_feat_joint' in levels:values['id_feat_joint']=out['id_feat_joint']
        if any(v.requires_grad for v in values.values()):raise ValueError('Prototype reference graph is not detached')
        if any(not torch.equal(b,before[n]) for n,b in model.named_buffers()):raise ValueError('Reference eval changed buffer')
        return {k:v.detach() for k,v in values.items()}
    finally:
        for m,mode in modes:m.training=mode

def relation_logits(query,refs,query_rx,mode,tau=.1):
    if query.ndim!=2 or refs.shape!=(30,query.shape[1]) or query_rx.shape!=(len(query),) or mode not in ('merged_RX','separate_RX'):raise ValueError('Relation shape/mode differs')
    if not bool(torch.isfinite(query).all() and torch.isfinite(refs).all()) or tau<=0:raise ValueError('Finite representations and positive temperature required')
    if refs.requires_grad:raise ValueError('Reference prototypes must be detached')
    if not set(query_rx.detach().cpu().tolist()).issubset(set(RX)):raise ValueError('Only registered source RX allowed')
    grid=refs.reshape(6,5,-1);other=torch.tensor([[j for j,r in enumerate(RX) if r!=int(rx)] for rx in query_rx.detach().cpu().tolist()],device=query.device)
    chosen=grid[:,other].permute(1,0,2,3);center=chosen.mean((1,2));q=F.normalize(query-center,dim=1,eps=1e-4);centered=chosen-center[:,None,None,:]
    if mode=='merged_RX':
        prototype=F.normalize(centered.mean(2),dim=-1,eps=1e-4);logits=(q[:,None,:]*prototype).sum(-1)/tau
    else:
        prototype=F.normalize(centered,dim=-1,eps=1e-4);per_RX=(q[:,None,None,:]*prototype).sum(-1)/tau;logits=torch.logsumexp(per_RX,2)-math.log(4.)
    if logits.shape!=(len(query),6):raise ValueError('All six TX must compete')
    return logits

def loss(query,refs,y,rx,mode,tau):
    logits=relation_logits(query,refs,rx,mode,tau);value=F.cross_entropy(logits,y)
    correct=logits.gather(1,y[:,None]).squeeze(1);wrong=logits.masked_fill(F.one_hot(y,6).bool(),float('-inf')).max(1).values
    return value,dict(loss=float(value.detach()),accuracy=float((logits.argmax(1)==y).float().mean()),margin=float((correct-wrong).mean().detach()),predictions=logits.argmax(1).detach(),query_feature_RMS=float(query.square().mean().sqrt().detach()),reference_feature_RMS=float(refs.square().mean().sqrt()),query_low_norm_rate=float((query.detach().norm(dim=1)<=1e-6).float().mean()),query_count=len(y),reference_RX_excludes_query=True,all6_classes_compete=True)
