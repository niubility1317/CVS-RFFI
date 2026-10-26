"""Frozen source-only geometry, RX-separated label propagation and inductive queries."""
import torch


def validate_features(z,rx,allowed):
    if (z.ndim!=2 or len(z)<1 or z.dtype not in (torch.float32,torch.float64) or z.requires_grad
            or rx.shape!=(len(z),) or rx.dtype!=torch.long or rx.device!=z.device or not allowed
            or set(rx.detach().cpu().tolist())-set(allowed) or not bool(torch.isfinite(z).all())
            or bool((z.norm(dim=1)<=1e-8).any())):
        raise ValueError('Finite frozen features with declared source RX only required')


@torch.no_grad()
def geometry(z,rx,allowed,*,k=20,block=512):
    validate_features(z,rx,allowed)
    if type(k) is not int or k<1 or type(block) is not int or block<1:raise ValueError('Positive fixed k and block required')
    z=z/z.norm(dim=1,keepdim=True);n=len(z)
    if len(set(allowed))==1:
        ij=torch.empty((2,0),dtype=torch.long,device=z.device);w=torch.empty(0,dtype=z.dtype,device=z.device)
        return dict(features=z,rx=rx.clone(),allowed=list(allowed),k=k,S=torch.sparse_coo_tensor(ij,w,(n,n)).coalesce(),
            degree=z.new_zeros(n),radius_similarity=z.new_ones(n),directed_cross_RX_edges=0,mutual_edges=0)
    if min(int((rx!=r).sum()) for r in set(rx.cpu().tolist()))<k:raise ValueError('Each source node requires k other-RX records')
    columns=[];scores=[]
    for lo in range(0,n,block):
        sim=z[lo:lo+block]@z.T;sim.masked_fill_(rx[lo:lo+block,None]==rx[None,:],-torch.inf)
        value,index=torch.topk(sim,k,dim=1,sorted=True);columns.append(index);scores.append(value.clamp(-1,1))
    columns=torch.cat(columns);scores=torch.cat(scores);radius=scores[:,-1].clone()
    rows=torch.arange(n,device=z.device).repeat_interleave(k);cols=columns.reshape(-1);value=scores.reshape(-1)
    keys=rows*n+cols;order=torch.argsort(keys);sorted_keys=keys[order];reverse=cols*n+rows
    pos=torch.searchsorted(sorted_keys,reverse);matched=(pos<len(keys))&(sorted_keys[pos.clamp_max(len(keys)-1)]==reverse)
    ii=torch.stack((rows[matched],cols[matched]));weights=((value[matched]-1)/.2).exp()
    W=torch.sparse_coo_tensor(ii,weights,(n,n)).coalesce();W=((W+W.transpose(0,1))*.5).coalesce()
    degree=torch.sparse.sum(W,dim=1).to_dense();ij=W.indices();den=(degree[ij[0]]*degree[ij[1]]).sqrt()
    S=torch.sparse_coo_tensor(ij,W.values()/den,(n,n)).coalesce()
    return dict(features=z,rx=rx.clone(),allowed=list(allowed),k=k,S=S,degree=degree,radius_similarity=radius,
        directed_cross_RX_edges=n*k,mutual_edges=int(S._nnz()))


@torch.no_grad()
def propagate(graph,labels,roles,num_classes=6):
    """roles=0 means visible L,1 hidden U;V/query never enters the graph."""
    z=graph['features'];n=len(z);allowed=graph['allowed'];rx=graph['rx'];r=len(allowed)
    if (labels.shape!=(n,) or labels.dtype!=torch.long or roles.shape!=(n,) or roles.dtype!=torch.long
            or labels.device!=z.device or roles.device!=z.device or num_classes<2
            or bool(((roles!=0)&(roles!=1)).any()) or bool((labels[roles==1]!=-1).any())
            or not bool((roles==0).any()) or bool(((labels[roles==0]<0)|(labels[roles==0]>=num_classes)).any())):
        raise ValueError('Only visible L seeds and hidden-label U nodes;no V/query labels permitted')
    Y=z.new_zeros(n,r*num_classes)
    for s,receiver in enumerate(allowed):
        mask=(roles==0)&(rx==receiver);indices=torch.where(mask)[0]
        Y[indices,s*num_classes+labels[mask]]=1.
    Q=Y.clone();residual=float('inf')
    for iterations in range(1,201):
        updated=.1*Y+.9*torch.sparse.mm(graph['S'],Q);residual=float((updated-Q).abs().max());Q=updated
        if not bool(torch.isfinite(Q).all()):raise FloatingPointError('Nonfinite source propagation')
        if residual<=1e-6:break
    else:raise ValueError('Fixed 200-iteration source diffusion did not converge')
    return dict(Q=Q.reshape(n,r,num_classes),iterations=iterations,residual=residual,classes=num_classes,
        source_L_seeds=int((roles==0).sum()),hidden_U_nodes=int((roles==1).sum()),V_or_target_nodes=0)


@torch.no_grad()
def exclude_own_RX(Q,rx,allowed):
    if (Q.ndim!=3 or Q.shape[1]!=len(allowed) or rx.shape!=(len(Q),) or rx.dtype!=torch.long
            or set(rx.detach().cpu().tolist())-set(allowed) or bool((Q<0).any()) or not bool(torch.isfinite(Q).all())):
        raise ValueError('Finite source-RX decomposed propagation required')
    source=torch.tensor(allowed,device=rx.device,dtype=torch.long);other=source[None,:]!=rx[:,None]
    mass=(Q*other[:,:,None]).sum(1);total=mass.sum(1,keepdim=True);probability=mass/total.clamp_min(torch.finfo(Q.dtype).tiny)
    return probability,other,total.squeeze(1)>0


@torch.no_grad()
def certify(raw_Q,restored_Q,rx,allowed,raw_confidence):
    if raw_Q.shape!=restored_Q.shape or raw_confidence.shape!=(len(rx),) or not bool(torch.isfinite(raw_confidence).all()) or bool(((raw_confidence<0)|(raw_confidence>1)).any()):
        raise ValueError('Aligned two-graph probabilities and frozen raw confidence required')
    rp,other,rv=exclude_own_RX(raw_Q,rx,allowed);gp,_,gv=exclude_own_RX(restored_Q,rx,allowed)
    rc,label=rp.max(1);gc,glabel=gp.max(1);needed=min(3,len(allowed)-1)
    def supports(Q):
        unique=(Q==Q.amax(2,keepdim=True)).sum(2)==1
        return (((Q.argmax(2)==label[:,None])&(Q.sum(2)>0)&unique)&other).sum(1)
    sr=supports(raw_Q);sg=supports(restored_Q)
    accepted=(raw_confidence<.95)&rv&gv&(rc>=.95)&(gc>=.95)&(label==glabel)&(sr>=needed)&(sg>=needed)
    if needed==0:accepted=accepted&False
    return dict(accepted=accepted,label=label,raw_probability=rp,restored_probability=gp,raw_support_RX=sr,restored_support_RX=sg,
        required_support_RX=needed,hard_pool=raw_confidence<.95)


@torch.no_grad()
def extend(graph,propagation,query_features,query_rx,*,block=512):
    """Each source-V query is independent;no labels or persistent updates accepted."""
    validate_features(query_features,query_rx,graph['allowed']);source=graph['features'];rx=graph['rx'];k=graph['k'];Q=propagation['Q']
    query_features=query_features/query_features.norm(dim=1,keepdim=True);output=[]
    if len(graph['allowed'])==1:return Q.new_zeros(len(query_features),Q.shape[1],Q.shape[2])
    for lo in range(0,len(query_features),block):
        z=query_features[lo:lo+block];qr=query_rx[lo:lo+block];sim=z@source.T;sim.masked_fill_(qr[:,None]==rx[None,:],-torch.inf)
        value,index=torch.topk(sim,k,dim=1,sorted=True);value=value.clamp(-1,1)
        valid=(value>=graph['radius_similarity'][index])&(graph['degree'][index]>0)
        weights=((value-1)/.2).exp()*valid/graph['degree'][index].clamp_min(torch.finfo(source.dtype).tiny).sqrt()
        # Query-degree normalization is a common positive scalar and cancels in categorical normalization.
        output.append(.9*(weights[:,:,None,None]*Q[index]).sum(1))
    return torch.cat(output)
