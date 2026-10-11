"""Symmetric L+U graph with hard L anchors; query prediction never fits state."""
import torch
from torch.nn import functional as F

def probability(q):
    if q.ndim!=2 or not torch.isfinite(q).all() or (q<0).any() or not torch.allclose(q.sum(1),torch.ones(len(q),device=q.device),atol=1e-5):
        raise ValueError('Invalid probability matrix')

@torch.no_grad()
def neighbors(query,bank,k=32,exclude_self=False,chunk=512):
    if query.ndim!=2 or bank.ndim!=2 or query.shape[1]!=bank.shape[1] or not torch.isfinite(query).all() or not torch.isfinite(bank).all():
        raise ValueError('Malformed frozen feature matrix')
    if not 0<k<len(bank) or (exclude_self and len(query)!=len(bank)):raise ValueError('Invalid graph neighbors')
    q=F.normalize(query,dim=1);b=F.normalize(bank,dim=1);indices=[];similarities=[]
    for start in range(0,len(q),chunk):
        value=q[start:start+chunk]@b.T
        if exclude_self:
            local=torch.arange(len(value),device=q.device);value[local,local+start]=-torch.inf
        sim,idx=value.topk(k,dim=1,sorted=True);indices.append(idx);similarities.append(sim)
    return torch.cat(indices),torch.cat(similarities)

def weights(similarity,temperature):
    if temperature<=0:raise ValueError('Positive kernel temperature required')
    return torch.exp((similarity-1)/temperature).clamp_min(1e-12)

def components(index,labeled_count):
    # Exact undirected connectivity, using only indices; no U labels available.
    n=len(index);parents=list(range(n));size=[1]*n
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,js in enumerate(index.cpu().tolist()):
        for j in js:
            a,b=find(i),find(j)
            if a!=b:
                if size[a]<size[b]:a,b=b,a
                parents[b]=a;size[a]+=size[b]
    roots=[find(i) for i in range(n)];anchored=set(roots[:labeled_count]);counts={}
    for r in roots:counts[r]=counts.get(r,0)+1
    return dict(component_count=len(counts),largest_component=max(counts.values()),
        components_without_L=sum(r not in anchored for r in counts),
        U_nodes_without_L_component=sum(r not in anchored for r in roots[labeled_count:]))

@torch.no_grad()
def fit(L_feature,L_y,U_feature,U_q,settings):
    if L_y.shape!=(len(L_feature),) or L_y.dtype!=torch.long:raise ValueError('Only L supplies integer TX anchors')
    if U_q.shape!=(len(U_feature),6) or set(L_y.cpu().tolist())!=set(range(6)):raise ValueError('Registered six-class source graph required')
    probability(U_q)
    bank=torch.cat((L_feature,U_feature));n=len(bank);nl=len(L_feature)
    idx,sim=neighbors(bank,bank,settings['k'],exclude_self=True)
    src=torch.arange(n,device=bank.device)[:,None].expand_as(idx).reshape(-1);dst=idx.reshape(-1)
    w=weights(sim,settings['temperature']).reshape(-1)/2
    edge=torch.stack((torch.cat((src,dst)),torch.cat((dst,src))))
    W=torch.sparse_coo_tensor(edge,torch.cat((w,w)),(n,n)).coalesce()
    degree=torch.sparse.sum(W,dim=1).to_dense();W=torch.sparse_coo_tensor(W.indices(),W.values()/degree.mean(),W.shape).coalesce()
    degree=torch.sparse.sum(W,dim=1).to_dense()
    anchor=F.one_hot(L_y,6).to(bank.dtype);q=torch.cat((anchor,U_q));value=q.clone()
    restart=torch.cat((torch.zeros(nl,device=bank.device),U_q.max(1).values))*settings['restart']
    delta=None
    for iteration in range(settings['max_iterations']):
        updated=(torch.sparse.mm(W,value)+restart[:,None]*q)/(degree+restart).clamp_min(1e-12)[:,None]
        updated[:nl]=anchor
        delta=float((updated-value).abs().max());value=updated
        if delta<=settings['tolerance']:break
    probability(value)
    if not torch.equal(value[:nl],anchor):raise AssertionError('L anchors changed')
    diagnostics=dict(iterations=iteration+1,max_change=delta,converged=delta<=settings['tolerance'],
        nodes=n,L_nodes=nl,U_nodes=n-nl,edges=W._nnz(),degree_min=float(degree.min()),degree_max=float(degree.max()),
        **components(idx,nl),V_in_fit=False,U_truth_access=False)
    return dict(bank=bank,distribution=value,L_feature=L_feature,L_y=L_y,index=idx,settings=settings,diagnostics=diagnostics)

@torch.no_grad()
def predict(state,feature,base_q):
    probability(base_q)
    k=state['settings']['k'];t=state['settings']['temperature']
    idx,sim=neighbors(feature,state['bank'],k);w=weights(sim,t);w=w/w.sum(1,keepdim=True)
    graph=(state['distribution'][idx]*w[:,:,None]).sum(1)
    li,ls=neighbors(feature,state['L_feature'],k);lw=weights(ls,t);lw=lw/lw.sum(1,keepdim=True)
    anchor=(F.one_hot(state['L_y'][li],6).to(lw.dtype)*lw[:,:,None]).sum(1)
    values=dict(model=base_q,L_anchor=anchor,model_L_anchor=.5*(base_q+anchor),graph=graph,model_graph=.5*(base_q+graph))
    for q in values.values():probability(q)
    return values

@torch.no_grad()
def relation_diagnostics(state,rx):
    idx=state['index'];nl=len(state['L_feature']);y=state['L_y'];ll=idx[:nl]<nl
    good=(y[idx[:nl].clamp_max(nl-1)]==y[:,None])&ll
    return dict(same_RX_neighbor_fraction=float((rx[idx]==rx[:,None]).float().mean()),
        L_L_edge_count=int(ll.sum()),L_L_TX_agreement=float(good.sum()/ll.sum()) if ll.any() else None,
        L_nodes_without_direct_L_neighbor=int((~ll.any(1)).sum()),
        U_true_neighbor_accuracy=None,reason='U TX truth remains hidden; same RX is not a TX label')
