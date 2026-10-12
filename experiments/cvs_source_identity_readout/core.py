"""PSD representation scatters and L-crossfit cross-RX prototype readouts."""
import math
import torch
from experiments.cvs_e20_cross_style.style import _paths

@torch.no_grad()
def extract(model,L,device,batch_size=128,progress=None):
    model.eval();features={k:[] for k in ('time_down','t2.pool','id_feat_joint')};logits=[];calls=0
    _,paths=_paths(model.base)
    for begin in range(0,len(L['ids']),batch_size):
        values={};handles=[]
        for level in ('time_down','t2.pool'):
            def hook(module,args,h,key=level):
                if key in values or h.ndim!=3:raise ValueError('Native representation map differs')
                values[key]=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),dim=1)
            handles.append(dict(model.base.named_modules())[paths[level]].register_forward_hook(hook))
        try:out=model.forward_identity_only(L['x'][begin:begin+batch_size].to(device))
        finally:
            for handle in handles:handle.remove()
        values['id_feat_joint']=out['id_feat_joint'];logits.append(out['tx_logits'].detach().cpu())
        for level in features:
            assert torch.isfinite(values[level]).all();features[level].append(values[level].detach().cpu())
        calls+=1
        if progress:progress(dict(batches=calls,packets=min(begin+batch_size,len(L['ids']))))
    return {k:torch.cat(v).double() for k,v in features.items()},torch.cat(logits),dict(model_forwards=calls,L_occurrences=len(L['ids']),first_order_derivative_calls=0)

def cells(L):
    y,rx,day=(L[k].cpu() for k in ('y','rx','day'));groups={}
    for t in sorted(y.unique().tolist()):
        for r in sorted(rx.unique().tolist()):
            for v in sorted(day.unique().tolist()):
                index=torch.where((y==t)&(rx==r)&(day==v))[0]
                if not len(index):raise ValueError('Missing TX/RX/day source cell')
                groups[t,r,v]=index
    return groups

def scatters(z,L,selection=None):
    if z.ndim!=2 or not torch.isfinite(z).all():raise ValueError('Finite representation matrix required')
    z=z.double();groups=cells(L);Y=sorted(L['y'].unique().tolist());R=sorted(L['rx'].unique().tolist());D=sorted(L['day'].unique().tolist())
    mask=torch.ones(len(z),dtype=torch.bool) if selection is None else selection
    # An excluded RX is absent from the fitted statistic, never zero-filled.
    R=[r for r in R if any(bool(mask[idx].any()) for (t,v,day),idx in groups.items() if v==r)]
    n=z.shape[1];grid=[];cell_values=[]
    for t in Y:
        row=[]
        for r in R:
            values=[]
            for day in D:
                idx=groups[t,r,day];idx=idx[mask[idx]]
                if not len(idx):raise ValueError('Crossfit removed an entire fitted cell')
                values.append(z[idx])
            row.append(torch.stack([v.mean(0) for v in values]).mean(0));cell_values.append(values)
        grid.append(torch.stack(row))
    grid=torch.stack(grid);grand=grid.mean((0,1));tx=grid.mean(1)-grand;rx=grid.mean(0)-grand
    interaction=grid-grid.mean(1,keepdim=True)-grid.mean(0,keepdim=True)+grand;b=interaction.reshape(-1,n)
    within=[];total=[]
    for i,values in enumerate(cell_values):
        mean=grid.reshape(-1,n)[i]
        within.append(torch.stack([(v-mean).T@(v-mean)/len(v) for v in values]).mean(0))
        total.append(torch.stack([(v-grand).T@(v-grand)/len(v) for v in values]).mean(0))
    out=dict(TX=tx.T@tx/len(Y),RX=rx.T@rx/len(R),interaction=b.T@b/(len(Y)*len(R)),within=torch.stack(within).mean(0),total=torch.stack(total).mean(0))
    residual=(out['total']-sum(out[k] for k in ('TX','RX','interaction','within'))).abs().max()
    scale=out['total'].abs().max().clamp_min(1e-12)
    if residual/scale>1e-8:raise ValueError('Representation scatter decomposition does not close')
    return out,grand,dict(relative_closure_error=float(residual/scale),balanced_cell_weights=True,days=len(D),fit_receivers=R,dimensions=n,identity_mean_rank=int(torch.linalg.matrix_rank(out['TX'],rtol=1e-8)),within_includes_day_variation=True)

def stable_projection(s,policy):
    n=s['TX'].shape[0];eye=torch.eye(n,dtype=torch.float64)
    denominator=s['within']+policy['alpha']*s['RX']+policy['beta']*s['interaction']
    scale=(denominator.trace()/n).clamp_min(1e-12);gamma=policy['shrinkage']
    denominator=(1-gamma)*denominator+gamma*scale*eye+policy['eps_relative']*scale*eye
    chol=torch.linalg.cholesky(denominator)
    left=torch.linalg.solve_triangular(chol,s['TX'],upper=False)
    reduced=torch.linalg.solve_triangular(chol,left.T,upper=False).T
    reduced=(reduced+reduced.T)/2;eigen,U=torch.linalg.eigh(reduced);order=eigen.argsort(descending=True)
    eigen=eigen[order];U=U[:,order];threshold=max(float(eigen.max()),1e-12)*1e-8
    rank=min(int((eigen>threshold).sum()),policy['max_rank']);rank=max(rank,0)
    W=torch.linalg.solve_triangular(chol.T,U[:,:rank],upper=True)
    residual=(s['TX']@W-(denominator@W)*eigen[:rank]).norm()/(s['TX']@W).norm().clamp_min(1e-12)
    if rank and residual>1e-7:raise ValueError('Generalized eigen residual too large')
    return W,dict(rank=rank,max_identity_rank=5,eigenvalues=eigen.tolist(),direction_component_energy={k:torch.diag(W.T@v@W).tolist() for k,v in s.items()},denominator_min_eigenvalue=float(torch.linalg.eigvalsh(denominator).min()),denominator_trace=float(denominator.trace()),generalized_eigen_relative_residual=float(residual))

def folds(L,seed,nfold=3):
    result=torch.full((len(L['ids']),),-1,dtype=torch.long);gen=torch.Generator().manual_seed(seed)
    for key,idx in sorted(cells(L).items()):
        ordered=sorted(idx.tolist(),key=lambda i:L['ids'][i]);order=torch.randperm(len(ordered),generator=gen)
        for j,i in enumerate(order.tolist()):result[ordered[i]]=j%nfold
    assert bool((result>=0).all());return result

def scores(z,L,fit,query,center,W,tau,separate):
    Z=z-center
    if W is not None:Z=Z@W
    if Z.shape[1]==0:return torch.zeros(int(query.sum()),6,dtype=torch.float64)
    q=torch.nn.functional.normalize(Z[query],dim=1,eps=1e-12);per=[]
    rxs=sorted(L['rx'][fit].unique().tolist())
    for tx in range(6):
        prototypes=[]
        for rx in (rxs if separate else [None]):
            mask=fit&(L['y']==tx)
            if rx is not None:mask=mask&(L['rx']==rx)
            # Equal RX/day cell means prevent count imbalance from determining
            # merged reference weights after physical-ID cross fitting.
            centroids=[]
            for receiver in (rxs if rx is None else [rx]):
                for day in sorted(L['day'].unique().tolist()):centroids.append(Z[mask&(L['rx']==receiver)&(L['day']==day)].mean(0))
            prototype=torch.stack(centroids).mean(0);prototypes.append(torch.nn.functional.normalize(prototype,dim=0,eps=1e-12))
        logits=q@torch.stack(prototypes).T/tau
        per.append(torch.logsumexp(logits,dim=1)-math.log(len(prototypes)))
    return torch.stack(per,dim=1)

def metrics(score,y):
    prediction=score.argmax(1);cm=torch.bincount(6*y+prediction,minlength=36).reshape(6,6);n=len(y)
    true=score.gather(1,y[:,None]).squeeze(1);wrong=score.masked_fill(torch.nn.functional.one_hot(y,6).bool(),float('-inf')).max(1).values
    return dict(count=n,accuracy=float((prediction==y).double().mean()),confusion=cm.tolist(),true_vs_max_other_margin=float((true-wrong).mean()),NLL=float(-score.log_softmax(1).gather(1,y[:,None]).mean()))

def summarize(features,logits,L,c):
    fold=folds(L,c['crossfit_seed'],c['crossfit_folds']);records=[];geometry=[];fitted=[]
    for level,z in features.items():
        s,_,proof=scatters(z,L);total=float(s['total'].trace());W,e=stable_projection(s,c['stable_projection'])
        geometry.append(dict(layer=level,**proof,full_L_projection_diagnostic=e,scatter_traces={k:float(v.trace()) for k,v in s.items()},energy_fractions={k:float(s[k].trace())/total if total else None for k in ('TX','RX','interaction','within')},scope='actual representation; fullL geometry descriptive; these fullL directions never used to score crossfit queries'))
        outputs={name:torch.zeros(len(z),6,dtype=torch.float64) for name in c['readouts']};coverage=torch.zeros(len(z),dtype=torch.long)
        for held_rx in c['receivers']:
            for k in range(c['crossfit_folds']):
                query=(L['rx']==held_rx)&(fold==k);fit=(L['rx']!=held_rx)&(fold!=k)
                assert not bool((fit&query).any()) and not set(i for i,m in enumerate(fit) if m)&set(i for i,m in enumerate(query) if m)
                stats,center,closure=scatters(z,L,fit);proj,eig=stable_projection(stats,c['stable_projection'])
                coverage[query]+=1
                fitted.append(dict(layer=level,held_rx=held_rx,fold=k,fit_count=int(fit.sum()),query_count=int(query.sum()),fit_query_disjoint=True,held_RX_excluded_from_statistics_and_prototypes=True,**eig))
                for name in c['readouts']:
                    score=scores(z,L,fit,query,center,proj if name.startswith('stable5') else None,c['temperature'],name.endswith('RX_separate'))
                    outputs[name][query]=score
                    records.append(dict(layer=level,readout=name,scope='held_RX_readout_fold',receiver=held_rx,fold=k,**metrics(score,L['y'][query])))
        assert bool((coverage==1).all())
        for name,score in outputs.items():
            records.append(dict(layer=level,readout=name,scope='overall_crossfit',receiver=None,fold=None,**metrics(score,L['y'])))
    records.append(dict(layer='native_identity_head',readout='native_frozen_head',scope='seen_L_in_sample',receiver=None,fold=None,**metrics(logits.double(),L['y'])))
    return dict(geometry=geometry,records=records,crossfit=fitted,query_coverage_per_method=len(L['ids']),feature_export=False,all_six_TX_uniform_rule=True,source_only=True,target_access=False)
