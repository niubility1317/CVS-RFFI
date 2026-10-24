"""Bounded source pseudo-class/RX contribution weights; no labels beyond raw pseudo targets."""
import torch


@torch.no_grad()
def contribution_weights(pseudo,accepted,source_rx):
    if (pseudo.ndim!=1 or accepted.shape!=pseudo.shape or source_rx.shape!=pseudo.shape
        or pseudo.dtype!=torch.long or source_rx.dtype!=torch.long or accepted.dtype!=torch.bool
        or pseudo.device!=accepted.device or pseudo.device!=source_rx.device
        or bool(((pseudo<0)|(pseudo>=6)|(source_rx<0)).any())):
        raise ValueError('Aligned hard rawpseudo/boolean acceptance/sourceRX required')
    weights=torch.ones(len(pseudo),dtype=torch.float64,device=pseudo.device)
    if not bool(accepted.any()):return weights
    y=pseudo[accepted];rx=source_rx[accepted];a=len(y)
    cls,ci,nc=torch.unique(y,return_inverse=True,return_counts=True)
    domains,ri,nr=torch.unique(rx,return_inverse=True,return_counts=True)
    ratio=((a/(len(cls)*nc[ci].double()))*(a/(len(domains)*nr[ri].double()))).sqrt()
    ratio=ratio/ratio.mean();maximum=ratio.max()
    mixing=torch.minimum(maximum.new_tensor(.5),.5/(maximum-1).clamp_min(1e-12))
    weights[accepted]=1+mixing*(ratio-1)
    if (not bool(torch.isfinite(weights).all()) or bool((weights[accepted]<.5-1e-12).any())
        or bool((weights[accepted]>1.5+1e-12).any())
        or abs(float(weights[accepted].sum())-a)>1e-8*max(1,a)):
        raise ValueError('Fixed contribution cap/positive acceptance/mean CE dose differs')
    return weights


def cv(values):
    values=values.double()
    return float(values.std(unbiased=False)/values.mean()) if float(values.mean())>0 else None


@torch.no_grad()
def evaluate(pseudo,accepted,source_rx,order,allowed_rxs):
    n=len(pseudo)
    if (n!=56700 or order.shape!=pseudo.shape or order.dtype!=torch.long or len(torch.unique(order))!=n
        or int(order.min())!=0 or int(order.max())!=n-1 or len(allowed_rxs)!=5
        or set(source_rx.tolist())!=set(allowed_rxs)):
        raise ValueError('Complete fixed fullsource U56700 permutation and five RX required')
    mass=torch.zeros(n,dtype=torch.float64);bounds=[]
    for begin in range(0,n,256):
        ix=order[begin:begin+256];a=accepted[ix];w=contribution_weights(pseudo[ix],a,source_rx[ix])
        mass[ix]=w*a
        bounds.append(dict(samples=len(ix),accepted=int(a.sum()),accepted_weight_sum=float((w*a).sum()),
            accepted_weight_min=float(w[a].min()) if bool(a.any()) else None,
            accepted_weight_max=float(w[a].max()) if bool(a.any()) else None))
    result={}
    for dimension,values,groups in (('pseudo_class',pseudo,range(6)),('source_RX',source_rx,allowed_rxs)):
        rows=[]
        for g in groups:
            m=values==g
            rows.append(dict(value=int(g),seen=int(m.sum()),accepted=int((m&accepted).sum()),weighted_mass=float(mass[m].sum())))
        before=cv(torch.tensor([r['accepted'] for r in rows]));after=cv(torch.tensor([r['weighted_mass'] for r in rows],dtype=torch.float64))
        reduction=(before-after)/before if before is not None and before>1e-12 else None
        result[dimension]=dict(rows=rows,accepted_mass_cv=before,weighted_mass_cv=after,relative_cv_reduction=reduction,
            minimum_accepted=min(r['accepted'] for r in rows),passes=min(r['accepted'] for r in rows)>=1000 and reduction is not None and reduction>=.1)
    return dict(samples=n,accepted=int(accepted.sum()),weighted_mass=float(mass.sum()),batch_size=256,batches=len(bounds),
        final_batch_samples=bounds[-1]['samples'],batch_weight_bounds=bounds,dimensions=result,
        passes=all(v['passes'] for v in result.values()),raw_labels_or_acceptance_changed=False,U_truth_reads=0,target_reads=0,model_updates=0)
