"""Finite-L, class-stratified Fishr moment diagnostics; no network updates."""
import math
import statistics
import torch
from torch.nn import functional as F

MS=(2,8,16)
ROUNDS=12


def sample_gradient(features,weight,bias,labels,a=None,b=None):
    value=features if a is None else features*(1+a)+b
    probability=F.linear(value,weight.detach(),bias.detach()).softmax(1)
    error=probability-F.one_hot(labels,6).to(value)
    augmented=torch.cat((value,torch.ones_like(value[:,:1])),1)
    return (error[:,:,None]*augmented[:,None,:]).flatten(1)


def exact_variances(gradients,labels,receivers):
    """Each receiver: uniform mixture over six finite class populations."""
    output=[]
    for rx in receivers.unique(sorted=True):
        cells=[gradients[(receivers==rx)&(labels==k)] for k in range(6)]
        if any(len(v)==0 for v in cells):raise ValueError('Missing registered class/RX cell')
        means=torch.stack([v.mean(0) for v in cells]);overall=means.mean(0)
        output.append(torch.stack([v.var(0,unbiased=False) for v in cells]).mean(0)+(means-overall).square().mean(0))
    return torch.stack(output)


def pooled_variances(gradients,receivers):
    return torch.stack([gradients[receivers==rx].var(0,unbiased=False) for rx in receivers.unique(sorted=True)])


def unbiased_variances(gradients,labels,receivers,cell_sizes):
    """Unbiased uniform-class finite-population variance with SRSWOR per cell.

    s_k^2 uses m-1, Bhat = Var_k(class sample means), denominator K.
    Vhat = Bhat + mean_k[(1-(K-1)/(K*m)-1/(K*N_k))*s_k^2].
    This corrects fixed class counts; pooled n-1 alone is not unbiased.
    """
    values=[];K=6
    for rx in receivers.unique(sorted=True).tolist():
        cells=[gradients[(receivers==rx)&(labels==k)] for k in range(K)]
        counts=[len(v) for v in cells]
        if len(set(counts))!=1 or counts[0]<2:raise ValueError('Equal class quota m>=2 required')
        m=counts[0];means=torch.stack([v.mean(0) for v in cells]);within=[]
        for k,cell in enumerate(cells):
            n=cell_sizes[(rx,k)]
            if m>n:raise ValueError('Sample quota exceeds finite cell population')
            coefficient=1-(K-1)/(K*m)-1/(K*n)
            within.append(coefficient*cell.var(0,unbiased=True))
        values.append((means-means.mean(0)).square().mean(0)+torch.stack(within).mean(0))
    return torch.stack(values)


def penalty(a,b=None):
    left=a-a.mean(0);right=left if b is None else b-b.mean(0)
    return (left*right).mean()


def _gradient(loss,a,b):
    result=torch.autograd.grad(loss,(a,b),retain_graph=True,allow_unused=False)
    return torch.cat([v.detach().flatten() for v in result])


def _direction(g,reference):
    norm=float(g.norm());ref=float(reference.norm());dot=float(g.dot(reference))
    return dict(norm=norm,reference_norm=ref,dot=dot,cosine=dot/(norm*ref) if norm>0 and ref>0 else None,
        norm_ratio=norm/ref if ref>0 else None,negative_alignment=dot<0 if norm>0 and ref>0 else None)


def _stats(values):
    values=[float(v) for v in values if v is not None]
    if not values:return dict(count=0,mean=None,std=None,min=None,max=None)
    return dict(count=len(values),mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else None,
        min=min(values),max=max(values),negative_count=sum(v<0 for v in values))


def _draw(cells,m,generator):
    return torch.cat([pool[torch.randperm(len(pool),generator=generator)[:m]] for pool in cells.values()])


def fishr_audit(features,head,labels,receivers,days,ids,seed):
    """L-only input; fixed m=2/8/16, 12 A/B rounds per day and RX control."""
    if features.ndim!=2 or not features.is_floating_point() or not torch.isfinite(features).all():raise ValueError('Finite pooled features [N,C] required')
    n,c=features.shape
    if len(ids)!=n or len(set(ids))!=n:raise ValueError('Unique L physical IDs required')
    for value in (labels,receivers,days):
        if value.shape!=(n,) or value.dtype!=torch.long:raise ValueError('Integer source metadata required')
    if set(labels.tolist())!=set(range(6)):raise ValueError('All six L classes required')
    if head.bias is None or head.in_features!=c or head.out_features!=6:raise ValueError('Fixed six-class linear readout required')
    saved_head={k:v.detach().clone() for k,v in head.state_dict().items()}
    saved_inputs=[value.detach().clone() for value in (features,labels,receivers,days)]
    # All CPU double arithmetic is diagnostic; no model graph or real head grad.
    x=features.detach().cpu().double().clone();y=labels.detach().cpu().clone();rx=receivers.detach().cpu().clone();day=days.detach().cpu().clone()
    weight=head.weight.detach().cpu().double().clone();bias=head.bias.detach().cpu().double().clone()
    a=torch.zeros(c,dtype=torch.float64,requires_grad=True);b=torch.zeros_like(a,requires_grad=True)
    probe=F.cross_entropy(F.linear(x*(1+a)+b,weight,bias),y);probe_gradient=_gradient(probe,a,b)
    generator=torch.Generator().manual_seed(int(seed)+71033)
    shuffled=rx.clone()
    for dy in day.unique(sorted=True):
        for k in range(6):
            index=torch.where((day==dy)&(y==k))[0]
            shuffled[index]=rx[index][torch.randperm(len(index),generator=generator)]
    records=[];references=[]
    for control,current_rx in [('original_RX',rx),('within_TX_day_shuffled_RX',shuffled)]:
        for day_index,dy in enumerate(day.unique(sorted=True).tolist()):
            day_ids=torch.where(day==dy)[0];domains=current_rx[day_ids].unique(sorted=True).tolist()
            if len(domains)<2:raise ValueError('At least two RX per day required')
            cells={(r,k):torch.where((day==dy)&(current_rx==r)&(y==k))[0] for r in domains for k in range(6)}
            sizes={key:len(value) for key,value in cells.items()}
            if min(sizes.values())<max(MS):raise ValueError('Every day/TX/RX cell must support m=16')
            exact_g=sample_gradient(x[day_ids],weight,bias,y[day_ids],a,b)
            truth=exact_variances(exact_g,y[day_ids],current_rx[day_ids]);exact_loss=penalty(truth)
            exact_gradient=_gradient(exact_loss,a,b);truth_detached=truth.detach();target=float(exact_loss.detach())
            references.append(dict(control=control,day=dy,L_packets=len(day_ids),RX=domains,cell_population_sizes={f'{r}:{k}':v for (r,k),v in sizes.items()},
                exact_loss=target,exact_gradient_norm=float(exact_gradient.norm()),probe_CE=float(probe.detach()),probe_CE_gradient_norm=float(probe_gradient.norm()),
                exact_vs_probe_CE=_direction(exact_gradient,probe_gradient)))
            for m in MS:
                sampling=torch.Generator().manual_seed(int(seed)+day_index*100003+m*1009+(0 if control=='original_RX' else 800011))
                for repeat in range(ROUNDS):
                    ia=_draw(cells,m,sampling);ib=_draw(cells,m,sampling)
                    ga=sample_gradient(x[ia],weight,bias,y[ia],a,b);gb=sample_gradient(x[ib],weight,bias,y[ib],a,b)
                    old=pooled_variances(ga,current_rx[ia]);va=unbiased_variances(ga,y[ia],current_rx[ia],sizes);vb=unbiased_variances(gb,y[ib],current_rx[ib],sizes)
                    losses=dict(old_pooled_n_square=penalty(old),stratified_unbiased_square=penalty(va),independent_AB_cross=penalty(va,vb))
                    results={}
                    for name,loss in losses.items():
                        g=_gradient(loss,a,b)
                        results[name]=dict(loss=float(loss.detach()),error_vs_exact=float(loss.detach())-target,
                            squared_error_vs_exact=(float(loss.detach())-target)**2,
                            gradient_vs_exact=_direction(g,exact_gradient),gradient_vs_full_L_CE=_direction(g,probe_gradient),
                            gradient_squared_error_vs_exact=float((g-exact_gradient).square().mean()))
                    records.append(dict(control=control,day=dy,m=m,round=repeat,batch_A=len(ia),batch_B=len(ib),
                        cross_batch_PID_overlap=len(set(ia.tolist())&set(ib.tolist())),each_batch_unique=True,
                        exact_loss=target,variance_estimation=dict(old_A_MSE=float((old.detach()-truth_detached).square().mean()),
                            unbiased_A_MSE=float((va.detach()-truth_detached).square().mean()),unbiased_B_MSE=float((vb.detach()-truth_detached).square().mean()),
                            old_A_mean_error=float((old.detach()-truth_detached).mean()),unbiased_A_mean_error=float((va.detach()-truth_detached).mean())),losses=results))
    summary=[]
    for control in ('original_RX','within_TX_day_shuffled_RX'):
        for dy in day.unique(sorted=True).tolist():
            for m in MS:
                selected=[r for r in records if r['control']==control and r['day']==dy and r['m']==m]
                for method in ('old_pooled_n_square','stratified_unbiased_square','independent_AB_cross'):
                    values=[r['losses'][method] for r in selected]
                    summary.append(dict(control=control,day=dy,m=m,method=method,
                        loss=_stats(v['loss'] for v in values),error_vs_exact=_stats(v['error_vs_exact'] for v in values),
                        squared_error_vs_exact=_stats(v['squared_error_vs_exact'] for v in values),
                        gradient_squared_error_vs_exact=_stats(v['gradient_squared_error_vs_exact'] for v in values),
                        gradient_cosine_vs_exact=_stats(v['gradient_vs_exact']['cosine'] for v in values),
                        gradient_cosine_vs_full_L_CE=_stats(v['gradient_vs_full_L_CE']['cosine'] for v in values),
                        cross_batch_PID_overlap=_stats(r['cross_batch_PID_overlap'] for r in selected)))
    if not all(torch.equal(saved_head[k],v) for k,v in head.state_dict().items()):raise ValueError('Real head changed')
    if not all(torch.equal(a,b) for a,b in zip(saved_inputs,(features,labels,receivers,days))):raise ValueError('Source inputs changed')
    return dict(status='SOURCE_MOMENT_AUDIT_COMPLETE',source_role='L_s only',target_access=False,V_access=False,U_access=False,
        identity_updates=0,head_updates=0,real_head_and_inputs_unchanged=True,seed=int(seed),m_values=list(MS),rounds_per_day=ROUNDS,
        dimension=c,gradient_coordinates=6*(c+1),virtual_parameters=2*c,
        gradient_scope='shared x*(1+a)+b at a=b=0; virtual feature affine only, NOT backbone gradients',
        probe_CE_scope='all legal L through the SAME fixed auxiliary six-class linear head; not the full identity model CE',
        exact_target='per-day six-TX-uniform finite-L population gradient variance, then equal-RX centered squared deviation',
        estimator='Bhat + mean_k[(1-5/(6*m)-1/(6*N_k))*s_k^2], s denominator m-1',
        cross_loss='mean((Vhat_A-mean_RX(Vhat_A))*(Vhat_B-mean_RX(Vhat_B))); both graphs differentiated',
        sampling='independent A/B simple random samples without replacement within each cell; cross-batch overlap allowed',
        shuffled_control='permute RX only within same TX/day, preserve cell counts; compare against its own finite-population exact target',
        interpretation='Cross loss may be negative; no clipping/absolute value. Twelve rounds describe realized noise, not proof of unbiasedness.',
        references=references,summary=summary,rounds=records)
