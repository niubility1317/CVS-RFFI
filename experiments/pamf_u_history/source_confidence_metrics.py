"""Source-V posterior diagnostics; logit gradients are not parameter gradients."""
import numpy as np


def probabilities(logits):
    z=np.asarray(logits,dtype=np.float64)
    if z.ndim!=2 or z.shape[1]!=6 or not len(z) or not np.isfinite(z).all():raise ValueError('Finite nonempty six-class source logits required')
    z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)


def statistics(logits,labels,*,role):
    if role!='V':raise ValueError('Only visible-label source validation diagnostics')
    p=probabilities(logits);y=np.asarray(labels)
    if y.shape!=(len(p),) or y.dtype.kind not in 'iu' or y.min()<0 or y.max()>=6:raise ValueError('Aligned visible six-class V labels required')
    pred=p.argmax(1);confidence=p.max(1);correct=pred==y;accepted=confidence>=.95;errors=~correct
    hard=p-np.eye(6)[pred];smooth=hard+.01*(np.eye(6)[pred]-1/6.)
    dot=(hard*smooth).sum(1);norm=np.sqrt(np.square(hard).sum(1));n=len(p);na=int(accepted.sum());wrong=int(errors.sum())
    ece=0.
    for k in range(15):
        chosen=(confidence>=k/15)&(confidence<(k+1)/15 if k<14 else confidence<=1.)
        if chosen.any():ece+=chosen.mean()*abs(confidence[chosen].mean()-correct[chosen].mean())
    return dict(count=n,correct=int(correct.sum()),accuracy_pct=100*correct.mean(),accepted=na,coverage_pct=100*accepted.mean(),
        accepted_correct=int((accepted&correct).sum()),accepted_precision_pct=100*correct[accepted].mean() if na else None,
        rejected=n-na,rejected_errors=int((~accepted&errors).sum()),rejected_share_of_all_errors_pct=100*(~accepted&errors).sum()/wrong if wrong else None,
        accepted_pseudo_CE_logit_gradient_norm_mean=float(norm[accepted].mean()) if na else None,
        rejected_pseudo_CE_logit_gradient_norm_mean=float(norm[~accepted].mean()) if (~accepted).any() else None,
        accepted_LS001_opposes_hard_pseudo_CE_pct=100*(dot[accepted]<0).mean() if na else None,
        mean_confidence=float(confidence.mean()),NLL=float(-np.log(p[np.arange(n),y].clip(1e-300)).mean()),
        Brier=float(np.square(p-np.eye(6)[y]).sum(1).mean()),ECE15_pct=100*ece)


def grouped(logits,y,rx,day):
    y=np.asarray(y);rx=np.asarray(rx);day=np.asarray(day)
    if len(y)!=len(rx) or len(y)!=len(day):raise ValueError('Aligned source validation metadata required')
    masks=[('overall','ALL',np.ones(len(y),bool))]
    for name,a in [('TX',y),('RX',rx),('day',day)]:masks.extend((name,str(v),a==v) for v in np.unique(a))
    masks.extend(('RX_day',str(r)+'/'+str(d),(rx==r)&(day==d)) for r in np.unique(rx) for d in np.unique(day))
    return [dict(dimension=name,stratum=value,**statistics(np.asarray(logits)[mask],y[mask],role='V')) for name,value,mask in masks if mask.any()]
