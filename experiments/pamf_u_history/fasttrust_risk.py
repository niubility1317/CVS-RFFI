"""FastTrust-inspired source threshold arithmetic; no learned identity/calibrator head."""
import numpy as np


def risk_threshold(score, safe, eligible, receivers, *, precision=.995, min_per_rx=20):
    score=np.asarray(score);safe=np.asarray(safe);eligible=np.asarray(eligible);receivers=np.asarray(receivers)
    if (score.ndim!=1 or any(v.shape!=score.shape for v in (safe,eligible,receivers))
        or not np.isfinite(score).all() or safe.dtype!=bool or eligible.dtype!=bool
        or not 0<precision<=1 or min_per_rx<1):
        raise ValueError('Aligned finite scores and source risk observations required')
    order=np.flatnonzero(eligible);order=order[np.argsort(-score[order],kind='stable')]
    if not len(order):return None
    count=np.arange(1,len(order)+1);valid=np.cumsum(safe[order])/count>=precision
    # A deployed >= threshold includes every tie, never an arbitrary sorted prefix.
    valid&=np.r_[score[order][:-1]!=score[order][1:],True]
    for rx in np.unique(receivers):
        m=receivers[order]==rx;n=np.cumsum(m);correct=np.cumsum(m&safe[order])
        valid&=(n>=min_per_rx)&(correct/np.maximum(n,1)>=precision)
    if not valid.any():return None
    end=np.flatnonzero(valid)[-1]
    return float(score[order[end]])


def leave_receiver_out(score,safe,eligible,receivers,*,precision=.995,min_per_rx=20):
    """Threshold uses four source RX, evaluates only the fifth; no network updates."""
    out=np.zeros(len(score),dtype=bool);folds=[]
    for rx in np.unique(receivers):
        train=receivers!=rx;held=~train
        threshold=risk_threshold(score[train],safe[train],eligible[train],receivers[train],
                                 precision=precision,min_per_rx=min_per_rx)
        chosen=held&eligible&(score>=threshold) if threshold is not None else np.zeros(len(score),dtype=bool)
        out|=chosen;n=int(chosen.sum());correct=int((chosen&safe).sum())
        folds.append(dict(held_source_RX=int(rx),threshold=threshold,selected=n,safe=correct,
                          precision_percent=100*correct/n if n else None))
    return out,folds
