"""Frozen two-domain labeler decisions; inputs contain predictions, never truth."""
import numpy as np


def probability(logits):
    value=np.asarray(logits,np.float64)
    if not np.isfinite(value).all():raise ValueError('Finite frozen classifier logits required')
    a=np.exp(value-value.max(axis=-1,keepdims=True))
    return a/a.sum(axis=-1,keepdims=True)


def certify(raw_logits,restored_logits,frozen_raw_confidence):
    raw=np.asarray(raw_logits);g=np.asarray(restored_logits);old=np.asarray(frozen_raw_confidence)
    if (raw.ndim!=2 or len(raw)<1 or raw.shape[1]!=6 or g.shape!=(12,len(raw),6) or old.shape!=(len(raw),)
            or not np.isfinite(old).all() or ((old<0)|(old>1)).any()):
        raise ValueError('Six-class raw and twelve-G predictions in original frozen hard-pool order required')
    r=probability(raw);p=probability(g).mean(axis=0);label=r.argmax(1);gl=p.argmax(1)
    rc=r.max(1);gc=p.max(1);hard=old<.95
    return dict(accepted=hard&(rc>=.95)&(gc>=.95)&(label==gl),label=label,
        raw_confidence=rc,G_confidence=gc,G_views_agree=(g.argmax(2)==gl[None,:]).sum(0),
        hard_pool=hard,raw_probability=r,G_mean_probability=p)


def quality_pass(certified,correct,net_correct):
    if (any(type(x) is not int for x in (certified,correct,net_correct)) or not 0<=correct<=certified or abs(net_correct)>certified):
        raise ValueError('Integer post-prediction source counts required')
    return certified>=200 and 100*correct>=99.5*certified and net_correct>=100
