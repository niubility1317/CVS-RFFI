"""Fixed seed-agreement diagnostics; no fitted thresholds or new classifier."""
import torch

THRESHOLD=.95


def policies(logits):
    if (logits.ndim!=3 or logits.shape[0]!=3 or not logits.shape[1] or logits.shape[2] not in (6,10)
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Three finite same-contract native heads in six/ten-class space required')
    probability=logits.detach().softmax(-1);confidence,prediction=probability.max(-1);result={}
    for i in range(3):
        result['single_'+str(i)]=dict(pseudo=prediction[i],accepted=confidence[i]>=THRESHOLD)
        peers=[j for j in range(3) if j!=i];a,b=peers
        result['peer_pair_for_'+str(i)]=dict(pseudo=prediction[a],
            accepted=(confidence[a]>=THRESHOLD)&(confidence[b]>=THRESHOLD)&(prediction[a]==prediction[b]))
    result['unanimous3']=dict(pseudo=prediction[0],accepted=(confidence>=THRESHOLD).all(0)&(prediction==prediction[0]).all(0))
    confidence,pseudo=probability.mean(0).max(-1)
    result['mean_probability3']=dict(pseudo=pseudo,accepted=confidence>=THRESHOLD)
    return result


def summarize(policy,truth,group=None):
    accepted=policy['accepted'];pseudo=policy['pseudo']
    if truth.shape!=pseudo.shape or accepted.shape!=truth.shape:raise ValueError('Aligned source-V counts required')
    if group is not None:
        if group.shape!=truth.shape:raise ValueError('Aligned source group required')
        accepted=accepted&group
    count=int(accepted.sum());correct=int((accepted&(pseudo==truth)).sum())
    available=len(truth) if group is None else int(group.sum())
    return dict(available=available,accepted=count,correct=correct,incorrect=count-correct,
        precision_percent=100*correct/count if count else None,coverage_percent=100*count/available if available else None)
