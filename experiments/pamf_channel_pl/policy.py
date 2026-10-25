"""Keep ordinary-PL acceptance and conserve every accepted sample's CE dose."""
import torch
from torch.nn import functional as F

RECIPE=dict(raw_threshold=.95,channel_threshold=.95,channel_to_raw_CE_ratio=.25,U_weight=1.,
    physical_half='alternating_native_T1_half',generator='separate_original_T1_Practical',
    channel_gate='raw_accepted_and_same_argmax_and_channel_confident',
    invalid_channel='exact_original_raw_PL',normalizer='per_sample_1_plus_channel_ratio',
    U_pair_KL=0.,EMA_teacher=False,new_head=False,new_inference_parameters=0)


def validate_recipe(recipe):
    if recipe!=RECIPE:raise ValueError('Only the preregistered ordinary-PL dose and source-channel policy allowed')


def raw_predictions(logits):
    if logits.ndim!=2 or not len(logits) or not bool(torch.isfinite(logits).all()):raise ValueError('Finite source teacher logits required')
    confidence,pseudo=logits.detach().softmax(1).max(1)
    return confidence,pseudo,confidence>=RECIPE['raw_threshold']


def channel_gate(pseudo,accepted,candidates,channel_logits):
    if (pseudo.ndim!=1 or accepted.shape!=pseudo.shape or pseudo.dtype!=torch.long or accepted.dtype!=torch.bool
            or candidates.ndim!=1 or candidates.dtype!=torch.long or len(candidates.unique())!=len(candidates)
            or len(candidates) and (int(candidates.min())<0 or int(candidates.max())>=len(pseudo))
            or channel_logits.ndim!=2 or len(channel_logits)!=len(candidates) or not bool(torch.isfinite(channel_logits).all())):
        raise ValueError('Aligned finite source teacher predictions and unique candidate indices required')
    if not len(candidates):return torch.empty(0,dtype=torch.bool,device=pseudo.device)
    confidence,prediction=channel_logits.detach().softmax(1).max(1)
    return accepted[candidates]&(prediction==pseudo[candidates])&(confidence>=RECIPE['channel_threshold'])


def dose_conserving_loss(raw_ce,accepted,indices,channel_logits,pseudo):
    if (raw_ce.ndim!=1 or not len(raw_ce) or not bool(torch.isfinite(raw_ce).all())
            or accepted.dtype!=torch.bool or accepted.shape!=raw_ce.shape or pseudo.shape!=raw_ce.shape or pseudo.dtype!=torch.long
            or indices.ndim!=1 or indices.dtype!=torch.long or len(indices.unique())!=len(indices)
            or len(indices) and (int(indices.min())<0 or int(indices.max())>=len(raw_ce) or not bool(accepted[indices].all()))):
        raise ValueError('Original raw batch normalization and accepted unique transfer indices required')
    baseline=(raw_ce*accepted).mean()
    if not len(indices):return baseline,baseline.detach()*0.
    if (channel_logits is None or channel_logits.ndim!=2 or len(channel_logits)!=len(indices)
            or not bool(torch.isfinite(channel_logits).all())):raise ValueError('Aligned finite existing-head channel logits required')
    extra_ce=F.cross_entropy(channel_logits,pseudo[indices],reduction='none')
    ratio=RECIPE['channel_to_raw_CE_ratio'];delta=(ratio/(1.+ratio))*(extra_ce-raw_ce[indices]).sum()/len(raw_ce)
    return baseline+delta,delta
