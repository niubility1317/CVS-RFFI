"""FastTrust-inspired H singleton/partial-set targets; ordinary raw PL is unchanged."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from .rx_mean_max_objective import Objective as MaxObjective,Exposure as MaxExposure,RECIPE as MAX_RECIPE

METHODS=('RXTrust-P',)
RECIPE=deepcopy(MAX_RECIPE)
RECIPE.update(H_target='rawPLaccepted:union(raw_argmax,H_argmax);otherwise_sameH_softKL',
    teacher='current_eval_detached_raw_anchor_and_sameH_probs_no_EMA',
    raw_target='unchanged_samebase_softKL_allU',H_route_threshold=.95,H_set_size='one_if_agree_else_two',
    H_set_loss='negative_log_sum_student_probability_in_set',H_denominator='all_U_no_selected_count_rescale',
    route_label_source='current_eval_detached_raw_probs_only_no_U_truth',new_F_forwards_over_RXMeanMax=0)


def candidates(raw_prob,H_prob):
    if (raw_prob.ndim!=2 or raw_prob.shape[1]!=6 or H_prob.shape!=raw_prob.shape or raw_prob.requires_grad or H_prob.requires_grad
        or not torch.isfinite(raw_prob).all() or not torch.isfinite(H_prob).all() or (raw_prob<0).any() or (H_prob<0).any()
        or not torch.allclose(raw_prob.sum(1),torch.ones_like(raw_prob[:,0]),atol=1e-5)
        or not torch.allclose(H_prob.sum(1),torch.ones_like(H_prob[:,0]),atol=1e-5)):raise ValueError('Two detached normalized six-class same-packet probabilities required')
    accepted=raw_prob.max(1).values>=.95;anchor=raw_prob.argmax(1);alternative=H_prob.argmax(1)
    mask=F.one_hot(anchor,6).bool()|F.one_hot(alternative,6).bool()
    return accepted,mask


def routed_loss(logits,raw_prob,H_prob):
    accepted,mask=candidates(raw_prob,H_prob)
    if logits.shape!=raw_prob.shape or not torch.isfinite(logits).all():raise ValueError('Aligned finite original student default-head logits required')
    logp=logits.log_softmax(1);partial=-torch.logsumexp(logp.masked_fill(~mask,-torch.inf),1)
    local=F.kl_div(logp,H_prob,reduction='none').sum(1)
    return torch.where(accepted,partial,local).mean(),accepted,mask


class Exposure(MaxExposure):
    def __init__(self,metadata):
        super().__init__(metadata);self.P_singleton=self.P_pair=self.P_fallback=0
        self.P_by_rx={str(r):dict(singleton=0,pair=0,fallback=0) for r in (1,3,4,6,8)};self.P_by_pseudo_class={str(k):0 for k in range(6)}

    def state(self):
        return dict(super().state(),P_H_singleton=self.P_singleton,P_H_pair=self.P_pair,P_H_soft_fallback=self.P_fallback,
            P_H_by_source_RX=self.P_by_rx,P_H_by_raw_pseudo_class=self.P_by_pseudo_class,
            P_target_rule=RECIPE['H_target'],P_U_TX_truth_read=False,P_extra_F_forwards=0)


class Objective(MaxObjective):
    def plain(self,student,device):
        if self.history is None:self.history=Exposure(student._cd_U_loader.dataset.metadata)
        return super().plain(student,device)

    def source_U_loss(self,logits,base_prob,mode):
        h=self.history
        if mode=='raw':value=super().source_U_loss(logits,base_prob,mode);single=pair=fallback=None
        else:
            value,accepted,mask=routed_loss(logits,self.raw_prob,base_prob)
            if not torch.equal(accepted,self.accepted):raise ValueError('The additional H route must preserve the actual ordinary PL acceptance rule')
            single=accepted&mask.sum(1).eq(1);pair=accepted&mask.sum(1).eq(2);fallback=~accepted
            h.P_singleton+=int(single.sum());h.P_pair+=int(pair.sum());h.P_fallback+=int(fallback.sum())
            rx=self.batch[3]['rx_i'].to(logits.device)
            for r in h.P_by_rx:
                chosen=rx==int(r)
                for name,scope in [('singleton',single),('pair',pair),('fallback',fallback)]:h.P_by_rx[r][name]+=int((chosen&scope).sum())
            pseudo=self.raw_prob.argmax(1)
            for k in h.P_by_pseudo_class:h.P_by_pseudo_class[k]+=int((accepted&(pseudo==int(k))).sum())
        self.last.update(RXP_H_singleton_seen=h.P_singleton,RXP_H_pair_seen=h.P_pair,RXP_H_soft_fallback_seen=h.P_fallback,
            RXP_current_singleton=0 if single is None else int(single.sum()),RXP_current_pair=0 if pair is None else int(pair.sum()),
            RXP_current_soft_fallback=0 if fallback is None else int(fallback.sum()))
        return value
