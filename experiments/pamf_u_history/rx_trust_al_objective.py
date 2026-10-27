"""Keep trusted U anchors; add true-label CE on existing RX-shifted L views."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from .rx_trust_a_objective import Objective as AnchorObjective, Exposure as AnchorExposure, RECIPE as A_RECIPE
from .covariance_objective import combine

METHODS=('RXTrust-AL',)
RECIPE=deepcopy(A_RECIPE)
RECIPE.update(L_gradient_scope='joint_true_L_TX_CE_LS0.01',
    L_target='known_source_L_TX_only;cross_entropy_label_smoothing0.01',
    L_base='unchanged_native_raw_or_ordinary_As_then_RXstyle;not_satellite_H',
    L_teacher_use='detached_monitor_and_matched_forward_budget_only',
    scope_scalar='(nU*unchanged_A_U+nL*true_L_CE)/(nU+nL)',
    new_F_forwards_over_RXTrustA=0)


class Exposure(AnchorExposure):
    def __init__(self,metadata):
        super().__init__(metadata)
        self.L_CE_seen=0
        self.L_CE_by_true_class={str(k):0 for k in range(6)}
        self.L_CE_by_base={'raw':0,'native_As':0}

    def state(self):
        return dict(super().state(),AL_L_true_CE_packets=self.L_CE_seen,
            AL_L_by_true_TX=self.L_CE_by_true_class,AL_L_by_base=self.L_CE_by_base,
            AL_L_rule=RECIPE['L_target'],AL_L_gradient_active=True,
            AL_L_label_source='original_labeled_batch_labels_subset_only',AL_U_TX_truth_read=False)


class Objective(AnchorObjective):
    def plain(self,student,device):
        if self.history is None:self.history=Exposure(student._cd_U_loader.dataset.metadata)
        return super().plain(student,device)

    def source_L_loss(self,logits,base_prob,labels,mode):
        if (logits.ndim!=2 or logits.shape[1]!=6 or len(logits)!=64 or not torch.isfinite(logits).all()
            or base_prob.shape!=logits.shape or base_prob.requires_grad or labels.shape!=(64,)
            or labels.dtype!=torch.long or labels.min()<0 or labels.max()>=6 or mode not in ('raw','H')):
            raise ValueError('Original 64 visible source L labels and six-class shifted L logits required')
        loss=F.cross_entropy(logits,labels,label_smoothing=.01)
        h=self.history;h.L_CE_seen+=64;h.L_CE_by_base['raw' if mode=='raw' else 'native_As']+=64
        for k in h.L_CE_by_true_class:h.L_CE_by_true_class[k]+=int((labels==int(k)).sum())
        self.last.update(AL_L_CE_seen=h.L_CE_seen,AL_L_current=64,AL_L_gradient_active=1,AL_L_LS=.01)
        return loss

    def combine_source_losses(self,u,l,nu,nl):
        value=combine(u,l,nu,nl,'LU')
        if self.c.get('rx_mean_calibration'):
            self.calibration_components={'U':nu*u/(nu+nl),'L':nl*l/(nu+nl)}
        return value
