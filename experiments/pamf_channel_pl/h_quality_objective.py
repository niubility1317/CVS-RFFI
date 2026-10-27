"""Adjust only accepted physical-H CE, retaining raw95 labels and cross-RX learning."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from .clean_h_objective import CleanHObjective,physical_mask
from .receiver_balance_objective import ReceiverFrequency,labeled_prior,semantic_reallocation

METHODS={'REPL_HQC_Fixed':'fixed_half','REPL_HQC_Soft':'soft','REPL_HQC_Gate':'gate'}


def recipe(method):
    if method not in METHODS:raise ValueError('Three fixed physical-H CE interventions only')
    return dict(mode=METHODS[method],scope='U physical H CE only, two existing mixed student logits',
        fixed_half=.5,soft_target=.5,soft_floor=.05,gate_target=.5,weight_gradient=False,
        raw_gate=.95,raw_pseudo_labels_unchanged=True,normalize='original whole U batch and 1.25, no H mass renormalization',
        receiver_balance='same RX_TX prior/EMA/clipped mean-one semantic weights as completed RB_RXTX',
        L_losses_unchanged=True,U_feature_contrasts_unchanged=True,extra_F=0,new_head=False)


def quality(logits,pseudo,mode):
    if mode not in METHODS.values() or logits.ndim!=2 or logits.shape!=(len(pseudo),6) or pseudo.dtype!=torch.long or pseudo.device!=logits.device or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite six-class current hidden-TX U logits and original raw pseudo labels required')
    with torch.no_grad():
        p=logits.softmax(1);py=p.gather(1,pseudo[:,None]).squeeze(1)
        if mode=='fixed_half':w=py.new_full(py.shape,.5)
        elif mode=='soft':w=(py/.5).clamp(.05,1.)
        else:w=((p.argmax(1)==pseudo)&(py>=.5)).to(logits.dtype)
    return w


def physical_H_delta(logits,pseudo,accepted,physical,semantic,mode):
    if (len(logits)!=6 or pseudo.ndim!=1 or not len(pseudo) or pseudo.dtype!=torch.long or accepted.dtype!=torch.bool or physical.dtype!=torch.bool
        or accepted.shape!=pseudo.shape or physical.shape!=pseudo.shape or int(physical.sum())*2!=len(pseudo)
        or semantic.shape!=pseudo.shape or semantic.requires_grad or not bool(torch.isfinite(semantic).all())
        or bool(((semantic<.5)|(semantic>2)).any()) or any(z.shape!=(len(pseudo),6) for z in logits)):
        raise ValueError('Six original logits and bounded detached semantic weights on the physical half required')
    masks=accepted&physical;delta=logits[0].sum()*0.;details=[]
    for z in logits[2:4]:
        weights=quality(z,pseudo,mode);ce=F.cross_entropy(z,pseudo,reduction='none',label_smoothing=.01)
        # Existing mixed CE is .5 * .5 * (CE_a + CE_b), then /1.25.
        delta=delta+(.25/1.25)*(ce*masks.to(z.dtype)*semantic.to(z.dtype)*(weights-1)).mean()
        selected=weights[masks]
        details.append(dict(eligible=int(masks.sum()),nonzero=int((selected>0).sum()),weight_sum=float(selected.double().sum()),
            full_weight=int((selected==1).sum()),zero_weight=int((selected==0).sum()),semantic_weight_sum=float((semantic[masks]*selected).double().sum())))
    return delta,details


class HQualityObjective(CleanHObjective):
    def __init__(self,c):
        if c.get('paper_method') not in METHODS or c.get('H_quality')!=recipe(c['paper_method']):raise ValueError('Fixed H-quality recipe required')
        proxy=deepcopy(c);proxy['paper_method']='REPL_H2_LU_Pair';super().__init__(proxy);self.c=c;self.pair_enabled=False
        self.frequency=ReceiverFrequency(c['source_rxs'],labeled_prior(c,self.contract,'RX_TX'),'RX_TX')
        self.hq_logits=self.hq_current=None;self.hq_calls=self.hq_captures=self.hq_packets=0
        self.hq_eligible=self.hq_nonzero=self.hq_full=self.hq_zero=0;self.hq_mass=self.hq_semantic_mass=0.

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        if self.hq_logits is None or bool(metadata['tx_label_visible'].any()):raise ValueError('Current hidden-TX U loss only')
        current=(pseudo,accepted,metadata['rx_i'].to(x.device).long())
        if self.hq_current is None:self.hq_current=current
        elif any(not torch.equal(a,b) for a,b in zip(self.hq_current,current)):raise ValueError('Six views must share original raw labels/gate/RX')
        logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        self.hq_logits.append(logits);self.hq_captures+=1;self.hq_packets+=len(x);return logits

    def loss(self,student,device):
        if self.hq_logits is not None:raise ValueError('Nested H quality context')
        self.hq_logits=[];self.hq_current=None
        try:
            zero=super().loss(student,device)
            if len(self.hq_logits)!=6 or self.hq_current is None or self.pending_gradient_loss is None:raise ValueError('Original six-view guarded U loss required')
            pseudo,accepted,rx=self.hq_current;semantic=self.frequency.weights(rx,pseudo,accepted)
            physical=physical_mask(len(pseudo),student._feature_l_steps+self.c['step_offset'],pseudo.device)
            rb,before,weighted=semantic_reallocation(self.hq_logits,pseudo,accepted,physical,semantic)
            delta,details=physical_H_delta(self.hq_logits,pseudo,accepted,physical,semantic,METHODS[self.c['paper_method']])
            self.pending_gradient_loss=self.pending_gradient_loss+self.settings['weight']*(rb+delta)
            if self.c.get('candidate_smoke'):
                actual=torch.autograd.grad(self.pending_gradient_loss,self.hq_logits,retain_graph=True)
                expected=torch.autograd.grad(self.settings['weight']*(weighted+delta),self.hq_logits,retain_graph=True)
                error=max(float((a-b).abs().max()) for a,b in zip(actual,expected))
                if error>2e-6 or any(not bool(torch.isfinite(a).all()) for a in actual):raise ValueError('Actual semantic plus H quality gradients differ')
                self.smoke_receiver_balance_gradient_check=dict(max_abs_error=error,views=6,H_quality=recipe(self.c['paper_method']),
                    actual_H_weight_details=details,raw_gate_and_labels_unchanged=True,extra_F=0)
            self.hq_calls+=1
            for d in details:
                self.hq_eligible+=d['eligible'];self.hq_nonzero+=d['nonzero'];self.hq_full+=d['full_weight'];self.hq_zero+=d['zero_weight']
                self.hq_mass+=d['weight_sum'];self.hq_semantic_mass+=d['semantic_weight_sum']
            self.metrics.update(HPL_loss=self.pending_gradient_loss.detach(),HPL_base_loss=(self.pending_gradient_loss/self.settings['weight']).detach(),
                HQ_calls=self.hq_calls,HQ_eligible=self.hq_eligible,HQ_nonzero=self.hq_nonzero,HQ_full=self.hq_full,HQ_zero=self.hq_zero,
                HQ_mass=self.hq_mass,HQ_semantic_mass=self.hq_semantic_mass,HQ_delta=delta.detach(),HQ_RB_delta=rb.detach())
            return zero
        finally:self.hq_logits=self.hq_current=None

    def state_dict(self):
        state=super().state_dict()
        if (self.hq_logits is not None or self.hq_calls!=self.calls or self.frequency.calls!=self.calls or self.frequency.accepted!=self.accepted
            or self.hq_captures!=6*self.calls or self.hq_packets!=6*self.seen or self.hq_nonzero+self.hq_zero!=self.hq_eligible
            or not 0<=self.hq_mass<=self.hq_eligible):raise ValueError('Original U budget/quality lifetime differs')
        state['receiver_balance']=dict(statistics=self.frequency.state_dict(),H_quality=dict(recipe=recipe(self.c['paper_method']),
            calls=self.hq_calls,captures=self.hq_captures,packets=self.hq_packets,eligible=self.hq_eligible,nonzero=self.hq_nonzero,full=self.hq_full,zero=self.hq_zero,
            weight_mass=self.hq_mass,semantic_weight_mass=self.hq_semantic_mass,extra_F=0,U_truth_reads=0,target_reads=0))
        return state
