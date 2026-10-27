"""Keep PAMF L smoothing; use ordinary hard-CE targets for the six accepted U views."""
from copy import deepcopy
import torch
from .clean_h_objective import CleanHObjective,physical_mask
from .receiver_balance_objective import ReceiverFrequency,labeled_prior,semantic_reallocation

METHODS={'REPL_UHardCE':'all_six_U_CE_hard'}


def recipe(method):
    if method not in METHODS:raise ValueError('One predeclared U target change only')
    return dict(mode=METHODS[method],base_U_label_smoothing=.01,effective_U_label_smoothing=0.,L_label_smoothing_unchanged=True,
        scope='all six accepted U CE terms only; original ordinary/raw KL and shallow-middle-joint cross-RX unchanged',
        raw_gate=.95,pseudo_labels_unchanged=True,CE_normalizer=1.25,extra_F=0,new_head=False,
        receiver_balance='same completed RX_TX prior, accepted pseudo/RX frequency and bounded mean-one semantic weights',
        exact_CE_delta='0.01 * (mean_class_logit - raw_pseudo_class_logit), original per-view coefficients')


def hard_CE_delta(logits,pseudo,accepted,semantic):
    if (len(logits)!=6 or pseudo.ndim!=1 or not len(pseudo) or pseudo.dtype!=torch.long or accepted.dtype!=torch.bool
        or accepted.shape!=pseudo.shape or semantic.shape!=pseudo.shape or semantic.requires_grad
        or semantic.device!=pseudo.device or accepted.device!=pseudo.device or not bool(torch.isfinite(semantic).all())
        or bool(((semantic<.5)|(semantic>2.)).any()) or bool(((pseudo<0)|(pseudo>=6)).any())
        or any(z.shape!=(len(pseudo),6) or z.device!=pseudo.device or not bool(torch.isfinite(z).all()) for z in logits)):
        raise ValueError('Six finite original U logits, original accepted labels and detached bounded weights required')
    # CE(LS=0)-CE(LS=.01) is exactly linear; no extra CE or softmax kernels.
    coefficients=(.25,.25,.25,.25,.125,.125);delta=logits[0].sum()*0.
    for z,coefficient in zip(logits,coefficients):
        difference=.01*(z.mean(1)-z.gather(1,pseudo[:,None]).squeeze(1))
        delta=delta+(coefficient/1.25)*(difference*accepted.to(z.dtype)*semantic.to(z.dtype)).mean()
    return delta


class UHardCEObjective(CleanHObjective):
    def __init__(self,c):
        if c.get('paper_method') not in METHODS or c.get('U_semantic_target')!=recipe(c['paper_method']):raise ValueError('Fixed hard U / unchanged L target recipe required')
        proxy=deepcopy(c);proxy['paper_method']='REPL_H2_LU_Pair';super().__init__(proxy);self.c=c;self.pair_enabled=False
        self.frequency=ReceiverFrequency(c['source_rxs'],labeled_prior(c,self.contract,'RX_TX'),'RX_TX')
        self.uh_logits=self.uh_current=None;self.uh_calls=self.uh_captures=self.uh_packets=0

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        if self.uh_logits is None or bool(metadata['tx_label_visible'].any()):raise ValueError('Current hidden-TX U loss only')
        current=(pseudo,accepted,metadata['rx_i'].to(x.device).long())
        if self.uh_current is None:self.uh_current=current
        elif any(not torch.equal(a,b) for a,b in zip(self.uh_current,current)):raise ValueError('Same raw95 labels/gate/RX across six views required')
        logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        self.uh_logits.append(logits);self.uh_captures+=1;self.uh_packets+=len(x);return logits

    def loss(self,student,device):
        if self.uh_logits is not None:raise ValueError('Nested U target context')
        self.uh_logits=[];self.uh_current=None
        try:
            zero=super().loss(student,device)
            if len(self.uh_logits)!=6 or self.uh_current is None or self.pending_gradient_loss is None:raise ValueError('Original six-view guarded U loss required')
            pseudo,accepted,rx=self.uh_current;semantic=self.frequency.weights(rx,pseudo,accepted)
            physical=physical_mask(len(pseudo),student._feature_l_steps+self.c['step_offset'],pseudo.device)
            rb,before,weighted=semantic_reallocation(self.uh_logits,pseudo,accepted,physical,semantic)
            delta=hard_CE_delta(self.uh_logits,pseudo,accepted,semantic)
            self.pending_gradient_loss=self.pending_gradient_loss+self.settings['weight']*(rb+delta)
            if self.c.get('candidate_smoke'):
                actual=torch.autograd.grad(self.pending_gradient_loss,self.uh_logits,retain_graph=True)
                expected=torch.autograd.grad(self.settings['weight']*(weighted+delta),self.uh_logits,retain_graph=True)
                error=max(float((a-b).abs().max()) for a,b in zip(actual,expected))
                if error>2e-6 or any(not bool(torch.isfinite(a).all()) for a in actual):raise ValueError('Actual hard U target gradients differ')
                self.smoke_receiver_balance_gradient_check=dict(max_abs_error=error,views=6,effective_U_CE_label_smoothing=0.,
                    original_raw95_labels_unchanged=True,L_losses_unchanged=True,extra_F=0)
            self.uh_calls+=1
            self.metrics.update(HPL_loss=self.pending_gradient_loss.detach(),HPL_base_loss=(self.pending_gradient_loss/self.settings['weight']).detach(),
                HPL_label_smoothing=0.,UH_base_label_smoothing=.01,UH_effective_label_smoothing=0.,UH_calls=self.uh_calls,
                UH_captures=self.uh_captures,UH_packets=self.uh_packets,UH_CE_delta=delta.detach(),UH_RB_delta=rb.detach())
            return zero
        finally:self.uh_logits=self.uh_current=None

    def state_dict(self):
        state=super().state_dict()
        if (self.uh_logits is not None or self.uh_calls!=self.calls or self.frequency.calls!=self.calls or self.frequency.accepted!=self.accepted
            or self.uh_captures!=6*self.calls or self.uh_packets!=6*self.seen):raise ValueError('Original six-view budget and U target lifetime differ')
        state['label_smoothing']=0.
        state['receiver_balance']=dict(statistics=self.frequency.state_dict(),U_semantic_target=dict(recipe=recipe(self.c['paper_method']),
            calls=self.uh_calls,captures=self.uh_captures,packets=self.uh_packets,extra_F=0,U_truth_reads=0,target_reads=0))
        return state
