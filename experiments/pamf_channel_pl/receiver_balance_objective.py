"""Bounded source-RX semantic contribution balancing on unchanged H2-LU views."""
from copy import deepcopy
from pathlib import Path
import torch
from torch.nn import functional as F
from .clean_h_objective import CleanHObjective,physical_mask

METHOD_MODES={'REPL_RB_RX':'RX','REPL_RB_RXTX':'RX_TX'}
BASE_RECIPE=dict(frequency_decay=.99,frequency_floor=1e-6,minimum=.5,maximum=2.,
    prior='all true L_s label/RX counts',frequency='accepted raw95 U pseudo labels and visible RX only',
    scope='CE and RDrop on six existing U logits; original feature contrasts unchanged',
    normalize='accepted mean one',CE_normalizer=1.25,segment_statistics='reset to L_s prior each segment',
    extra_F=0,extra_CE_kernels=6,extra_KL_kernels=3,new_head=False)

def recipe(method):
    if method not in METHOD_MODES:raise ValueError('Two source receiver-balance variants only')
    return dict(BASE_RECIPE,mode=METHOD_MODES[method])


@torch.no_grad()
def bounded_mean_one(ratio):
    if ratio.ndim!=1 or not torch.is_floating_point(ratio) or not bool(torch.isfinite(ratio).all()) or bool((ratio<=0).any()):
        raise ValueError('Finite positive detached contribution ratios required')
    if not len(ratio):return ratio.detach().double()
    z=ratio.detach().double().log();lower=z.new_tensor(.5).log();upper=z.new_tensor(2.).log()
    lo=lower-z.max()-1.;hi=upper-z.min()+1.
    # Log-space bisection also handles sparse groups without a huge linear bracket.
    for _ in range(48):
        mid=(lo+hi)/2;current=(z+mid).clamp(lower,upper).exp()
        below=current.mean()<1.;lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
    w=(z+(lo+hi)/2).clamp(lower,upper).exp().clamp(.5,2.)
    if not bool(torch.isfinite(w).all()) or abs(float(w.sum())-len(w))>1e-8*max(1,len(w)):
        raise ValueError('Bounded accepted contribution mass must remain unchanged')
    return w


def group_indices(rx,pseudo,source_rxs,mode):
    if (mode not in ('RX','RX_TX') or rx.ndim!=1 or pseudo.shape!=rx.shape or rx.dtype!=torch.long or pseudo.dtype!=torch.long
        or rx.device!=pseudo.device or not source_rxs or len(set(source_rxs))!=len(source_rxs)
        or bool(((pseudo<0)|(pseudo>=6)).any())):raise ValueError('Visible source RX and model pseudo class0..5 only')
    allowed=torch.tensor(source_rxs,dtype=torch.long,device=rx.device);matches=rx[:,None]==allowed[None,:]
    if not bool((matches.sum(1)==1).all()):raise ValueError('No target/foreign receiver in source balancing')
    groups=matches.long().argmax(1)
    return groups if mode=='RX' else 6*groups+pseudo


class ReceiverFrequency:
    def __init__(self,source_rxs,labeled_counts,mode):
        self.source_rxs=list(source_rxs);self.mode=mode;n=len(source_rxs)*(6 if mode=='RX_TX' else 1)
        if mode not in ('RX','RX_TX') or labeled_counts.shape!=(n,) or not bool(torch.isfinite(labeled_counts).all()) or not bool((labeled_counts>0).all()):
            raise ValueError('Every declared source group needs a positive true-L prior')
        self.labeled_counts=labeled_counts.detach().double().clone();self.prior=self.labeled_counts/self.labeled_counts.sum();self.frequency=self.prior.clone()
        self.calls=self.updates=self.seen=self.accepted=0;self.mass=0.;self.min_weight=self.max_weight=1.;self.max_dose_error=0.
        self.accepted_group_counts=torch.zeros(n,dtype=torch.long);self.last={}

    @torch.no_grad()
    def weights(self,rx,pseudo,accepted):
        if accepted.dtype!=torch.bool or accepted.shape!=rx.shape or accepted.device!=rx.device:raise ValueError('Exact raw95 source acceptance mask required')
        group=group_indices(rx,pseudo,self.source_rxs,self.mode);a=int(accepted.sum());n=len(rx)
        self.prior=self.prior.to(rx.device);self.frequency=self.frequency.to(rx.device);self.accepted_group_counts=self.accepted_group_counts.to(rx.device)
        weights=torch.ones(n,dtype=torch.float64,device=rx.device)
        if a:
            counts=torch.bincount(group[accepted],minlength=len(self.prior));self.accepted_group_counts+=counts
            self.frequency.mul_(.99).add_(counts.double()/a,alpha=.01);self.frequency/=self.frequency.sum();self.updates+=1
            ratios=(self.prior/self.frequency.clamp_min(1e-6))[group[accepted]]
            weights[accepted]=bounded_mean_one(ratios)
            self.min_weight=min(self.min_weight,float(weights[accepted].min()));self.max_weight=max(self.max_weight,float(weights[accepted].max()))
        mass=float(weights[accepted].sum());error=abs(mass-a);self.calls+=1;self.seen+=n;self.accepted+=a;self.mass+=mass;self.max_dose_error=max(self.max_dose_error,error)
        self.last=dict(accepted=a,weight_mass=mass,weight_min=float(weights[accepted].min()) if a else 1.,weight_max=float(weights[accepted].max()) if a else 1.,dose_error=error)
        return weights

    def state_dict(self):
        if not bool(torch.isfinite(self.frequency).all()) or abs(float(self.frequency.sum())-1)>1e-10 or abs(self.mass-self.accepted)>1e-7*max(1,self.accepted):
            raise ValueError('Finite source-only accepted-frequency statistics required')
        return dict(mode=self.mode,prior_L_counts=self.labeled_counts.cpu().tolist(),frequency=self.frequency.cpu().tolist(),
            accepted_group_counts=self.accepted_group_counts.cpu().tolist(),calls=self.calls,frequency_updates=self.updates,seen=self.seen,accepted=self.accepted,
            accepted_weight_mass=self.mass,minimum_weight=self.min_weight,maximum_weight=self.max_weight,max_batch_dose_error=self.max_dose_error,
            initialized_from_V_or_U_truth=False,frequency_updates_from='U raw95 accepted pseudo class and true source RX',model_EMA_updates=0)


def semantic_reallocation(logits,pseudo,accepted,physical,weights):
    if (len(logits)!=6 or pseudo.ndim!=1 or not len(pseudo) or len(pseudo)%2 or pseudo.dtype!=torch.long
        or accepted.dtype!=torch.bool or accepted.shape!=pseudo.shape or physical.dtype!=torch.bool or physical.shape!=pseudo.shape
        or int(physical.sum())*2!=len(pseudo) or weights.shape!=pseudo.shape or weights.requires_grad
        or any(z.shape!=(len(pseudo),6) or z.device!=pseudo.device or not bool(torch.isfinite(z).all()) for z in logits)
        or accepted.device!=pseudo.device or physical.device!=pseudo.device or weights.device!=pseudo.device
        or not bool(torch.isfinite(weights).all()) or bool(((weights<.5)|(weights>2.)).any())
        or not torch.equal(weights[~accepted],torch.ones_like(weights[~accepted]))
        or abs(float(weights[accepted].double().sum())-int(accepted.sum()))>1e-6*max(1,int(accepted.sum()))):
        raise ValueError('Six unchanged source logits, original physical half, and bounded mean-one accepted weights required')
    ce=[F.cross_entropy(z,pseudo,reduction='none',label_smoothing=.01) for z in logits]
    def kl(a,b):
        p,q=a.log_softmax(1),b.log_softmax(1);return .5*((p.exp()-q.exp())*(p-q)).sum(1)
    ordinary=.5*(ce[0]+ce[1])+kl(logits[0],logits[1])
    mixed_ce=.5*(ce[2]+ce[3]);mixed_kl=kl(logits[2],logits[3])
    raw=.5*(ce[4]+ce[5])+kl(logits[4],logits[5])
    packet=.5*ordinary+.5*mixed_ce+.5*(~physical).to(mixed_kl.dtype)*mixed_kl+.25*raw
    mask=accepted.to(packet.dtype);w=weights.to(packet.dtype)
    before=(packet*mask).mean()/1.25;after=(packet*mask*w).mean()/1.25
    delta=(packet*mask*(w-1)).mean()/1.25
    return delta,before,after


def labeled_prior(c,contract,mode):
    import numpy as np
    with np.load(Path(c['source_cache'])/'L_s_metadata.npz',allow_pickle=False) as data:
        if data['ids'].tolist()!=contract['role_ids']['L_s'] or len(data['y'])!=6300:raise ValueError('Original true L_s6300 metadata only')
        y=torch.tensor(data['y'].tolist(),dtype=torch.long);rx=torch.tensor(data['rx_i'].tolist(),dtype=torch.long)
    groups=group_indices(rx,y,c['source_rxs'],mode);n=len(c['source_rxs'])*(6 if mode=='RX_TX' else 1)
    return torch.bincount(groups,minlength=n)


class ReceiverBalancedObjective(CleanHObjective):
    def __init__(self,c):
        if c.get('paper_method') not in METHOD_MODES or c.get('receiver_balance')!=recipe(c['paper_method']):raise ValueError('Fixed receiver-balance recipe required')
        proxy=deepcopy(c);proxy['paper_method']='REPL_H2_LU_Pair';super().__init__(proxy);self.c=c;self.pair_enabled=False
        mode=METHOD_MODES[c['paper_method']];self.frequency=ReceiverFrequency(c['source_rxs'],labeled_prior(c,self.contract,mode),mode)
        self._rb_logits=self._rb_current=None;self.rb_captures=self.rb_packets=0;self.rb_applied_mass_error=0.

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        if self._rb_logits is None or 'tx_label_visible' not in metadata or bool(metadata['tx_label_visible'].any()):raise ValueError('Only current hidden-TX U student logits may be balanced')
        rx=metadata['rx_i'].to(x.device).long();group_indices(rx,pseudo,self.c['source_rxs'],self.frequency.mode)
        if self._rb_current is None:self._rb_current=(pseudo,accepted,rx)
        elif any(not torch.equal(a,b) for a,b in zip(self._rb_current,(pseudo,accepted,rx))):raise ValueError('Six views must share original raw labels/gate/RX')
        logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        self._rb_logits.append(logits);self.rb_captures+=1;self.rb_packets+=len(pseudo);return logits

    def loss(self,student,device):
        if self._rb_logits is not None:raise ValueError('Nested source contribution context')
        self._rb_logits=[];self._rb_current=None
        try:
            zero=super().loss(student,device)
            if len(self._rb_logits)!=6 or self._rb_current is None or self.pending_gradient_loss is None:raise ValueError('Original six U student logits and guarded loss required')
            pseudo,accepted,rx=self._rb_current;weights=self.frequency.weights(rx,pseudo,accepted)
            physical=physical_mask(len(pseudo),student._feature_l_steps+self.c['step_offset'],pseudo.device)
            delta,before,after=semantic_reallocation(self._rb_logits,pseudo,accepted,physical,weights)
            self.pending_gradient_loss=self.pending_gradient_loss+self.settings['weight']*delta
            applied=weights.to(self._rb_logits[0].dtype);self.rb_applied_mass_error=max(self.rb_applied_mass_error,abs(float(applied[accepted].double().sum())-int(accepted.sum())))
            if self.c.get('candidate_smoke'):
                actual=torch.autograd.grad(self.pending_gradient_loss,self._rb_logits,retain_graph=True)
                expected=torch.autograd.grad(self.settings['weight']*after,self._rb_logits,retain_graph=True)
                error=max(float((a-b).abs().max()) for a,b in zip(actual,expected))
                if error>2e-6 or any(not bool(torch.isfinite(a).all()) for a in actual):raise ValueError('Actual semantic-logit gradients differ from weighted CE/RDrop')
                self.smoke_receiver_balance_gradient_check=dict(max_abs_error=error,views=6,weighted_CE_and_RDrop=True,feature_contrast_reweighted=False,extra_model_forwards=0,U_truth_reads=0)
            f=self.frequency
            self.metrics.update(HPL_loss=self.pending_gradient_loss.detach(),HPL_base_loss=(self.pending_gradient_loss/self.settings['weight']).detach(),
                HPL_RB_calls=f.calls,HPL_RB_frequency_updates=f.updates,HPL_RB_seen=f.seen,HPL_RB_accepted=f.accepted,
                HPL_RB_captures=self.rb_captures,HPL_RB_packet_views=self.rb_packets,HPL_RB_mass=f.mass,
                HPL_RB_min=f.min_weight,HPL_RB_max=f.max_weight,HPL_RB_max_dose_error=f.max_dose_error,HPL_RB_applied_dose_error=self.rb_applied_mass_error,
                HPL_RB_before=before.detach(),HPL_RB_after=after.detach(),HPL_RB_delta=delta.detach())
            return zero
        finally:self._rb_logits=self._rb_current=None

    def state_dict(self):
        value=super().state_dict();f=self.frequency
        if (self._rb_logits is not None or f.calls!=self.calls or f.seen!=self.seen or f.accepted!=self.accepted
            or self.rb_captures!=6*self.calls or self.rb_packets!=6*self.seen or self.rb_applied_mass_error>1e-4):raise ValueError('Original U budget and contribution lifetimes differ')
        value['receiver_balance']=dict(recipe=recipe(self.c['paper_method']),statistics=f.state_dict(),captures=self.rb_captures,captured_packet_views=self.rb_packets,
            max_FP32_accepted_mass_error=self.rb_applied_mass_error,extra_CE_kernels=6*self.calls,extra_KL_kernels=3*self.calls,
            extra_F=0,new_inference_parameters=0,U_truth_reads=0,target_reads=0,L_losses_reweighted=False,feature_contrasts_reweighted=False)
        return value
