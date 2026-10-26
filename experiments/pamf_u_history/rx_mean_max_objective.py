"""Select the largest same-packet prediction change among four source RX styles."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from .rx_mean_objective import Objective as MeanObjective,Exposure as MeanExposure,RECIPE as MEAN_RECIPE,soft_loss
from .covariance_objective import schedule,combine

METHODS=('RXMean-MaxU',)
RECIPE=deepcopy(MEAN_RECIPE)
RECIPE.update(style_selection='per_packet_max_teacher_to_candidate_KL_over_other4_sourceRX',L_gradient_scope='detached',
    candidate_variants=4,candidate_teacher_packets_per_U=4,candidate_teacher_calls_per_update=1,
    candidate_selection='current_eval_detached_no_TX_truth_no_batch_reassignment',
    coefficient_rule='min(1,.1*median(primary_norm)/median(aux_norm));12train_batches_6raw6H,noV')


def all_styles(shift,iq,receivers):
    if iq.ndim!=3 or tuple(iq.shape[1:])!=(2,256) or iq.dtype!=torch.float32 or not torch.isfinite(iq).all():raise ValueError('Finite source FP32 packet IQ required')
    rx=receivers.to(iq.device);matches=rx[:,None]==shift.rxs[None,:]
    if len(rx)!=len(iq) or not bool(matches.sum(1).eq(1).all()):raise ValueError('Original source RX for each physical packet required')
    original=matches.long().argmax(1);target=(original[None,:]+torch.arange(1,5,device=iq.device)[:,None]).remainder(5)
    spectrum=torch.fft.fft(torch.complex(iq[:,0],iq[:,1]),dim=1)
    z=torch.fft.ifft(spectrum[None,:,:]*shift.transfer[original[None,:],target],dim=2)
    return torch.stack((z.real,z.imag),2),shift.rxs[target]


def choose(probabilities,candidate_logits):
    if (probabilities.ndim!=2 or probabilities.shape[1]!=6 or candidate_logits.shape!=(4,*probabilities.shape)
        or probabilities.requires_grad or candidate_logits.requires_grad or not torch.isfinite(probabilities).all() or not torch.isfinite(candidate_logits).all()
        or (probabilities<0).any() or not torch.allclose(probabilities.sum(1),torch.ones_like(probabilities[:,0]),atol=1e-5)):
        raise ValueError('Four finite detached six-class candidate predictions and normalized same-base probabilities required')
    losses=F.kl_div(candidate_logits.log_softmax(2),probabilities[None,:,:].expand_as(candidate_logits),reduction='none').sum(2)
    return losses.argmax(0),losses


class Exposure(MeanExposure):
    def state(self):
        result=super().state()
        result.update(candidate_U_packets=4*self.seen,teacher_U_packets=5*self.seen+self.H_seen,candidate_variants=4,
            selection_policy=RECIPE['style_selection'],selection_uses_U_TX=False)
        return result


class Objective(MeanObjective):
    def plain(self,student,device):
        if self.history is None:self.history=Exposure(student._cd_U_loader.dataset.metadata)
        return super().plain(student,device)

    def source_U_loss(self,logits,base_prob,mode):
        return soft_loss(logits,base_prob)

    def extra(self,student,kwargs):
        if self.L_forwards!=5 or set(self.inputs)!={'H','raw'}:raise ValueError('Original five extra L forwards required')
        call=student._feature_l_steps+self.c.get('step_offset',0)-4440
        if call!=self.history.calls+1:raise ValueError('Continuous own E20+10+20 trajectory required')
        mode=self.c.get('_calibration_base_override') or ('raw' if ((call-1)//222)%2==0 else 'H')
        if mode not in ('raw','H') or ('_calibration_base_override' in self.c and not self.c['rx_mean_calibration']):raise ValueError('Fixed alternating raw/H epoch schedule')
        replica,scene=schedule(call);device=kwargs['labels'].device;n=len(self.batch[0])
        base=self.batch[0].to(device) if mode=='raw' else self.cache.take(self.batch[3],replica,scene,device)
        variants,styles=all_styles(self.shift,base,self.batch[3]['rx_i']);index=torch.arange(n,device=device)
        mask=(torch.arange(128,device=device)+student._feature_l_steps).remainder(2)!=0
        lbase=self.inputs[mode][mask];lrx=kwargs['extra'][1]['rx_i'].to(device)[mask]
        lshift,_=self.shift(lbase,lrx,torch.arange(64,device=device),call)
        student.eval()
        try:
            with torch.no_grad():
                up=self.raw_prob if mode=='raw' else student(base).softmax(1);lp=student(lbase).softmax(1)
                logits=student(variants.reshape(4*n,2,256)).reshape(4,n,6)
                selected,losses=choose(up,logits)
                shifted=variants[selected,index];selected_styles=styles[selected,index]
            ul=student(shifted);ll=student(lshift)
        finally:student.train()
        u=self.source_U_loss(ul,up,mode);l=soft_loss(ll,lp);value=combine(u,l,n,64,'U')
        if self.c['rx_mean_calibration']:
            self.calibration_nodes=(ul,ll);self.candidates={self.c['paper_method']:value}
        else:self.candidates={}
        h=self.history;h.calls+=1;h.seen+=n;h.accepted+=int(self.accepted.sum());h.visits[self.ids]+=1;h.L_seen+=64
        h.raw_seen+=n*(mode=='raw');h.H_seen+=n*(mode=='H');h.by_replica[replica-1]+=n*(mode=='H');h.by_scene[scene]+=n*(mode=='H')
        h.nonzero_U+=int((shifted-base).flatten(1).abs().amax(1).gt(0).sum())
        for rx in self.batch[3]['rx_i'].unique().tolist():
            chosen=self.batch[3]['rx_i']==rx;r=h.by_rx.setdefault(str(rx),dict(seen=0,accepted=0,mean_shift=0))
            r['seen']+=int(chosen.sum());r['mean_shift']+=int(chosen.sum());r['accepted']+=int((chosen.to(device)&self.accepted).sum())
            for style in selected_styles.unique().tolist():
                count=int((chosen.to(device)&(selected_styles==style)).sum())
                if count:h.styles[str(rx)+'->'+str(style)]=h.styles.get(str(rx)+'->'+str(style),0)+count
        weight=self.c['rx_mean_coefficient']*min(call/444.,1.)
        random_offset=(self.ids.to(device)+call-1).remainder(4)
        self.last.update(RXM_loss=value.detach(),RXM_U_loss=u.detach(),RXM_L_loss=l.detach(),RXM_weight=weight,RXM_seen=h.seen,RXM_calls=h.calls,
            RXM_H_base=int(mode=='H'),RXM_L_pairs=h.L_seen,RXM_teacher_U_seen=5*h.seen+h.H_seen,RXM_nonzero_shift_U=h.nonzero_U,
            RXW_candidate_U_packets=4*h.seen,RXW_selected_KL=losses[selected,index].mean(),RXW_random_KL=losses[random_offset,index].mean(),
            RXW_all_style_KL_mean=losses.mean(),**{'RXM_'+k:v for k,v in self.shift.diagnostic.items()})
        return weight*value
