"""Ordinary PL plus local distributional smoothness on every hidden-label source U."""
import torch
from torch.nn import functional as F
from .history import KEYS

METHODS=('PL-VAT',)
RECIPE=dict(positive_threshold=.95,positive_weight=1.,relative_probe=.001,relative_radius=.03,
            VAT_weight=.1,ramp_updates=444,power_iterations=1,start_epoch=21,
            all_U=True,EMA_model=False,extra_teacher_forwards=1,extra_student_forwards=1)


def unit_direction(x):
    dims=tuple(range(1,x.ndim));tiny=torch.finfo(x.dtype).tiny
    scaled=x/x.abs().amax(dims,keepdim=True).clamp_min(tiny)
    return scaled/scaled.square().sum(dims,keepdim=True).sqrt().clamp_min(tiny)


class Exposure:
    def __init__(self,metadata):
        rows=list(zip(*(metadata[k].tolist() for k in KEYS)))
        if len(set(rows))!=len(rows):raise ValueError('Unique physical source-U records required')
        self.lookup={row:i for i,row in enumerate(rows)};self.visits=torch.zeros(len(rows),dtype=torch.long)
        self.calls=self.seen=self.accepted=self.nonzero_directions=0;self.max_relative_radius=0.;self.probe_checks=0;self.by_rx={}

    def indices(self,batch,role,training):
        if role!='U_s' or not training or bool((batch[1]!=-1).any()) or bool(batch[3]['tx_label_visible'].any()):
            raise ValueError('VAT accepts hidden-label training U_s only')
        rows=list(zip(*(batch[3][k].tolist() for k in KEYS)))
        if len(set(rows))!=len(rows) or any(r not in self.lookup for r in rows):raise ValueError('Foreign or duplicate VAT physical records')
        return torch.tensor([self.lookup[r] for r in rows],dtype=torch.long)

    def state(self):
        return dict(calls=self.calls,seen=self.seen,accepted=self.accepted,unique_seen=int((self.visits>0).sum()),
            min_visits=int(self.visits.min()),max_visits=int(self.visits.max()),VAT_seen=self.seen,
            nonzero_directions=self.nonzero_directions,nonzero_direction_fraction=self.nonzero_directions/max(self.seen,1),
            max_relative_radius=self.max_relative_radius,probe_parameter_grad_and_buffer_checks=self.probe_checks,
            student_U_packets=2*self.seen,teacher_U_packets=2*self.seen,input_gradient_probes=self.calls,
            model_EMA_updates=0,U_truth_read=False,target_reads=0,new_inference_parameters=0,default_head_joint=True,
            by_source_RX=self.by_rx,algorithm='ordinary_PL_plus_all_U_VAT')


class Objective:
    def __init__(self,c):self.c=c;self.history=None;self.last={}

    def __call__(self,student,device):
        data=student._cd_U_loader.dataset
        if data.role!='U_s' or 'y' in data.metadata:raise ValueError('Only hidden source U can enter VAT')
        if self.history is None:self.history=Exposure(data.metadata)
        try:batch=next(student._cd_U_iter)
        except StopIteration:student._cd_U_iter=iter(student._cd_U_loader);batch=next(student._cd_U_iter)
        ids=self.history.indices(batch,data.role,student.training);x=batch[0].to(device);domain=batch[2].to(device);n=len(x)
        scale=x.flatten(1).norm(dim=1).reshape(-1,1,1)
        if bool((scale<=0).any()) or not bool(torch.isfinite(x).all()):raise ValueError('Finite nonzero source IQ required')
        check=self.history.probe_checks==0
        if check:
            buffers={k:v.detach().clone() for k,v in student.named_buffers()}
            grads={k:None if p.grad is None else p.grad.detach().clone() for k,p in student.named_parameters()}
        student.eval()
        try:
            with torch.no_grad():reference=student(x).softmax(1).detach()
            d=(RECIPE['relative_probe']*scale*unit_direction(torch.randn_like(x))).requires_grad_(True)
            q=student(x+d).log_softmax(1);probe=F.kl_div(q,reference,reduction='batchmean')
            gradient=torch.autograd.grad(probe,d,only_inputs=True,create_graph=False)[0]
            if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('Nonfinite VAT input direction')
            perturbation=(RECIPE['relative_radius']*scale*unit_direction(gradient.detach())).detach()
            if check:
                if any(not torch.equal(v,buffers[k]) for k,v in student.named_buffers()):raise ValueError('VAT probe changed buffers')
                for name,p in student.named_parameters():
                    old=grads[name]
                    if (old is None)!=(p.grad is None) or (old is not None and not torch.equal(old,p.grad)):
                        raise ValueError('Input-only VAT probe changed parameter gradients')
                self.history.probe_checks+=1
        finally:student.train()
        logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
        confidence,pseudo=reference.max(1);accepted=confidence>=.95
        positive=(F.cross_entropy(logits,pseudo,reduction='none')*accepted).mean()
        adversarial=student.forward_identity_only(x+perturbation,domain_labels=domain)['tx_logits']
        smooth=F.kl_div(adversarial.log_softmax(1),reference,reduction='batchmean')
        update=student._feature_l_steps+self.c.get('step_offset',0)-4440
        if update<=0:raise ValueError('Fixed VAT begins after original E20')
        relative=perturbation.flatten(1).norm(dim=1)/scale.flatten()
        if not bool(torch.isfinite(relative).all()) or float(relative.max())>RECIPE['relative_radius']*(1+1e-4):
            raise ValueError('VAT perturbation exceeds fixed IQ-relative radius')
        h=self.history;h.calls+=1;h.seen+=n;h.accepted+=int(accepted.sum());h.visits[ids]+=1
        h.nonzero_directions+=int((relative>0).sum());h.max_relative_radius=max(h.max_relative_radius,float(relative.max()))
        for rx in batch[3]['rx_i'].unique().tolist():
            mask=batch[3]['rx_i']==rx;counts=h.by_rx.setdefault(str(rx),dict(seen=0,accepted=0,VAT_seen=0))
            counts['seen']+=int(mask.sum());counts['VAT_seen']+=int(mask.sum());counts['accepted']+=int((mask.to(device)&accepted).sum())
        student._cd_U_seen+=n;student._cd_U_accepted+=int(accepted.sum())
        ramp=min(update/RECIPE['ramp_updates'],1.)
        self.last=dict(VAT_positive_loss=positive.detach(),VAT_smooth_loss=smooth.detach(),VAT_ramp=ramp,
            VAT_seen=h.seen,VAT_accepted=h.accepted,VAT_nonzero_directions=h.nonzero_directions,VAT_max_relative_radius=h.max_relative_radius)
        return positive+RECIPE['VAT_weight']*ramp*smooth
