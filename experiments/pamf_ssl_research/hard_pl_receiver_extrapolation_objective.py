"""Extrapolate full deployed features;keep unpaired samples' original CE unchanged."""
import torch
from torch.nn import functional as F
from .hard_pl_joint_objective import JointNoAsMixStyle
from .hard_pl_guard_objective import Guarded
from .source_receiver_extrapolation import extrapolate


class ReceiverExtrapolation(JointNoAsMixStyle):
    def __init__(self,c):
        super().__init__(c);self.rxe_calls=self.rxe_heads=self.rxe_packets=self.rxe_eligible=0;self._rxe_current=None

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        self._rxe_forward_index+=1
        logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        if self._rxe_forward_index<=2:
            h=self._joint_current[-1][0];partner,valid,_,_=self.shared_plan
            self._rxe_current.append((h,logits,pseudo,accepted,metadata['rx_i'].to(x.device).long(),partner,valid))
        return logits

    def loss(self,student,device):
        self._rxe_current=[];self._rxe_forward_index=0
        try:
            value=super().loss(student,device)
            if self._rxe_forward_index!=6 or len(self._rxe_current)!=2:raise ValueError('Six original U passes and two existing fulljoint graphs required')
            ordinary=[];extra=[];features=[];eligible=0
            recipe=self.c['hard_pl_receiver_extrapolation']
            for h,logits,pseudo,accepted,rx,partner,valid in self._rxe_current:
                z=extrapolate(h,pseudo,accepted,rx,partner,valid,recipe['alpha'])
                out=student.id_backbone.encoder.classify_features(z);mask=(accepted&valid).to(out.dtype)
                ordinary.append((F.cross_entropy(logits,pseudo,reduction='none',label_smoothing=.01)*mask).mean())
                extra.append((F.cross_entropy(out,pseudo,reduction='none',label_smoothing=.01)*mask).mean())
                eligible+=int(valid.sum());features.append(z)
                if self.c.get('hard_pl_smoke'):z.retain_grad()
            old=torch.stack(ordinary).mean();new=torch.stack(extra).mean()
            value=value+self.settings['weight']*recipe['extrapolated_CE_coefficient']/recipe['CE_normalizer']*(new-old)
            self.rxe_calls+=1;self.rxe_heads+=2;self.rxe_packets+=sum(len(v[0]) for v in self._rxe_current);self.rxe_eligible+=eligible
            self.metrics.update(HPL_loss=value.detach(),HPL_base_loss=(value/self.settings['weight']).detach(),
                HPL_RXE_calls=self.rxe_calls,HPL_RXE_head_calls=self.rxe_heads,HPL_RXE_packets=self.rxe_packets,HPL_RXE_eligible_views=self.rxe_eligible,
                HPL_RXE_alpha=recipe['alpha'],HPL_RXE_ordinary_valid_CE=old.detach(),HPL_RXE_extrapolated_valid_CE=new.detach(),
                HPL_RXE_reallocated_CE_coefficient=.25,HPL_RXE_CE_normalizer=1.25)
            if self.c.get('hard_pl_smoke'):
                named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
                gradients=torch.autograd.grad(new,[p for _,p in named],retain_graph=True,allow_unused=True)
                norms=dict(signature=0.,response=0.,head=0.,core=0.)
                for (name,_),g in zip(named,gradients):
                    if g is None:continue
                    if not bool(torch.isfinite(g).all()):raise ValueError('Finite full-feature extrapolated CE gradients required')
                    group=('signature' if '.sequence_pool.' in name else 'response' if '.response_projection.' in name or name.endswith('.response_gain')
                        else 'head' if '.cls_head.' in name else 'core')
                    norms[group]+=float(g.detach().double().square().sum())
                norms={k:v**.5 for k,v in norms.items()}
                if min(norms.values())<=0:raise ValueError('New full-feature CE must retain signature,response,core and existing-head gradients')
                self.smoke_rxe_gradients=norms;self.smoke_rxe_features=features
            return value
        finally:self._rxe_current=None

    def state_dict(self):
        s=super().state_dict()
        if (self._rxe_current is not None or self.rxe_calls!=self.calls or self.rxe_heads!=2*self.calls
            or self.rxe_packets!=2*self.seen or not 0<=self.rxe_eligible<=2*self.accepted):
            raise ValueError('Exactly two existing full-feature heads and accepted crossRX pairs per U loss required')
        s['U_receiver_extrapolation']=dict(recipe=self.c['hard_pl_receiver_extrapolation'],calls=self.rxe_calls,
            extra_head_calls=self.rxe_heads,extra_head_packet_views=self.rxe_packets,eligible_packet_views=self.rxe_eligible,
            extra_backbone_forwards=0,teacher_or_L_head_calls=0,new_head=False,persistent_hooks=False,
            all_raw_accepted_retained=True,full_deployment_signature_retained=True,full_deployment_response_retained=True,CE_dose_normalizer=1.25)
        return s


class GuardedReceiverExtrapolation(Guarded,ReceiverExtrapolation):pass
