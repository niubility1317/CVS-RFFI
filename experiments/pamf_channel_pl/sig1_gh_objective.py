"""Separate source channel and restored U domains; optional rejected physical pairs."""
import torch
from torch.nn import functional as F
from .sig1_role_objective import RestoredRoles,mask

METHODS=('REPL_GH_U','REPL_GH_U_Pair')
ROLES={method:(False,True) for method in METHODS}
PAIR_WEIGHT=6.*.25/1.25


def rejected_peer_loss(a,b,rejected):
    if a.ndim!=2 or a.shape!=b.shape or a.shape[1]!=6 or rejected.shape!=(len(a),) or rejected.dtype!=torch.bool or not len(a):
        raise ValueError('Aligned original-head same-packet logits and raw-rejection mask required')
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):raise ValueError('Finite G logits required')
    pa,pb=a.softmax(1).detach(),b.softmax(1).detach()
    # Targets stop gradients independently; no confidence sharpening or hard TX.
    loss=.5*(F.kl_div(a.log_softmax(1),pb,reduction='none').sum(1)+F.kl_div(b.log_softmax(1),pa,reduction='none').sum(1))
    return (loss*rejected).mean()


class GHObjective(RestoredRoles):
    def __init__(self,c,g,engine):
        if c['paper_method'] not in METHODS or c['hard_pl_no_uas'] is not False:raise ValueError('Two explicit G/H U methods only')
        # Identical existing G-U role; only the physical mixed half becomes H.
        super().__init__(dict(c,paper_method='REPL_G_U'),g,engine);self.c=c
        self.pair_enabled=c['paper_method']=='REPL_GH_U_Pair';self.pair_captures=None
        self.pair_calls=self.pair_seen=self.pair_rejected=self.H_calls=self.H_packets=0

    def mixed_view(self,student,x,ordinary,physical,metadata,epoch,step,pseudo,accepted):
        value=super().mixed_view(student,x,ordinary,physical,metadata,epoch,step,pseudo,accepted)
        if self.c.get('candidate_smoke'):
            self.smoke_H_relative_difference=float((value[physical]-x[physical]).norm()/x[physical].norm().clamp_min(1e-12))
            if self.smoke_H_relative_difference<=1e-7:raise ValueError('Actual source H must differ from raw U')
        self.H_calls+=1;self.H_packets+=int(physical.sum());return value

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        index=self._joint_forward_index+1
        logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        if index in (3,4):
            if self.pair_captures is None:raise ValueError('G pair escaped ordered U objective')
            step=student._feature_l_steps+self.c['step_offset'];ordinary=~mask(len(x),step,x.device)
            self.pair_captures.append((logits[ordinary],(~accepted[ordinary]).detach()))
        return logits

    def loss(self,student,device):
        if self.pair_captures is not None:raise ValueError('Nested paired objective')
        self.pair_captures=[]
        try:
            value=super().loss(student,device)
            if len(self.pair_captures)!=2 or self.pending_gradient_loss is None:raise ValueError('Original guarded loss plus two existing G logits required')
            (a,rejected),(b,second)=self.pair_captures
            if not torch.equal(rejected,second):raise ValueError('Both G views require the same physical raw gate')
            extra=rejected_peer_loss(a,b,rejected) if self.pair_enabled else (a.sum()+b.sum())*0.
            if self.c.get('candidate_smoke') and self.pair_enabled:
                named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
                grads=torch.autograd.grad(extra,[p for _,p in named],retain_graph=True,allow_unused=True)
                norms=dict(signature=0.,head=0.,other=0.)
                for (n,_),v in zip(named,grads):
                    if v is None:continue
                    if not bool(torch.isfinite(v).all()):raise ValueError('Finite actual rejected-pair gradients required')
                    key='signature' if 'sequence_pool.' in n else 'head' if '.cls_head.' in n else 'other';norms[key]+=float(v.detach().double().square().sum())
                if min(norms.values())<=0 or not bool(rejected.any()):raise ValueError('Actual rejected G pair must train deployed signature/head/body')
                self.smoke_rejected_pair_gradient_norms={k:v**.5 for k,v in norms.items()}
            self.pending_gradient_loss=self.pending_gradient_loss+PAIR_WEIGHT*extra
            self.pair_calls+=1;self.pair_seen+=len(rejected);self.pair_rejected+=int(rejected.sum())
            self.metrics.update(GH_H_calls=self.H_calls,GH_H_packets=self.H_packets,GH_pair_calls=self.pair_calls,
                GH_pair_seen=self.pair_seen,GH_pair_rejected=self.pair_rejected,GH_pair_enabled=int(self.pair_enabled),
                GH_pair_loss=extra.detach(),GH_pair_weight=PAIR_WEIGHT if self.pair_enabled else 0.,
                HPL_loss=self.pending_gradient_loss.detach(),HPL_base_loss=(self.pending_gradient_loss/self.settings['weight']).detach())
            return value
        finally:self.pair_captures=None

    def state_dict(self):
        state=super().state_dict()
        if (self.pair_captures is not None or self.H_calls!=self.calls or self.H_packets*2!=self.seen or self.pair_calls!=self.calls
                or self.pair_seen*2!=self.seen or self.physical_seen!=self.H_packets or self.raw_As_seen!=0):
            raise ValueError('Exactly one H/G half-batch per original six-view U update required')
        state['restored_roles'].pop('As_and_raw_halves_preserved')
        state['restored_roles'].update(original_L_As_and_U_H_halves_preserved=True,U_physical_half='unrestored source practical H',
            lowconfidence_extra_loss=self.pair_enabled)
        state['GH']=dict(H_calls=self.H_calls,H_packets=self.H_packets,physical_H_student_views=self.physical_student_views,
            pair_calls=self.pair_calls,pair_packets=self.pair_seen,pair_raw_rejected=self.pair_rejected,pair_enabled=self.pair_enabled,
            pair_weight=PAIR_WEIGHT if self.pair_enabled else 0.,pair_targets='detached opposite G probabilities; raw95 rejected only; no hard TX',
            pair_normalizer='whole original G ordinary half, not accepted-only',pair_extra_F_forwards=0,pair_extra_G_forwards=0,
            guard='combined existing accepted-U plus rejected-G pair under unchanged U/L cap1')
        return state
