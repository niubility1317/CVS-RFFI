"""Move existing ordinaryCE dose to a weaker feature path;retain full deployment fusion."""
import torch
from torch.nn import functional as F
from .hard_pl_joint_objective import JointNoAsMixStyle
from .hard_pl_guard_objective import Guarded


def ce(logits,pseudo,accepted):
    return (F.cross_entropy(logits,pseudo,reduction='none',label_smoothing=.01)*accepted).mean()


class Complement(JointNoAsMixStyle):
    def __init__(self,c):
        super().__init__(c);self.complement_calls=self.complement_head_calls=self.complement_packets=0;self.complement_active=False

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        self._complement_forward_index+=1
        if self._complement_forward_index>2:return super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        values=[];reference=student.id_backbone.encoder;self.complement_active=True
        handle=reference.sequence_pool.register_forward_hook(lambda module,inputs,output:values.append(output))
        try:logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        finally:handle.remove();self.complement_active=False
        if len(values)!=1 or values[0].shape!=(len(x),160):raise ValueError('One actual same-pass sequence contribution required')
        joint=self._joint_current[-1][0];masked=joint-values[0]
        if self.c.get('hard_pl_smoke'):masked.retain_grad()
        self._complement_current.append((masked,logits,pseudo,accepted));return logits

    def loss(self,student,device):
        self._complement_current=[];self._complement_forward_index=0
        try:
            value=super().loss(student,device)
            if self._complement_forward_index!=6 or len(self._complement_current)!=2:raise ValueError('Original6U forwards,two complementary head views required')
            ordinary=[];masked=[]
            for h,logits,pseudo,accepted in self._complement_current:
                ordinary.append(ce(logits,pseudo,accepted))
                masked.append(ce(student.id_backbone.encoder.classify_features(h),pseudo,accepted))
            ordinary=torch.stack(ordinary).mean();masked=torch.stack(masked).mean();recipe=self.c['hard_pl_complement']
            value=value+self.settings['weight']*recipe['masked_CE_coefficient']/recipe['CE_normalizer']*(masked-ordinary)
            self.complement_calls+=1;self.complement_head_calls+=2;self.complement_packets+=sum(len(v[0]) for v in self._complement_current)
            self.metrics.update(HPL_loss=value.detach(),HPL_base_loss=(value/self.settings['weight']).detach(),
                HPL_complement_calls=self.complement_calls,HPL_complement_head_calls=self.complement_head_calls,
                HPL_complement_packets=self.complement_packets,HPL_complement_masked_CE=masked.detach(),
                HPL_complement_ordinary_CE=ordinary.detach(),HPL_complement_reallocated_CE_coefficient=.25)
            if self.c.get('hard_pl_smoke'):
                named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
                gradients=torch.autograd.grad(masked,[p for _,p in named],retain_graph=True,allow_unused=True)
                norms=dict(signature=0.,head=0.,core=0.)
                for (name,_),g in zip(named,gradients):
                    if g is None:continue
                    if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite complementary gradient')
                    group='signature' if '.sequence_pool.' in name else 'head' if '.cls_head.' in name else 'core'
                    norms[group]+=float(g.detach().double().square().sum())
                norms={k:v**.5 for k,v in norms.items()}
                if norms['signature']>1e-7 or norms['head']<=0 or norms['core']<=0:
                    raise ValueError('Masked CE must bypass signature while training existing core/default head')
                self.smoke_complement_gradients=norms;self.smoke_complement_features=[v[0] for v in self._complement_current]
            return value
        finally:self._complement_current=None

    def state_dict(self):
        state=super().state_dict()
        if (self.complement_active or self.complement_calls!=self.calls or self.complement_head_calls!=2*self.calls
            or self.complement_packets!=2*self.seen or getattr(self,'_complement_current',None) is not None):
            raise ValueError('Complementary head calls must reuse exactly two existing U forward graphs')
        state['U_complement']=dict(recipe=self.c['hard_pl_complement'],calls=self.complement_calls,
            extra_head_calls=self.complement_head_calls,extra_head_packet_views=self.complement_packets,
            extra_backbone_forwards=0,teacher_or_L_head_calls=0,new_head=False,persistent_hooks=False,
            full_deployment_signature_retained=True,CE_dose_normalizer=1.25)
        return state


class GuardedComplement(Guarded,Complement):pass
