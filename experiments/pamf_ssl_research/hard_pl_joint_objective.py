"""No new forward/head: contrast the same joint160 used by deployed F."""
from contextlib import contextmanager
from types import MethodType
import torch
from experiments.cric_dura_comparison.f_model import contrast
from .hard_pl_no_as_mixstyle_objective import NoAsMixStyleObjective
from .hard_pl_guard_objective import Guarded


@contextmanager
def capture_joint(reference):
    had='features' in reference.__dict__;previous=reference.__dict__.get('features');original=reference.features;captured=[]
    def features(self,x):
        value=original(x)
        if value.ndim!=2 or value.shape!=(len(x),160):raise ValueError('Actual CompositeReference joint160 required')
        captured.append(value);return value
    reference.features=MethodType(features,reference)
    try:
        yield captured
        if len(captured)!=1:raise ValueError('Exactly one actual joint160 per existing U student forward required')
    finally:
        if had:reference.features=previous
        else:delattr(reference,'features')


class JointNoAsMixStyle(NoAsMixStyleObjective):
    def __init__(self,c):
        super().__init__(c);self.joint_calls=self.joint_captures=self.joint_packets=self.joint_anchors=0;self.joint_active=False

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        if self._joint_current is None:raise ValueError('Joint feature capture escaped U loss')
        self._joint_forward_index+=1
        if self._joint_forward_index>2:return super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        self.joint_active=True
        try:
            with capture_joint(student.id_backbone.encoder) as current:
                logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
            h=current[0]
            if self.c.get('hard_pl_smoke'):h.retain_grad()
            self._joint_current.append((h,pseudo,accepted,metadata['rx_i'].to(x.device).long()))
        finally:self.joint_active=False
        return logits

    def loss(self,student,device):
        self._joint_current=[];self._joint_forward_index=0
        try:
            value=super().loss(student,device)
            if self._joint_forward_index!=6 or len(self._joint_current)!=2:raise ValueError('Six original U forwards,two actual joint captures required')
            values=[];anchors=0
            for h,pseudo,accepted,rx in self._joint_current:
                if bool(accepted.any()):
                    aux,detail=contrast({'joint160':h[accepted]},pseudo[accepted],rx[accepted]);anchors+=detail[0]['valid_anchors']
                else:aux=h.sum()*0.
                values.append(aux)
            aux=torch.stack(values).mean();recipe=self.c['hard_pl_joint_contrast'];extra=recipe['weight']/recipe['CE_normalizer']*aux
            value=value+self.settings['weight']*extra
            self.joint_calls+=1;self.joint_captures+=2;self.joint_packets+=sum(len(v[0]) for v in self._joint_current);self.joint_anchors+=anchors
            self.metrics.update(HPL_loss=value.detach(),HPL_base_loss=(value/self.settings['weight']).detach(),
                HPL_joint_loss=aux.detach(),HPL_joint_weight=recipe['weight'],HPL_joint_calls=self.joint_calls,
                HPL_joint_captures=self.joint_captures,HPL_joint_packets=self.joint_packets,HPL_joint_anchors=self.joint_anchors)
            if self.c.get('hard_pl_smoke'):
                self.smoke_joint_features=[v[0] for v in self._joint_current]
                named=[(n,p) for n,p in student.named_parameters() if p.requires_grad and '.sequence_pool.' in n]
                if not named:raise ValueError('Deployed learned SIG_TIME_X3 branch not found')
                gradients=torch.autograd.grad(aux,[p for _,p in named],retain_graph=True,allow_unused=True)
                if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):raise ValueError('Nonfinite U joint signature gradient')
                norm=sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)**.5
                if norm<=0:raise ValueError('U joint objective must actually train learned signature branch')
                self.smoke_joint_signature_gradient=norm
            return value
        finally:self._joint_current=None

    def state_dict(self):
        state=super().state_dict()
        if self.joint_active or self.joint_calls!=self.calls or self.joint_captures!=2*self.calls or self.joint_packets!=2*self.seen:
            raise ValueError('Actual joint objective/capture lifetime differs')
        state['U_joint_contrast']=dict(recipe=self.c['hard_pl_joint_contrast'],calls=self.joint_calls,
            captures=self.joint_captures,captured_packets=self.joint_packets,anchors=self.joint_anchors,
            extra_forward_calls=0,teacher_or_L_captures=0,persistent_feature_wrapper=False)
        return state


class GuardedJointNoAs(Guarded,JointNoAsMixStyle):pass
