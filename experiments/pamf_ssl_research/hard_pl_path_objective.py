"""Use weaker existing feature paths without deleting their deployment contributions."""
import torch
from torch.nn import functional as F
from .hard_pl_joint_objective import JointNoAsMixStyle
from .hard_pl_guard_objective import Guarded


def ce(logits,pseudo,accepted):
    return (F.cross_entropy(logits,pseudo,reduction='none',label_smoothing=.01)*accepted).mean()


class PathComplement(JointNoAsMixStyle):
    def __init__(self,c):
        super().__init__(c);self.path_calls=self.path_head_calls=self.path_packets=0;self.path_active=False

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        self._path_forward_index+=1
        if self._path_forward_index>2:return super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        signature=[];response=[];reference=student.id_backbone.encoder;self.path_active=True
        handles=[reference.sequence_pool.register_forward_hook(lambda module,inputs,output:signature.append(output)),
            reference.response_projection.register_forward_hook(lambda module,inputs,output:response.append(output))]
        try:logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        finally:
            for handle in handles:handle.remove()
            self.path_active=False
        if len(signature)!=1 or len(response)!=1 or any(v[0].shape!=(len(x),160) for v in (signature,response)):
            raise ValueError('One same-pass actual signature and response contribution required')
        joint=self._joint_current[-1][0];no_signature=joint-signature[0]
        core=no_signature-response[0]*reference.response_gain.tanh()
        if self.c.get('hard_pl_smoke'):no_signature.retain_grad();core.retain_grad()
        self._path_current.append((no_signature,core,logits,pseudo,accepted));return logits

    def loss(self,student,device):
        self._path_current=[];self._path_forward_index=0
        try:
            value=super().loss(student,device)
            if self._path_forward_index!=6 or len(self._path_current)!=2:raise ValueError('Original6U and two existing feature pairs required')
            ordinary=[];no_signature=[];core=[]
            for h,k,logits,pseudo,accepted in self._path_current:
                ordinary.append(ce(logits,pseudo,accepted))
                no_signature.append(ce(student.id_backbone.encoder.classify_features(h),pseudo,accepted))
                core.append(ce(student.id_backbone.encoder.classify_features(k),pseudo,accepted))
            ordinary=torch.stack(ordinary).mean();no_signature=torch.stack(no_signature).mean();core=torch.stack(core).mean()
            recipe=self.c['hard_pl_path'];coefficient=recipe['core_CE_coefficient']
            value=value+self.settings['weight']*coefficient/recipe['CE_normalizer']*(no_signature+core-2*ordinary)
            self.path_calls+=1;self.path_head_calls+=4;self.path_packets+=4*self._path_current[0][0].shape[0]
            self.metrics.update(HPL_loss=value.detach(),HPL_base_loss=(value/self.settings['weight']).detach(),
                HPL_path_calls=self.path_calls,HPL_path_head_calls=self.path_head_calls,HPL_path_packets=self.path_packets,
                HPL_path_no_signature_CE=no_signature.detach(),HPL_path_core_CE=core.detach(),HPL_path_ordinary_CE=ordinary.detach(),
                HPL_path_reallocated_CE_coefficient=.125)
            if self.c.get('hard_pl_smoke'):
                named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
                gradients=torch.autograd.grad(core,[p for _,p in named],retain_graph=True,allow_unused=True)
                norms=dict(signature=0.,response=0.,head=0.,core=0.)
                for (name,_),g in zip(named,gradients):
                    if g is None:continue
                    if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite core-only pseudo CE gradient')
                    group=('signature' if '.sequence_pool.' in name else 'response' if '.response_projection.' in name or name.endswith('.response_gain')
                        else 'head' if '.cls_head.' in name else 'core')
                    norms[group]+=float(g.detach().double().square().sum())
                norms={k:v**.5 for k,v in norms.items()}
                if max(norms['signature'],norms['response'])>1e-7 or min(norms['head'],norms['core'])<=0:
                    raise ValueError('Core-only CE must bypass both exclusive branches while training existing core/head')
                self.smoke_path_gradients=norms;self.smoke_path_features=[v for item in self._path_current for v in item[:2]]
            return value
        finally:self._path_current=None

    def state_dict(self):
        state=super().state_dict()
        if (self.path_active or self.path_calls!=self.calls or self.path_head_calls!=4*self.calls
            or self.path_packets!=4*self.seen or getattr(self,'_path_current',None) is not None):
            raise ValueError('Path CE must reuse exactly two current forward graphs and four existing-head views')
        state['U_path_complement']=dict(recipe=self.c['hard_pl_path'],calls=self.path_calls,
            extra_head_calls=self.path_head_calls,extra_head_packet_views=self.path_packets,
            extra_backbone_forwards=0,teacher_or_L_head_calls=0,new_head=False,persistent_hooks=False,
            full_deployment_signature_retained=True,full_deployment_response_retained=True,CE_dose_normalizer=1.25)
        return state


class GuardedPathComplement(Guarded,PathComplement):pass
