"""Use current true-L raw features as detached cross-RX anchors for accepted U/H."""
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
import torch
from torch.nn import functional as F
from experiments.pamf_ssl_research.hard_pl_joint_objective import capture_joint
from .clean_h_objective import CleanHObjective,physical_mask

RECIPE=dict(weight=.05,CE_normalizer=1.25,temperature=.2,normalizer='whole physical half',
            anchors='two existing true-L raw joint160 forwards,detached',queries='two existing U H physical halves',
            positives='accepted raw95 pseudo TX equals true L TX, different physical RX',
            negatives='true L TX differs from accepted U pseudo TX',same_TX_same_RX='excluded',extra_F=0)


def anchor_gradient_norms(named,gradients):
    norms=dict(signature=0.,classification_weight=0.,other_features=0.)
    for (name,_),g in zip(named,gradients):
        if g is None:continue
        if not bool(torch.isfinite(g).all()):raise FloatingPointError('Finite H-anchor gradients required')
        # The original cls_head also owns feature-fusion norms/gain. Only its
        # final cosine class weight is outside the joint-feature loss graph.
        group='signature' if 'sequence_pool.' in name else 'classification_weight' if name.endswith('.cls_head.weight') or name=='cls_head.weight' else 'other_features'
        norms[group]+=float(g.detach().double().square().sum())
    return {k:v**.5 for k,v in norms.items()}


def anchored_contrast(query,pseudo,accepted,rx,keys,labels,key_rx):
    n=len(query);m=len(keys)
    if (query.shape!=(n,160) or keys.shape!=(m,160) or not n or not m
        or any(t.shape!=(n,) for t in (pseudo,accepted,rx)) or labels.shape!=(m,) or key_rx.shape!=(m,)
        or accepted.dtype!=torch.bool or any(t.dtype!=torch.long for t in (pseudo,rx,labels,key_rx))
        or pseudo.min()<0 or pseudo.max()>=6 or labels.min()<0 or labels.max()>=6):raise ValueError('Joint160, model pseudo TX, true-L labels and physical RX must align')
    if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(keys).all()):raise FloatingPointError('Finite source joint features required')
    same=pseudo[:,None]==labels[None,:];pos=same&(rx[:,None]!=key_rx[None,:]);neg=~same
    valid=accepted&pos.any(1)&neg.any(1)
    details=dict(queries=n,accepted=int(accepted.sum()),valid=int(valid.sum()),positive_pairs=int(pos[valid].sum()),negative_pairs=int(neg[valid].sum()))
    if not bool(valid.any()):return query.sum()*0.,details
    q=F.normalize(query[valid],dim=1,eps=1e-4);k=F.normalize(keys.detach(),dim=1,eps=1e-4)
    score=q@k.T/RECIPE['temperature'];p=pos[valid];count=p.sum(1)
    value=torch.logsumexp(score.masked_fill(~(pos|neg)[valid],-torch.inf),1)-(score*p).sum(1)/count-count.to(score.dtype).log()
    return value.sum()/n,details


class LabeledHAnchor(CleanHObjective):
    def __init__(self,c):
        if c.get('labeled_H_anchors')!=RECIPE:raise ValueError('Fixed true-L raw anchor intervention only')
        proxy=deepcopy(c);proxy['paper_method']='REPL_H2_LU_Pair';super().__init__(proxy);self.c=c
        self.L_keys=self.L_y=self.L_rx=self.H_queries=None;self.L_original=None
        self.LA_L_calls=self.LA_L_captures=self.LA_L_packets=self.LA_calls=self.LA_H_captures=self.LA_queries=self.LA_accepted=self.LA_valid=self.LA_positives=self.LA_negatives=0

    @contextmanager
    def labeled(self,student,kwargs):
        meta=kwargs['extra'][1];y=kwargs['labels']
        if self.L_keys is not None or y.shape!=(128,) or not bool(meta['tx_label_visible'].all()):raise ValueError('All true L labels and one current L context required')
        self.L_keys=[];self.L_original=[];self.L_y=y.detach();self.L_rx=meta['rx_i'].to(y.device).long().detach();index=0
        try:
            with super().labeled(student,kwargs):
                original=student.forward_identity_only
                def forward(this,value,*args,**kw):
                    nonlocal index
                    i=index
                    if index<5:index+=1
                    if i in (3,4):
                        with capture_joint(student.id_backbone.encoder) as captured:out=original(value,*args,**kw)
                        h=captured[0]
                        if h.shape!=(128,160):raise ValueError('Two existing raw L128 features required')
                        self.L_keys.append(h.detach());self.L_original.append(h);self.LA_L_captures+=1;self.LA_L_packets+=128
                        return out
                    return original(value,*args,**kw)
                student.forward_identity_only=MethodType(forward,student)
                try:yield
                finally:student.forward_identity_only=original
                if index!=5 or len(self.L_keys)!=2:raise ValueError('Capture only original fourth/fifth extra L raw forwards')
            self.LA_L_calls+=1
        finally:self.L_keys=self.L_y=self.L_rx=self.L_original=None

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        index=self._joint_forward_index+1
        if index not in (3,4):return super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        if self.H_queries is None:raise ValueError('H feature capture escaped current U loss')
        with capture_joint(student.id_backbone.encoder) as captured:logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        mask=physical_mask(len(x),student._feature_l_steps+self.c['step_offset'],x.device)
        self.H_queries.append((captured[0][mask],pseudo[mask],accepted[mask],metadata['rx_i'].to(x.device).long()[mask]))
        self.LA_H_captures+=1;return logits

    def loss(self,student,device):
        if self.H_queries is not None or self.L_keys is None or len(self.L_keys)!=2:raise ValueError('Two actual current true-L raw anchors precede U loss')
        self.H_queries=[]
        try:
            zero=super().loss(student,device)
            if len(self.H_queries)!=2 or self.pending_gradient_loss is None:raise ValueError('Two existing physical H feature views and guarded U loss required')
            keys=torch.cat(self.L_keys);labels=self.L_y.repeat(2);rx=self.L_rx.repeat(2);values=[]
            for q,p,a,r in self.H_queries:
                value,detail=anchored_contrast(q,p,a,r,keys,labels,rx);values.append(value)
                self.LA_queries+=detail['queries'];self.LA_accepted+=detail['accepted'];self.LA_valid+=detail['valid'];self.LA_positives+=detail['positive_pairs'];self.LA_negatives+=detail['negative_pairs']
            aux=torch.stack(values).mean()
            if self.c.get('candidate_smoke'):
                named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
                gradients=torch.autograd.grad(aux,[p for _,p in named],retain_graph=True,allow_unused=True);norms=anchor_gradient_norms(named,gradients)
                lg=torch.autograd.grad(aux,self.L_original,retain_graph=True,allow_unused=True)
                if norms['signature']<=0 or norms['other_features']<=0 or norms['classification_weight']!=0 or any(g is not None and bool(g.any()) for g in lg) or not self.LA_valid:raise ValueError('H queries train deployed features; L anchors detached; no replacement head')
                self.smoke_anchor_gradient_norms=norms
            self.pending_gradient_loss=self.pending_gradient_loss+self.settings['weight']*RECIPE['weight']/RECIPE['CE_normalizer']*aux;self.LA_calls+=1
            self.metrics.update(LA_calls=self.LA_calls,LA_L_raw_captures=self.LA_L_captures,LA_L_raw_packets=self.LA_L_packets,LA_H_captures=self.LA_H_captures,
                LA_queries=self.LA_queries,LA_accepted=self.LA_accepted,LA_valid=self.LA_valid,LA_positive_pairs=self.LA_positives,LA_negative_pairs=self.LA_negatives,LA_loss=aux.detach(),LA_weight=RECIPE['weight'],
                HPL_loss=self.pending_gradient_loss.detach(),HPL_base_loss=(self.pending_gradient_loss/self.settings['weight']).detach())
            return zero
        finally:self.H_queries=None

    def guide(self,labeled,student):
        value=super().guide(labeled,student);self.metrics['LA_L_calls']=self.LA_L_calls;return value

    def state_dict(self):
        state=super().state_dict()
        if (self.L_keys is not None or self.H_queries is not None or self.LA_calls!=self.calls or self.LA_L_calls!=self.calls
            or self.LA_L_captures!=2*self.calls or self.LA_L_packets!=256*self.calls or self.LA_H_captures!=2*self.calls or self.LA_queries!=self.seen
            or not 0<=self.LA_valid<=self.LA_accepted<=self.LA_queries):raise ValueError('Complete true-L/H capture lifetime and original forward budget required')
        state['L_anchor_H']=dict(recipe=RECIPE,calls=self.LA_calls,L_calls=self.LA_L_calls,L_raw_captures=self.LA_L_captures,L_raw_packets=self.LA_L_packets,
            H_captures=self.LA_H_captures,H_queries=self.LA_queries,accepted_queries=self.LA_accepted,valid_queries=self.LA_valid,positive_pairs=self.LA_positives,negative_pairs=self.LA_negatives,
            anchors_detached=True,extra_F_forwards=0,new_inference_parameters=0,U_truth_reads=0)
        return state
