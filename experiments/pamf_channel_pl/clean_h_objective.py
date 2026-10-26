"""REPL using only clean and direct practical-channel IQ, with physical U pairs."""
from collections import Counter
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
from experiments.cric_dura_comparison.io import read
from .restored_source import SCENES,source_channel
from .sig1_repl_offset_objective import NoStyleNoRXE

METHODS=('REPL_H2_U','REPL_H2_U_Pair','REPL_H2_LU_Pair')
PAIR_WEIGHT=1.2


def physical_mask(n,step,device):
    if n<=0 or n%2 or type(step) is not int or step<=0:raise ValueError('Even original batch and positive step required')
    return (torch.arange(n,device=device)+step).remainder(2).bool()


def view_plan(step,role,view):
    if type(step) is not int or step<=4440 or role not in ('L','U') or view not in ('a','b'):raise ValueError('Source E21+ physical views only')
    return SCENES[(step-1+int(view=='b'))%3],'PAMF/clean_H/'+role+'/E'+str((step-1)//222+1)+'/'+view


def physical_peer_loss(a,b,rejected):
    if a.ndim!=2 or a.shape!=b.shape or a.shape[1]!=6 or not len(a) or rejected.shape!=(len(a),) or rejected.dtype!=torch.bool:
        raise ValueError('Aligned same-packet original-head logits and raw rejection mask required')
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):raise ValueError('Finite H logits required')
    pa,pb=a.softmax(1).detach(),b.softmax(1).detach()
    loss=.5*(F.kl_div(a.log_softmax(1),pb,reduction='none').sum(1)+F.kl_div(b.log_softmax(1),pa,reduction='none').sum(1))
    return (loss*rejected).mean()


class CleanHObjective(NoStyleNoRXE):
    def __init__(self,c):
        super().__init__(c)
        if c['paper_method'] not in METHODS or c['hard_pl_no_uas'] is not False:raise ValueError('Fixed clean/H methods only')
        self.dual_L=c['paper_method']=='REPL_H2_LU_Pair';self.pair_enabled=c['paper_method']!='REPL_H2_U'
        self.contract=read(c['source_contract']);self.current_U=None;self.pair_capture=None
        self.L_calls=self.H_L_calls=self.H_U_calls=self.H_L_packets=self.H_U_packets=0
        self.pair_calls=self.pair_packets=self.pair_rejected=0;self.scenes=Counter();self.active_L=False;self.last_H={};self.last_diversity={}

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch)
        if bool(batch[3]['tx_label_visible'].any()):raise ValueError('U TX must stay hidden')
        self.current_U=batch[0]

    def H(self,raw,metadata,physical,step,role,view):
        if bool(metadata['tx_label_visible'].all())!=(role=='L') or (role=='U' and bool(metadata['tx_label_visible'].any())):raise ValueError('H source role visibility differs')
        n=len(raw);sub={k:v[physical.to(v.device)] if torch.is_tensor(v) and v.ndim and len(v)==n else v for k,v in metadata.items()}
        scene,namespace=view_plan(step,role,view)
        with torch.no_grad():out=source_channel(raw[physical].detach(),sub,scene,self.contract,namespace=namespace).detach()
        if out.shape!=(n//2,2,256) or not bool(torch.isfinite(out).all()):raise ValueError('Finite source H half-batch required')
        self.scenes[role+'::'+scene]+=len(out)
        if role=='L':self.H_L_calls+=1;self.H_L_packets+=len(out)
        else:self.H_U_calls+=1;self.H_U_packets+=len(out)
        if self.c.get('candidate_smoke'):
            if view=='a':self.last_H[role]=out.clone()
            else:
                delta=float((out-self.last_H[role]).norm()/self.last_H[role].norm().clamp_min(1e-12))
                if delta<=1e-7:raise ValueError('Two actual H views must differ')
                self.last_diversity[role]=delta
        return out

    @contextmanager
    def labeled(self,student,kwargs):
        if self.active_L or kwargs['labeled_count']!=128:raise ValueError('One ordered full-L context required')
        self.active_L=True;index=0;original=student.forward_identity_only;practical=student._cd_practical
        raw=kwargs['labeled_inputs'].detach();meta=kwargs['extra'][1];step=kwargs['optimizer_step']+self.c['step_offset'];physical=physical_mask(128,step,raw.device)
        def forward(this,value,*args,**kw):
            nonlocal index
            if index<5:
                i=index;index+=1
                if len(value)!=128:raise ValueError('Original five L128 extra forwards required')
                if self.dual_L and i in (1,2):
                    value=value.clone();value[physical]=self.H(raw,meta,physical,step,'L','a' if i==1 else 'b')
            return original(value,*args,**kw)
        student.forward_identity_only=MethodType(forward,student)
        # The original L mixer otherwise generates an unused shared H before replacement.
        if self.dual_L:student._cd_practical=lambda x,*args:x.detach().clone()
        try:
            yield
            if index!=5:raise ValueError('Original ordinary/mixed/raw L order differs')
            self.L_calls+=1
        finally:student.forward_identity_only=original;student._cd_practical=practical;self.active_L=False

    def mixed_view(self,student,x,ordinary,physical,metadata,epoch,step,pseudo,accepted):
        # Each H is generated at its existing mixed forward; no unused extra H.
        value=ordinary.clone();value[physical]=x[physical];return value

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        index=self._joint_forward_index+1;physical=None
        if index in (3,4):
            if self.current_U is None or len(self.current_U)!=len(x):raise ValueError('Aligned current raw U required')
            step=student._feature_l_steps+self.c['step_offset'];physical=physical_mask(len(x),step,x.device);raw=self.current_U.to(x.device)
            x=x.clone();x[physical]=self.H(raw,metadata,physical,step,'U','a' if index==3 else 'b')
        logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        if physical is not None:
            if self.pair_capture is None:raise ValueError('Physical H pair escaped current U loss')
            self.pair_capture.append((logits[physical],(~accepted[physical]).detach()))
        return logits

    def loss(self,student,device):
        if self.pair_capture is not None:raise ValueError('Nested physical U pair')
        self.pair_capture=[]
        try:
            zero=super().loss(student,device)
            if len(self.pair_capture)!=2 or self.pending_gradient_loss is None:raise ValueError('Two existing H logits and guarded U loss required')
            (a,mask),(b,second)=self.pair_capture
            if not torch.equal(mask,second):raise ValueError('Physical pair must share raw95 gate')
            extra=physical_peer_loss(a,b,mask) if self.pair_enabled else (a.sum()+b.sum())*0.
            if self.c.get('candidate_smoke') and self.pair_enabled:
                named=[(n,p) for n,p in student.named_parameters() if p.requires_grad];grads=torch.autograd.grad(extra,[p for _,p in named],retain_graph=True,allow_unused=True)
                norms=dict(signature=0.,head=0.,other=0.)
                for (n,_),v in zip(named,grads):
                    if v is None:continue
                    if not bool(torch.isfinite(v).all()):raise ValueError('Finite H-pair gradients required')
                    key='signature' if 'sequence_pool.' in n else 'head' if '.cls_head.' in n else 'other';norms[key]+=float(v.detach().double().square().sum())
                if min(norms.values())<=0 or not bool(mask.any()):raise ValueError('Rejected H pair must train deployed signature/head/body')
                self.smoke_pair_gradient_norms={k:v**.5 for k,v in norms.items()}
            self.pending_gradient_loss=self.pending_gradient_loss+PAIR_WEIGHT*extra
            self.pair_calls+=1;self.pair_packets+=len(mask);self.pair_rejected+=int(mask.sum())
            self.metrics.update(HV_L_calls=self.L_calls,HV_H_L_calls=self.H_L_calls,HV_H_L_packets=self.H_L_packets,
                HV_H_U_calls=self.H_U_calls,HV_H_U_packets=self.H_U_packets,HV_pair_calls=self.pair_calls,HV_pair_packets=self.pair_packets,
                HV_pair_rejected=self.pair_rejected,HV_pair_enabled=int(self.pair_enabled),HV_pair_weight=PAIR_WEIGHT if self.pair_enabled else 0.,
                HV_pair_loss=extra.detach(),HPL_loss=self.pending_gradient_loss.detach(),HPL_base_loss=(self.pending_gradient_loss/self.settings['weight']).detach())
            return zero
        finally:self.pair_capture=None

    def state_dict(self):
        state=super().state_dict()
        if (self.active_L or self.pair_capture is not None or self.L_calls!=self.calls or self.H_U_calls!=2*self.calls or self.H_U_packets!=self.seen
                or self.H_L_calls!=2*self.calls*int(self.dual_L) or self.H_L_packets!=128*self.calls*int(self.dual_L)
                or self.pair_calls!=self.calls or self.pair_packets*2!=self.seen or self.physical_seen*2!=self.seen or self.raw_As_seen!=0):
            raise ValueError('Original full-L/six-U budget and physical H counts differ')
        state['H2']=dict(dual_L=self.dual_L,L_calls=self.L_calls,H_L_calls=self.H_L_calls,H_L_packets=self.H_L_packets,
            native_shared_L_H_packets=64*self.L_calls*int(not self.dual_L),H_U_calls=self.H_U_calls,H_U_packets=self.H_U_packets,
            scenes=dict(self.scenes),pair_enabled=self.pair_enabled,pair_calls=self.pair_calls,pair_packets=self.pair_packets,pair_rejected=self.pair_rejected,
            pair_weight=PAIR_WEIGHT if self.pair_enabled else 0.,pair_normalizer='whole physical half',pair_targets='opposite detached H probabilities; no hard TX',
            extra_F_forwards=0,restoration_weight_reads=0,restoration_IQ_used=0,L_true_labels_all=True,L_As_KL_weight=0.,ordinary_cross_RX_preserved=True)
        return state

    def guide(self,labeled,student):
        value=super().guide(labeled,student)
        self.metrics.update(HV_L_calls=self.L_calls,HV_H_L_calls=self.H_L_calls,HV_H_L_packets=self.H_L_packets,
            HV_native_L_H_packets=64*self.L_calls*int(not self.dual_L),HV_H_U_calls=self.H_U_calls,HV_H_U_packets=self.H_U_packets)
        return value
