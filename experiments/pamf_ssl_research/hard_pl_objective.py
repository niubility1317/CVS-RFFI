"""Ordinary hard pseudo labels; optional six-pass PAMF training-only objective."""
import torch
from torch.nn import functional as F
from experiments.cric_dura_comparison.f_model import PREFIX,contrast


def masked_pair(a,b,y,accepted,kl_weight):
    if (a.shape!=b.shape or a.ndim!=2 or y.shape!=a.shape[:1] or accepted.shape!=y.shape
        or y.dtype!=torch.long or accepted.dtype!=torch.bool):
        raise ValueError('Paired logits and exact boolean acceptance mask required')
    if not len(y):
        zero=(a.sum()+b.sum())*0.;return zero,zero,zero
    ce=.5*(F.cross_entropy(a,y,reduction='none',label_smoothing=.01)
           +F.cross_entropy(b,y,reduction='none',label_smoothing=.01))
    p,q=a.log_softmax(1),b.log_softmax(1)
    kl=.5*((p.exp()-q.exp())*(p-q)).sum(1)
    ce=(ce*accepted).mean();kl=(kl*accepted).mean()
    return ce+kl_weight*kl,ce,kl


def mechanism_loss(o0,o1,m0,m1,r0,r1,y,accepted,physical,aux):
    if physical.dtype!=torch.bool or physical.shape!=y.shape or not len(y) or int(physical.sum())*2!=len(y):
        raise ValueError('Exact even ordinary/physical half partition required')
    ordinary,_,okl=masked_pair(o0,o1,y,accepted,1.)
    mixed,_,_=masked_pair(m0[~physical],m1[~physical],y[~physical],accepted[~physical],1.)
    as_loss,_,askl=masked_pair(m0[physical],m1[physical],y[physical],accepted[physical],0.)
    raw,_,rkl=masked_pair(r0,r1,y,accepted,1.)
    loss=(.5*ordinary+.25*mixed+.25*as_loss+.25*raw+.1*aux)/1.25
    return loss,dict(ordinary_KL=okl.detach(),raw_KL=rkl.detach(),As_KL=askl.detach(),
        As_KL_weight=0.,contrast=aux.detach(),CE_normalizer=1.25)


class HardPLObjective:
    ema_updates=0
    def __init__(self,c):
        self.c=c;self.settings=c['hard_pl'];self.calls=self.seen=self.accepted=self.student_views=0
        self.physical_seen=self.physical_student_views=self.raw_As_seen=0;self.metrics={};self.capture_active=False
        self.contrast_anchors={name:0 for name in ('time_down','t2.pool')}

    def loss(self,student,device):
        try:batch=next(student._cd_U_iter)
        except StopIteration:
            student._cd_U_iter=iter(student._cd_U_loader);batch=next(student._cd_U_iter)
        if not bool((batch[1]==-1).all()):raise ValueError('U truth exposed')
        x=batch[0].to(device);domain=batch[2].to(device);n=len(x)
        was=student.training
        try:
            student.eval()
            with torch.no_grad():confidence,pseudo=student(x).softmax(1).max(1)
        finally:student.train(was)
        accepted=confidence>=self.settings['threshold'];detail={}
        if self.settings['method']=='HPL2':
            logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
            base=(F.cross_entropy(logits,pseudo,reduction='none')*accepted).mean();views=n
        else:
            step=student._feature_l_steps+self.c.get('step_offset',0);epoch=(step-1)//222+1
            meta={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch[3].items()}
            rx=meta['rx_i'].long()
            if set(rx.tolist())-set(self.c['source_rxs']):raise ValueError('Non-source U RX')
            physical=(torch.arange(n,device=device)+step).remainder(2).bool()
            no_uas=bool(self.c.get('hard_pl_no_uas'))
            with torch.no_grad():
                ordinary=x+.015*torch.randn_like(x);mixed=ordinary.clone()
                pm={k:v[physical] if torch.is_tensor(v) and v.ndim and len(v)==n else v for k,v in meta.items()}
                mixed[physical]=x[physical] if no_uas else student._cd_practical(x[physical],pm,epoch,step)
                if self.c.get('hard_pl_smoke'):
                    self.smoke_U_As_input_raw=bool(no_uas and torch.equal(mixed[physical],x[physical]))
            maps=[];outputs=[];self.capture_active=True
            try:
                for _ in range(2):
                    current={};handles=[]
                    try:
                        for name in ('time_down','t2.pool'):
                            def capture(module,inputs,output,name=name):
                                current[name]=output
                                if self.c.get('hard_pl_smoke'):output.retain_grad()
                            handles.append(student.get_submodule(PREFIX+name).register_forward_hook(capture))
                        outputs.append(student.forward_identity_only(ordinary,domain_labels=domain)['tx_logits'])
                    finally:
                        for handle in handles:handle.remove()
                    if set(current)!=set(self.contrast_anchors):raise ValueError('Actual U shallow/middle maps missing')
                    maps.append(current)
            finally:self.capture_active=False
            auxiliaries=[]
            for current in maps:
                selected={name:value[accepted] for name,value in current.items()}
                if bool(accepted.any()):
                    value,stats=contrast(selected,pseudo[accepted],rx[accepted])
                else:
                    value=sum(h.sum()*0. for h in current.values());stats=[]
                auxiliaries.append(value)
                for stat in stats:self.contrast_anchors[stat['level']]+=stat['valid_anchors']
            aux=torch.stack(auxiliaries).mean()
            more=[student.forward_identity_only(value,domain_labels=domain)['tx_logits'] for value in (mixed,mixed,x,x)]
            base,detail=mechanism_loss(*outputs,*more,pseudo,accepted,physical,aux);views=6*n
            if no_uas:self.raw_As_seen+=int(physical.sum())
            else:
                self.physical_seen+=int(physical.sum());self.physical_student_views+=2*int(physical.sum())
            if self.c.get('hard_pl_smoke'):self.smoke_maps=maps
        loss=self.settings['weight']*base
        self.calls+=1;self.seen+=n;self.accepted+=int(accepted.sum());self.student_views+=views
        student._cd_U_seen+=n;student._cd_U_accepted+=int(accepted.sum())
        self.metrics=dict(HPL_loss=loss.detach(),HPL_base_loss=base.detach(),HPL_weight=self.settings['weight'],
            HPL_calls=self.calls,HPL_seen=self.seen,HPL_accepted=self.accepted,HPL_confidence=confidence.mean(),
            HPL_student_views=self.student_views,HPL_physical_seen=self.physical_seen,
            HPL_physical_student_views=self.physical_student_views,HPL_EMA_updates=0.,
            HPL_raw_As_seen=self.raw_As_seen,
            **{'HPL_'+key:value for key,value in detail.items()},
            **{'HPL_anchors_'+key:value for key,value in self.contrast_anchors.items()})
        return loss

    def state_dict(self):
        return dict(method=self.settings['method'],calls=self.calls,seen=self.seen,accepted=self.accepted,
            student_views=self.student_views,physical_seen=self.physical_seen,
            physical_student_views=self.physical_student_views,contrast_anchors=self.contrast_anchors,
            raw_As_seen=self.raw_As_seen,
            EMA_updates=0,U_truth_read=False)
