"""Final T1: six real L forwards, As CE/KL0 and cumulative ORD128 CRIC."""
from types import MethodType
import torch
from torch.nn import functional as F
from .f_model import install as install_backbone,clear,contrast


def pair(a,b,y,smoothing=.01,kl_weight=1.):
    ce=.5*(F.cross_entropy(a,y,label_smoothing=smoothing)+F.cross_entropy(b,y,label_smoothing=smoothing))
    p,q=a.log_softmax(1),b.log_softmax(1)
    kl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
    return ce+kl_weight*kl,ce,kl


def plain_pl(self,device):
    try:batch=next(self._cd_U_iter)
    except StopIteration:
        self._cd_U_iter=iter(self._cd_U_loader);batch=next(self._cd_U_iter)
    if not bool((batch[1]==-1).all()):raise ValueError('U truth exposed')
    x=batch[0].to(device);self.eval()
    with torch.no_grad():confidence,pseudo=self(x).softmax(1).max(1)
    self.train();accepted=confidence>=.95
    logits=self.forward_identity_only(x,domain_labels=batch[2].to(device))['tx_logits']
    self._cd_U_seen+=len(x);self._cd_U_accepted+=int(accepted.sum())
    return (F.cross_entropy(logits,pseudo,reduction='none')*accepted).mean()


def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
    if self is not model or not self.training or labeled_count!=128 or optimizer_step!=self._feature_l_steps:
        raise ValueError('Ordered true L128 T1 update required')
    c=self._cd_config;arm=c.get('ablation');step=optimizer_step+c.get('step_offset',0);epoch=(step-1)//222+1
    pending=self._cd_pending
    if pending is None or pending['x'] is None:raise ValueError('Actual ordinary IQ absent')
    ordinary=pending['x'];first=labeled_output['tx_logits'];raw=labeled_inputs.detach()
    if raw.shape!=(128,2,256) or labels.min()<0 or labels.max()>=c['num_classes']:raise ValueError('T1 source slots/classes differ')
    if arm=='F15':
        clear(self);total=F.cross_entropy(first,labels);sat,_=self._cd_augmentor(ordinary,(step-1)/11100.,step)
        combined=self.forward_identity_only(torch.cat((ordinary,sat)),domain_labels=torch.cat((domains,domains)))['tx_logits']
        total=total+.68*F.cross_entropy(combined[128:],labels);self._cd_extra_L+=256;self._cd_sat_seen+=128
        metrics=dict(T1_CE_control=1.,T1_L_packets=384)
    else:
        single=arm=='F05';kl_weight=0. if arm=='F04' or single else 1.
        if single:second=first;second_maps=pending['first']
        else:
            self._cd_phase='second'
            second_out=self.forward_identity_only(ordinary,domain_labels=domains)
            second=second_out['tx_logits'];second_maps=pending['second'];self._cd_extra_L+=128
        first_maps=pending['first'];clear(self)
        common,_,kl=pair(first,second,labels,kl_weight=kl_weight)
        mask=(torch.arange(128,device=labels.device)+optimizer_step).remainder(2)!=0
        mixed=ordinary.clone()
        if arm!='F17':
            meta={k:v[mask.to(v.device)] if torch.is_tensor(v) and v.ndim and len(v)==128 else v for k,v in extra[1].items()}
            satellite=self._cd_practical(raw[mask],meta,epoch,step);mixed[mask]=satellite
            self._cd_As_seen+=64
        else:mixed[mask]=raw[mask]
        appended=None
        if c['satellite_concat']:
            appended,info=self._cd_augmentor(ordinary,(step-1)/11100.,step)
            mixed=torch.cat((mixed,appended));md=torch.cat((domains,domains))
            self._cd_sat_seen+=128;self._cd_sat_applied+=128*int(info['applied'])
        else:md=domains
        m0=self.forward_identity_only(mixed,domain_labels=md)['tx_logits']
        m1=m0 if single else self.forward_identity_only(mixed,domain_labels=md)['tx_logits']
        self._cd_extra_L+=len(mixed)*(1 if single else 2)
        ordinary_mixed,_,_=pair(m0[:128][~mask],m1[:128][~mask],labels[~mask],kl_weight=kl_weight)
        as_loss,_,as_kl=pair(m0[:128][mask],m1[:128][mask],labels[mask],kl_weight=1. if arm=='F18' else 0.)
        if arm=='F19':raw_loss=first.sum()*0.
        else:
            r0=self.forward_identity_only(raw,domain_labels=domains)['tx_logits']
            r1=r0 if single else self.forward_identity_only(raw,domain_labels=domains)['tx_logits']
            raw_loss,_,_=pair(r0,r1,labels,kl_weight=kl_weight);self._cd_extra_L+=128*(1 if single else 2)
        total=.5*common+.25*ordinary_mixed+.25*as_loss+.25*raw_loss
        if appended is not None:total=total+.68*pair(m0[128:],m1[128:],labels,kl_weight=0.)[0]
        weight=0. if arm=='F16' else .1*min(max((epoch-5)/10.,0.),1.)
        stats=[];aux=first.sum()*0.
        if weight:
            rx=extra[1]['rx_i'].to(labels.device).long()
            if set(rx.tolist())-set(c['source_rxs']):raise ValueError('Non-source physical RX')
            maps=[first_maps] if single else [first_maps,second_maps]
            if arm=='F03':maps=[{'joint160':labeled_output['z_id']}] if single else [{'joint160':labeled_output['z_id']},{'joint160':second_out['z_id']}]
            values=[]
            for m in maps:
                value,detail=contrast(m,labels,rx);values.append(value);stats+=detail
            aux=torch.stack(values).mean();total=total+weight*aux
        if (c['regime']=='matched_sat_plain_pl' or arm=='F21') and epoch>=c.get('U_start_epoch',1):
            total=total+plain_pl(self,first.device)
        metrics=dict(T1_ordinary_KL=kl.detach(),T1_As_KL=as_kl.detach(),T1_As_KL_weight=1. if arm=='F18' else 0.,
            T1_contrast=aux.detach(),CRIC_supcon_weight=weight,CRIC_valid_anchors=sum(d['valid_anchors'] for d in stats),
            CRIC_total_anchors=sum(d['total_anchors'] for d in stats),T1_L_packets=128+self._cd_extra_L-self._cd_previous_extra)
    self._cd_previous_extra=self._cd_extra_L;self._cd_calls+=1
    metrics.update(T1_As_seen=self._cd_As_seen,T1_sat_seen=self._cd_sat_seen,T1_PL_seen=self._cd_U_seen,T1_PL_accepted=self._cd_U_accepted)
    return total-identity_loss,metrics


hook._accepts_labeled_inputs=True


def install(model,c):
    model=install_backbone(model,c);model._source_labeled_feature_hook=MethodType(hook,model)
    model._cd_As_seen=model._cd_previous_extra=0
    if c.get('ablation')!='F15':
        from .t1_practical import Practical
        model._cd_practical=Practical(c)
    elif not hasattr(model,'_cd_augmentor'):
        from experiments.phase1_external_dg_aug.augmentation import DefaultConcat
        model._cd_augmentor=DefaultConcat(2027)
    model._feature_learning_report.update(final_method='pamf_t1_e50',trajectory='20(13L+7LU)+10LU+20LU',As_KL=0.)
    return model
