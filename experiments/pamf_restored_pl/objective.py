"""Retain ordinary self-PL; add one matched frozen-target soft objective."""
from contextlib import contextmanager
from functools import wraps
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.restored_dose_source import soft_loss
from .cache import SourceCache
from .plan import validate_config


def ramp_weight(weight,global_step):
    return weight*min(max((global_step-6660)/444.,0.),1.)


class RestoredObjective:
    def __init__(self,c):
        validate_config(c);self.c=c;self.recipe=c['restored_pl'];self.calls=self.seen=self.raw_accepted=0
        self.extra_seen=self.extra_accepted=self.extra_forwards=self.restored_forwards=0
        self.cache=None if self.recipe['extra']=='none' else SourceCache(c)
        self.metrics={}

    def loss(self,student,device):
        try:batch=next(student._cd_U_iter)
        except StopIteration:
            student._cd_U_iter=iter(student._cd_U_loader);batch=next(student._cd_U_iter)
        if (not bool((batch[1]==-1).all()) or bool(torch.as_tensor(batch[3]['tx_label_visible']).any())
                or set(torch.as_tensor(batch[3]['rx_i']).tolist())-set(self.c['source_rxs'])):
            raise ValueError('Only hidden-label source U permitted')
        x=batch[0].to(device);domain=batch[2].to(device);student.eval()
        with torch.no_grad():confidence,pseudo=student(x).softmax(1).max(1)
        student.train();accepted=confidence>=.95
        logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
        main=(F.cross_entropy(logits,pseudo,reduction='none')*accepted).mean()
        n=len(x);self.calls+=1;self.seen+=n;self.raw_accepted+=int(accepted.sum())
        student._cd_U_seen+=n;student._cd_U_accepted+=int(accepted.sum())
        extra=main*0.;weight=0.
        if self.cache is not None:
            step=self.calls+6660;k=(step-1)%3
            q,mask,gx=self.cache.take(batch[3],k,device,self.recipe['extra']=='restored')
            view=x if gx is None else gx
            extra=sum(soft_loss(student.forward_identity_only(view,domain_labels=domain)['tx_logits'],q,mask) for _ in range(2))/2
            self.extra_seen+=n;self.extra_accepted+=int(mask.sum());self.extra_forwards+=2*n
            if gx is not None:self.restored_forwards+=2*n
            weight=ramp_weight(self.recipe['extra_weight'],step)
        self.metrics=dict(RPL_main_loss=main.detach(),RPL_extra_soft_loss=extra.detach(),RPL_extra_weight=weight,
            RPL_U_seen=self.seen,RPL_raw_accepted=self.raw_accepted,RPL_extra_accepted=self.extra_accepted,
            RPL_extra_forwards=self.extra_forwards,RPL_restored_forwards=self.restored_forwards,
            RPL_ensemble_updates=0,RPL_G_updates=0,RPL_U_truth=0)
        return main+weight*extra

    def state_dict(self):
        return dict(calls=self.calls,U_seen=self.seen,raw_accepted=self.raw_accepted,extra_seen=self.extra_seen,
            extra_accepted=self.extra_accepted,extra_student_forwards=self.extra_forwards,
            restored_student_forwards=self.restored_forwards,recipe=self.recipe,
            teacher_updates=0,G_updates=0,U_truth_read=False,target_reads=0,source_V_training_reads=0,
            new_head=False,teacher_training_forward_cost='shared offline cache,reported separately')


@contextmanager
def installed(c,objective):
    from experiments.pamf_repl_completion.objectives import installed as baseline
    from experiments.cric_dura_comparison import t1_model
    old_plain,old_hook=t1_model.plain_pl,t1_model.hook
    def plain(student,device):return objective.loss(student,device)
    @wraps(old_hook)
    def hook(self,**kwargs):
        value,metrics=old_hook(self,**kwargs)
        return value,dict(metrics,**objective.metrics)
    with baseline(c,{},False):
        t1_model.plain_pl,t1_model.hook=plain,hook
        try:yield
        finally:t1_model.plain_pl,t1_model.hook=old_plain,old_hook
