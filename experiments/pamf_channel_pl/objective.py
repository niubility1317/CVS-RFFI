"""Two confidence checks on source IQ,with exact ordinary-PL fallback."""
from contextlib import contextmanager
from functools import wraps
import torch
from torch.nn import functional as F
from .policy import RECIPE,validate_recipe,raw_predictions,channel_gate,dose_conserving_loss


def subset_metadata(metadata,indices,count):
    return {key:value.index_select(0,indices.to(value.device))
        if torch.is_tensor(value) and value.ndim and len(value)==count else value for key,value in metadata.items()}


class ChannelPL:
    def __init__(self,c,simulator=None):
        validate_recipe(c['channel_pl']);self.c=c
        if any(c.get(k) for k in ('truth','target_truth','views_root','target_inputs','dataset_path','tests')):
            raise ValueError('Channel-PL source worker rejects target and truth inputs')
        if simulator is None:
            from experiments.cric_dura_comparison.t1_practical import Practical
            simulator=Practical(c)
        self.simulator=simulator;self.owner=None;self.metrics={}
        self.calls=self.seen=self.accepted=self.generated=self.channel_accepted=0
        self.teacher_channel_calls=self.student_channel_calls=0

    def loss(self,model,device):
        if not model.training:raise ValueError('Only the native source training hook may compute Channel-PL loss')
        if self.owner is not None and self.owner is not model:raise ValueError('One source model per objective scope required')
        self.owner=model
        try:batch=next(model._cd_U_iter)
        except StopIteration:
            model._cd_U_iter=iter(model._cd_U_loader);batch=next(model._cd_U_iter)
        if (not bool((batch[1]==-1).all()) or 'tx_label_visible' not in batch[3]
                or bool(torch.as_tensor(batch[3]['tx_label_visible']).any())):
            raise ValueError('Only original U_s with hidden TX labels may enter Channel-PL')
        x=batch[0].to(device);domain=batch[2].to(device);metadata=batch[3];n=len(x)
        if not n:raise ValueError('Nonempty original source U batch required')
        step=int(model._feature_l_steps)+int(self.c.get('step_offset',0));epoch=(step-1)//222+1
        selected=valid_indices=simulated=None;teacher_channel_packets=0
        model.eval()
        try:
            with torch.no_grad():
                raw=model(x)
                if raw.shape!=(n,self.c['num_classes']):raise ValueError('Original jointly trained default head required')
                _,pseudo,accepted=raw_predictions(raw)
                physical=(torch.arange(n,device=x.device)+step).remainder(2).bool()
                selected=torch.nonzero(accepted&physical,as_tuple=False).flatten()
                valid_indices=torch.empty(0,dtype=torch.long,device=x.device)
                if len(selected):
                    simulated=self.simulator(x[selected],subset_metadata(metadata,selected,n),epoch,step)
                    if simulated.shape!=x[selected].shape or not bool(torch.isfinite(simulated).all()):raise ValueError('Finite aligned native source channel IQ required')
                    q=model(simulated);valid=channel_gate(pseudo,accepted,selected,q)
                    valid_indices=selected[valid];simulated=simulated[valid]
                    teacher_channel_packets=len(selected)
        finally:model.train()
        logits=model.forward_identity_only(x,domain_labels=domain)['tx_logits']
        ce=F.cross_entropy(logits,pseudo,reduction='none');channel_logits=None
        if len(valid_indices):channel_logits=model.forward_identity_only(simulated,domain_labels=domain[valid_indices])['tx_logits']
        value,delta=dose_conserving_loss(ce,accepted,valid_indices,channel_logits,pseudo)
        if not bool(torch.isfinite(value)):raise ValueError('Finite dose-conserving U objective required')
        model._cd_U_seen+=n;model._cd_U_accepted+=int(accepted.sum())
        self.calls+=1;self.seen+=n;self.accepted+=int(accepted.sum());self.generated+=teacher_channel_packets
        self.channel_accepted+=len(valid_indices);self.teacher_channel_calls+=int(teacher_channel_packets>0)
        self.student_channel_calls+=int(len(valid_indices)>0)
        self.metrics=dict(CPL_calls=self.calls,CPL_seen=self.seen,CPL_accepted=self.accepted,
            CPL_channel_generated=self.generated,CPL_channel_accepted=self.channel_accepted,
            CPL_raw_student_views=self.seen,CPL_channel_student_views=self.channel_accepted,
            CPL_raw_teacher_views=self.seen,CPL_channel_teacher_views=self.generated,
            CPL_original_raw_PL_loss=(ce.detach()*accepted).mean(),CPL_channel_reallocation_delta=delta.detach(),
            CPL_loss=value.detach(),CPL_U_weight=1.,CPL_channel_ratio=.25)
        return value

    def state_dict(self):
        if not 0<=self.channel_accepted<=self.generated<=self.accepted<=self.seen:
            raise ValueError('Aligned source acceptance and physical-half counters required')
        return dict(recipe=dict(RECIPE),calls=self.calls,seen=self.seen,accepted=self.accepted,
            generated=self.generated,channel_accepted=self.channel_accepted,
            teacher_views=self.seen+self.generated,student_views=self.seen+self.channel_accepted,
            extra_teacher_calls=self.teacher_channel_calls,extra_student_calls=self.student_channel_calls,
            source_channel_coverage=dict(getattr(self.simulator,'coverage',{})),
            raw_gate_unchanged=True,per_accepted_sample_CE_dose=1.,EMA_updates=0,U_truth_read=False,
            new_head=False,new_inference_parameters=0,U_pair_KL=0.,source_V_training_reads=0,G_training_reads=0,target_reads=0)


_ACTIVE=False


@contextmanager
def installed(c,holder):
    global _ACTIVE
    if _ACTIVE:raise ValueError('No nested Channel-PL source scope')
    validate_recipe(c['channel_pl'])
    from experiments.pamf_repl_completion.objectives import installed as baseline
    from experiments.cric_dura_comparison import t1_model
    original_plain,original_hook=t1_model.plain_pl,t1_model.hook
    def plain(model,device):
        if 'objective' not in holder:holder['objective']=ChannelPL(c)
        return holder['objective'].loss(model,device)
    @wraps(original_hook)
    def hook(model,**kwargs):
        value,metrics=original_hook(model,**kwargs)
        if 'objective' in holder:metrics=dict(metrics,**holder['objective'].metrics)
        return value,metrics
    _ACTIVE=True
    try:
        with baseline(c,holder,False):
            t1_model.plain_pl,t1_model.hook=plain,hook
            try:yield
            finally:t1_model.plain_pl,t1_model.hook=original_plain,original_hook
    finally:_ACTIVE=False
