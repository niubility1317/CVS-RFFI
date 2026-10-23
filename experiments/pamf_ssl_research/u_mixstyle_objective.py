"""Temporary student-U feature transforms with detached identity-protected moments."""
import torch
from experiments.cric_dura_comparison.f_model import PREFIX
from .objective import Objective


def protected_partners(probability,weight,receivers,allowed,threshold):
    p,w=probability.detach(),weight.detach();n=len(p)
    if receivers.shape!=(n,) or set(receivers.tolist())-set(allowed):
        raise ValueError('Only original source RX may define style partners')
    if (p.ndim!=2 or w.shape!=(n,) or not bool(torch.isfinite(p).all())
        or not bool(torch.isfinite(w).all()) or bool((p<0).any())
        or not torch.allclose(p.sum(1),torch.ones(n,device=p.device),atol=1e-5)
        or bool(((w<0)|(w>1)).any())):
        raise ValueError('Normalized detached teacher probabilities/confidence required')
    cls=p.argmax(1);confident=w>=threshold
    eligible=(cls[:,None]==cls[None,:])&(receivers[:,None]!=receivers[None,:])&confident[:,None]&confident[None,:]
    valid=eligible.any(1)
    index=torch.rand(n,n,device=p.device).masked_fill(~eligible,-1.).argmax(1)
    index=torch.where(valid,index,torch.arange(n,device=p.device))
    return index,valid


def mix_moments(h,partner,valid,lam,eps):
    n=len(h);lam=lam.detach()
    if (h.ndim!=3 or partner.shape!=(n,) or valid.shape!=(n,) or lam.shape!=(n,1,1)
        or bool(((partner<0)|(partner>=n)).any()) or not bool(torch.isfinite(lam).all())
        or bool(((lam<0)|(lam>1)).any()) or eps<=0):
        raise ValueError('Aligned temporal features and bounded style plan required')
    mean=h.mean(-1,keepdim=True).detach()
    std=(h.var(-1,unbiased=False,keepdim=True)+eps).sqrt().detach()
    mixed_mean=lam*mean+(1-lam)*mean[partner]
    mixed_std=lam*std+(1-lam)*std[partner]
    mixed=(h-mean)/std*mixed_std+mixed_mean
    return torch.where(valid[:,None,None],mixed,h)


class UnlabeledMixStyleObjective(Objective):
    def __init__(self,student,config):
        super().__init__(student,config)
        self.u_mixstyle_calls=self.u_mixstyle_seen=self.u_mixstyle_eligible=0
        self.u_mixstyle_applied={name:0 for name in config['u_mixstyle']['levels']}
        self._style_context_active=False

    def loss(self,student,device):
        if self.current_epoch<self.c['u_mixstyle']['start_epoch']:
            raise ValueError('U-MixStyle is only the fixed E31-to-E50 continuation')
        value=super().loss(student,device)
        self.metrics.update(self._style_metrics)
        return value

    def unlabeled_student_forward(self,student,strong,domain,target,weight,receivers):
        if self._style_context_active:
            raise ValueError('A single current U student style context required')
        recipe=self.c['u_mixstyle'];n=len(strong)
        partner,valid=protected_partners(target,weight,receivers,self.c['source_rxs'],recipe['eligible_weight_min'])
        cls=target.detach().argmax(1)
        if bool(((cls[valid]!=cls[partner[valid]])|(receivers[valid]==receivers[partner[valid]])).any()):
            raise ValueError('Actual style plan crossed TX predictions or retained same RX')
        alpha=torch.tensor(recipe['alpha'],device=strong.device)
        lam=torch.distributions.Beta(alpha,alpha).sample((n,1,1))
        observed=set();handles=[];applied={};smoke_maps={}
        self._style_context_active=True
        try:
            for name in recipe['levels']:
                def transform(module,inputs,output,name=name):
                    if name in observed or not torch.is_tensor(output) or len(output)!=n:
                        raise ValueError('Exactly one actual U student sequence per style level required')
                    observed.add(name)
                    enabled=self.c.get('u_mixstyle_smoke') or bool(torch.rand((),device=output.device)<recipe['probability_per_level'])
                    value=mix_moments(output,partner,valid,lam,recipe['eps']) if enabled else output
                    applied[name]=int(valid.sum()) if enabled else 0
                    if self.c.get('u_mixstyle_smoke'):
                        value.retain_grad();smoke_maps[name]=value
                    return value
                handles.append(student.get_submodule(PREFIX+name).register_forward_hook(transform))
            output=student.forward_identity_only(strong,domain_labels=domain)
            if observed!=set(recipe['levels']):raise ValueError('Declared student style levels missing')
        finally:
            for handle in handles:handle.remove()
            self._style_context_active=False
        self.u_mixstyle_calls+=1;self.u_mixstyle_seen+=n;self.u_mixstyle_eligible+=int(valid.sum())
        for name in recipe['levels']:self.u_mixstyle_applied[name]+=applied[name]
        self._style_metrics=dict(SSL_U_MIX_calls=self.u_mixstyle_calls,SSL_U_MIX_seen=self.u_mixstyle_seen,
            SSL_U_MIX_eligible=self.u_mixstyle_eligible,SSL_U_MIX_last_eligible=int(valid.sum()),
            **{'SSL_U_MIX_applied_'+name:count for name,count in self.u_mixstyle_applied.items()},
            **{'SSL_U_MIX_last_applied_'+name:count for name,count in applied.items()})
        if self.c.get('u_mixstyle_smoke'):self._smoke_maps=smoke_maps
        return output

    def state_dict(self):
        value=super().state_dict()
        value.update(u_mixstyle=self.c['u_mixstyle'],u_mixstyle_calls=self.u_mixstyle_calls,
            u_mixstyle_seen=self.u_mixstyle_seen,u_mixstyle_eligible=self.u_mixstyle_eligible,
            u_mixstyle_applied=self.u_mixstyle_applied)
        return value

    def load_state_dict(self,state):
        if state.get('u_mixstyle') and state['u_mixstyle']!=self.c['u_mixstyle']:
            raise ValueError('U-MixStyle state belongs to another recipe')
        super().load_state_dict(state)
        for key in ('u_mixstyle_calls','u_mixstyle_seen','u_mixstyle_eligible'):
            setattr(self,key,state.get(key,0))
        self.u_mixstyle_applied=dict(state.get('u_mixstyle_applied',self.u_mixstyle_applied))
