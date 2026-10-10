"""Single-point weak conditional MixStyle, independent of legacy schedules."""
from collections import defaultdict
from contextlib import contextmanager
import copy
import numpy as np
import torch


def _values(value):
    return value.detach().cpu().tolist() if torch.is_tensor(value) else list(value)


class WeakConditionalMixStyle:
    def __init__(self,seed,p=.15,eta=.2,alpha=.1,eps=1e-6):
        if not 0<=p<=.15 or not 0<=eta<=.2 or alpha<=0: raise ValueError('Weak style bounds violated')
        self.p,self.eta,self.alpha,self.eps=float(p),float(eta),float(alpha),float(eps)
        self.rng=np.random.default_rng(int(seed)); self._context=None; self.handle=None
        self.stats=defaultdict(lambda: defaultdict(float)); self.last={}; self.hook_name=None

    def probability(self,epoch): return self.p*min(1.,max(0.,(float(epoch)-20)/20))

    def attach(self,model):
        if self.handle is not None: raise RuntimeError('Style hook already attached')
        points=[(n,m) for n,m in model.named_modules() if n.split('.')[-1]=='time_down']
        if len(points)!=1: raise ValueError('Exactly one time_down hook is required')
        self.hook_name,point=points[0]
        self.handle=point.register_forward_hook(lambda module,args,out:self.apply(out))
        return self

    install=attach

    def detach(self):
        if self.handle is not None: self.handle.remove(); self.handle=None
        self._context=None

    @contextmanager
    def context(self,*,y,rx,day,pid,role,epoch):
        if self._context is not None: raise RuntimeError('Nested style context forbidden')
        self._context=dict(y=_values(y),rx=_values(rx),day=_values(day),pid=[tuple(p) if isinstance(p,(list,tuple)) else p for p in _values(pid)],role=str(role),epoch=int(epoch))
        try: yield self
        finally: self._context=None

    def apply(self,h):
        ctx=self._context
        # Consume context at the hook: a repeated forward cannot accidentally
        # inherit labels/role from a preceding call, even inside one with-block.
        self._context=None
        role='unscoped' if ctx is None else ctx['role']; stats=self.stats[role]
        stats['calls']+=1; stats['packets']+=len(h)
        self.last=dict(role=role,applied=False,donors=[],a=[])
        if ctx is None or role!='native_L_clean': stats['disabled_calls']+=1; return h
        if h.ndim!=3: raise ValueError('time_down style requires [B,C,T]')
        if any(len(ctx[k])!=len(h) for k in ('y','rx','day','pid')): raise ValueError('Style physical metadata size mismatch')
        if any(int(v)<0 for k in ('y','rx','day') for v in ctx[k]): raise ValueError('Style requires visible source L metadata')
        p=self.probability(ctx['epoch']); stats['probability_sum']+=p; stats['probability_observations']+=1
        if p<=0: stats['warmup_disabled_calls']+=1; return h
        donors=[]
        for i in range(len(h)):
            legal=[j for j in range(len(h)) if ctx['y'][i]==ctx['y'][j] and ctx['rx'][i]!=ctx['rx'][j] and ctx['day'][i]==ctx['day'][j] and ctx['pid'][i]!=ctx['pid'][j]]
            donors.append(legal)
        covered=sum(bool(v) for v in donors)
        stats['eligible_packets']+=covered; stats['eligible_observations']+=len(h)
        if self.rng.random()>=p: stats['bernoulli_disabled_calls']+=1; return h
        stats['enabled_calls']+=1
        chosen=[int(self.rng.choice(v)) if v else i for i,v in enumerate(donors)]
        a=self.eta*(1-self.rng.beta(self.alpha,self.alpha,size=len(h)))
        a=np.asarray([v if donors[i] else 0. for i,v in enumerate(a)])
        a_t=torch.as_tensor(a,device=h.device,dtype=h.dtype)[:,None,None]
        mu=h.mean(-1,keepdim=True).detach(); sigma=(h.var(-1,unbiased=False,keepdim=True)+self.eps).sqrt().detach()
        mixed_mu=(1-a_t)*mu+a_t*mu[chosen]; mixed_sigma=(1-a_t)*sigma+a_t*sigma[chosen]
        output=(h-mu)/sigma*mixed_sigma+mixed_mu
        mask=torch.as_tensor([bool(v) for v in donors],device=h.device)[:,None,None]
        output=torch.where(mask,output,h)
        ratios=(output.detach()-h.detach()).flatten(1).norm(dim=1)/h.detach().flatten(1).norm(dim=1).clamp_min(self.eps)
        stats['changed_packets']+=int((ratios>0).sum()); stats['delta_ratio_sum']+=float(ratios.sum()); stats['delta_ratio_observations']+=len(h)
        stats['movement_sum']+=float(a.sum()); stats['movement_observations']+=len(h)
        stats['movement_max']=max(stats['movement_max'],float(a.max(initial=0)))
        self.last=dict(role=role,applied=True,donors=chosen,a=a.tolist(),eligible=covered,delta_ratio=ratios.cpu().tolist())
        return output

    def snapshot(self):
        result={}
        for role,raw in self.stats.items():
            v=dict(raw)
            for name,num,den in [('donor_coverage','eligible_packets','eligible_observations'),('changed_ratio','changed_packets','eligible_observations'),('mean_delta_ratio','delta_ratio_sum','delta_ratio_observations'),('mean_movement','movement_sum','movement_observations')]:
                v[name]=v.get(num,0)/v[den] if v.get(den,0) else None
            result[role]=v
        return dict(hook=self.hook_name,p=self.p,eta=self.eta,alpha=self.alpha,roles=result)

    def state_dict(self):
        return copy.deepcopy(dict(rng=self.rng.bit_generator.state,stats={k:dict(v) for k,v in self.stats.items()}))

    def load_state_dict(self,state):
        self.rng.bit_generator.state=copy.deepcopy(state['rng']); self.stats=defaultdict(lambda:defaultdict(float))
        for k,v in state['stats'].items(): self.stats[k].update(v)
