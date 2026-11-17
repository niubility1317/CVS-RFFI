"""Capture exact U component graphs and apply preregistered reference rules."""
from contextlib import contextmanager
from pathlib import Path
import torch
from experiments.pamf_repl_attribution_factorial import objective as original
from .observer import ReferenceObserver
from .projection import reference_loss


class PriorityREPL(original.AttributionREPL):
    def __init__(self, c):
        super().__init__(c)
        path=None if c.get('hard_pl_smoke') else Path(c['output_root'])/'reference_adamw_samples.jsonl.gz'
        self.observer=ReferenceObserver(path)

    def guide(self, labeled, student):
        value=super().guide(labeled, student)
        mode=self.c['tuning_recipe']['mode']
        self.metrics['HPL_priority_native_guard_active']=int(mode=='native')
        if mode!='native':
            # The inherited configuration is still used for native objective validation;
            # its cap is not applied by the replacement optimizer rule.
            self.metrics['HPL_GP_U_to_L_norm_cap']=0.
        if self.calls % 100 == 0 or self.c.get('hard_pl_smoke'):
            reference=self.parts[self.c['tuning_recipe']['reference']]
            named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
            grads=torch.autograd.grad(reference,[p for _,p in named],retain_graph=True,allow_unused=True)
            empty=tuple(None for _ in named)
            self.observer.source_step=int(self.c.get('step_offset',0))+self.calls
            self.observer.capture(student,named,grads,empty,empty)
        return value

    def state_dict(self):
        result=super().state_dict()
        result['U_gradient_guard']['effective_rule']=self.c['tuning_recipe']['mode']
        result['sampled_reference_adamw'] = dict(calls=self.observer.calls,
            pending=self.observer.pending is not None, cadence=100,
            reference=self.c['tuning_recipe']['reference'], live_parameters_modified=False)
        return result


@contextmanager
def installed(c, holder):
    from experiments.pamf_ssl_research import hard_pl_guard_objective as guard, hard_pl_objective
    from experiments.pamf_ssl_research.optimizer_step_guard import installed as optimizer_scope
    old_guard, old_pair, old_class = guard.protected_loss, hard_pl_objective.masked_pair, original.AttributionREPL
    recipe=c['tuning_recipe']; calls=[]
    def masked_pair(a,b,y,accepted,kl_weight):
        loss,ce,kl=old_pair(a,b,y,accepted,kl_weight*recipe['kl_factor'])
        calls.append((kl_weight,kl))
        if len(calls)==4:
            if [x[0] for x in calls] != [1.,1.,0.,1.]:
                raise ValueError('Unexpected U paired-view KL ordering')
            weighted=sum(coef*weight*value for coef,(weight,value) in zip((.5,.25,.25,.25),calls))
            holder['objective'].parts['pair_kl']=weighted*recipe['kl_factor']*c['hard_pl']['weight']/1.25
            calls.clear()
        return loss,ce,kl
    def protected(labeled,unlabeled,named,*,max_u_ratio):
        obj=holder['objective']; ref=obj.parts[recipe['reference']]
        raw_value, raw_details, raw_grads, stats=reference_loss(labeled,unlabeled,ref,named,cap=None)
        if recipe['mode']=='native':
            value,details,gradients=old_guard(labeled,unlabeled,named,max_u_ratio=max_u_ratio)
        elif recipe['mode']=='raw':
            value,details,gradients=raw_value,raw_details,raw_grads
        else:
            value,details,gradients,stats=reference_loss(labeled,unlabeled,ref,named,cap=.5)
        # The native control transforms the raw total too; record the direction
        # of its actual effective gradient, not the raw pre-guard direction.
        if recipe['mode']=='native':
            from experiments.pamf_ssl_research.u_gradprotect_objective import parameter_group
            refs=torch.autograd.grad(ref,[p for _,p in named],retain_graph=True,allow_unused=True)
            for group in ('t2','cls_head','other'):
                zero=torch.zeros((),device=named[0][1].device,dtype=torch.float64)
                pairs=[(r,g) for (name,_),r,g in zip(named,refs,gradients)
                       if parameter_group(name)==group and r is not None]
                remainders=[(torch.zeros_like(r) if g is None else g)-r for r,g in pairs]
                stats[group+'_reference_remainder_dot_after']=float(sum(
                    ((r.double()*v.double()).sum() for (r,_),v in zip(pairs,remainders)),zero))
                stats[group+'_applied_remainder_norm']=float(sum(
                    (v.double().square().sum() for v in remainders),zero).sqrt())
        obj.metrics.update({'HPL_reference_'+key:value for key,value in stats.items()})
        return value,details,gradients
    original.AttributionREPL=PriorityREPL
    guard.protected_loss=protected; hard_pl_objective.masked_pair=masked_pair
    try:
        with optimizer_scope():
            with original.installed(c,holder):
                yield
                if calls: raise ValueError('Incomplete paired-view KL reference')
    finally:
        guard.protected_loss=old_guard; hard_pl_objective.masked_pair=old_pair; original.AttributionREPL=old_class
