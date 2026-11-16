"""Reference step from a segment-local EMA of full labeled gradients."""
from contextlib import contextmanager
from pathlib import Path
from experiments.pamf_repl_reference_dominance import objective as reference
from experiments.pamf_ssl_research.source_branch_gradients import _relation


def update_reference_ema(previous,current,beta):
    if not 0 < beta < 1 or (previous is not None and len(previous)!=len(current)):
        raise ValueError('Aligned gradient tensors and EMA beta strictly between zero and one required')
    if previous is None:
        return tuple(None if value is None else value.detach().clone() for value in current)
    result=[]
    for old,value in zip(previous,current):
        if old is None and value is None:result.append(None)
        elif old is None:result.append(value.detach().clone()*(1-beta))
        elif value is None:result.append(old.detach().clone()*beta)
        else:result.append(old.detach()*beta+value.detach()*(1-beta))
    return tuple(result)


class EMAReferenceGuard(reference.ReferenceDominanceGuard):
    def __init__(self,destination,cap,beta):
        super().__init__(destination,cap)
        if not 0 < beta < 1:raise ValueError('EMA beta strictly between zero and one required')
        self.beta=beta;self.gradient_ema=None;self.ema_updates=0

    def capture(self,model,named,full,clean,physical):
        super().capture(model,named,full,clean,physical)
        self.gradient_ema=update_reference_ema(self.gradient_ema,full,self.beta)
        self.ema_updates+=1

    def _reference_step(self,model,named,optimizer,gradients,*,clip_impl):
        if self.gradient_ema is None or self.ema_updates!=self.requests:
            raise ValueError('Current segment source-labeled EMA reference required')
        values,meta=super()._reference_step(model,named,optimizer,self.gradient_ema,clip_impl=clip_impl)
        meta.update(reference_loss='segment_local_full_L_gradient_EMA',reference_beta=self.beta,
            current_full_L_vs_reference_gradient={group:_relation(named,gradients,self.gradient_ema,group)
                                                  for group in ('t2','cls_head','other')})
        return values,meta

    def state_dict(self):
        state=super().state_dict()
        state.update(reference_definition='current same-state AdamW step from segment-local full-L gradient EMA; prior joint moments retained',
            reference_gradient_beta=self.beta,reference_gradient_EMA_updates=self.ema_updates,
            additional_update_includes='current full L minus EMA reference and all U objectives',
            reference_EMA_reset='at each source segment boundary')
        return state


class EMAReferenceObjective(reference.ReferenceDominanceObjective):
    def __init__(self,config):
        super().__init__(config)
        recipe=config['tuning_recipe'];dest=None if config.get('hard_pl_smoke') else Path(config['output_root'])/'step_guard_metrics.jsonl.gz'
        self.parameter_guard=EMAReferenceGuard(dest,float(recipe['reference_cap']),float(recipe['reference_beta']))


@contextmanager
def installed(config,holder):
    previous=reference.ReferenceDominanceObjective;reference.ReferenceDominanceObjective=EMAReferenceObjective
    try:
        with reference.installed(config,holder):yield
    finally:reference.ReferenceDominanceObjective=previous
