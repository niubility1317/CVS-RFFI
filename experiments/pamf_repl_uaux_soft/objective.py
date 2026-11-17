"""Apply registered auxiliary-only rules and record actual AdamW counterfactuals."""
from contextlib import contextmanager
from pathlib import Path
import torch
from experiments.pamf_repl_priority import objective as priority
from .observer import DualObserver
from .projection import auxiliary_loss


class SoftREPL(priority.PriorityREPL):
    def __init__(self, c):
        super().__init__(c)
        root = Path(c['output_root'])
        self.observer = DualObserver(None if c.get('hard_pl_smoke') else root/'reference_adamw_samples.jsonl.gz',
                                     None if c.get('hard_pl_smoke') else root/'adamw_LU_samples.jsonl.gz')

    def guide(self, labeled, student):
        sample = self.calls % 100 == 0 or self.c.get('hard_pl_smoke')
        if sample:
            params = tuple(p for p in student.parameters() if p.requires_grad)
            gl = torch.autograd.grad(labeled,params,retain_graph=True,allow_unused=True)
            gu = torch.autograd.grad(self.pending_gradient_loss,params,retain_graph=True,allow_unused=True)
            self.observer.labeled_gradients = tuple(None if g is None else g.detach().clone() for g in gl)
            self.observer.raw_gradients = tuple(None if a is None and b is None else
                ((torch.zeros_like(p) if a is None else a)+(torch.zeros_like(p) if b is None else b)).detach().clone()
                for a,b,p in zip(gl,gu,params))
        return super().guide(labeled,student)


@contextmanager
def installed(c,holder):
    old_cls, old_reference = priority.PriorityREPL, priority.reference_loss
    recipe = c['tuning_recipe']
    def reference(labeled,unlabeled,ref,named,*,cap):
        if cap is None:
            return old_reference(labeled,unlabeled,ref,named,cap=None)
        parts = holder['objective'].parts
        other = sum(parts[k] for k in ('shallow','pair_kl','joint','extrap_delta') if k != recipe['reference'])
        return auxiliary_loss(labeled,unlabeled,ref,other,named,alpha=recipe['alpha'],beta=recipe['beta'])
    priority.PriorityREPL, priority.reference_loss = SoftREPL, reference
    try:
        with priority.installed(c,holder):
            yield
    finally:
        priority.PriorityREPL, priority.reference_loss = old_cls, old_reference
