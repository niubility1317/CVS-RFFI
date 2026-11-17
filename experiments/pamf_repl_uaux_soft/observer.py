"""Observe same-state L-only, raw joint and actual AdamW updates."""
import gzip
import json
from pathlib import Path
import torch
from experiments.pamf_repl_priority.observer import ReferenceObserver
from experiments.pamf_ssl_research.optimizer_reference import labeled_step
from experiments.pamf_ssl_research.u_gradprotect_objective import parameter_group


class DualObserver(ReferenceObserver):
    def __init__(self, destination, extra):
        super().__init__(destination)
        self.extra = extra
        self.extra_stream = None
        self.labeled_gradients = self.raw_gradients = None
        self.labeled_reference = self.raw_reference = None

    def _reference_step(self, model, named, optimizer, gradients, *, clip_impl):
        if self.labeled_gradients is None or self.raw_gradients is None:
            raise ValueError('L and raw joint gradients required for AdamW diagnosis')
        self.labeled_reference, _ = labeled_step(model, named, optimizer, self.labeled_gradients, clip_impl=clip_impl)
        self.raw_reference, _ = labeled_step(model, named, optimizer, self.raw_gradients, clip_impl=clip_impl)
        return super()._reference_step(model, named, optimizer, gradients, clip_impl=clip_impl)

    @torch.no_grad()
    def after_step(self, optimizer, args, kwargs):
        if self.pending is None or self.labeled_reference is None or self.raw_reference is None:
            raise ValueError('Missing same-state optimizer comparison')
        _, named, _ = self.pending
        groups = {}
        for group in ('t2', 'cls_head', 'other'):
            ids = [i for i,(name,_) in enumerate(named) if parameter_group(name) == group]
            zero = torch.zeros((), device=named[0][1].device, dtype=torch.float64)
            l = [torch.zeros_like(named[i][1],dtype=torch.float64) if self.labeled_gradients[i] is None
                 else self.labeled_gradients[i].double() for i in ids]
            before = [(self.raw_reference[i]-self.labeled_reference[i]).double() for i in ids]
            after = [(named[i][1].detach()-self.labeled_reference[i]).double() for i in ids]
            dot = lambda a,b: sum(((x*y).sum() for x,y in zip(a,b)),zero)
            ll,bb,aa = dot(l,l),dot(before,before),dot(after,after)
            lb,la = dot(l,before),dot(l,after)
            if not bool(torch.isfinite(ll+bb+aa+lb+la)):
                raise ValueError('Finite same-state AdamW comparison required')
            groups[group] = dict(labeled_gradient_norm=float(ll.sqrt()), raw_extra_update_norm=float(bb.sqrt()),
                actual_extra_update_norm=float(aa.sqrt()), labeled_dot_raw_update=float(lb),
                labeled_dot_actual_update=float(la),
                labeled_cos_raw_update=float(lb/(ll*bb).sqrt()) if ll>0 and bb>0 else None,
                labeled_cos_actual_update=float(la/(ll*aa).sqrt()) if ll>0 and aa>0 else None)
        if self.extra is not None:
            if self.extra_stream is None:
                self.extra_stream = gzip.open(Path(self.extra),'xt',encoding='utf-8',compresslevel=3)
            self.extra_stream.write(json.dumps(dict(source_step=self.source_step,groups=groups,
                positive_dot_opposes_labeled_descent=True,copied_existing_moments=True),allow_nan=False)+'\n')
            self.extra_stream.flush()
        super().after_step(optimizer,args,kwargs)
        self.labeled_gradients = self.raw_gradients = None
        self.labeled_reference = self.raw_reference = None

    def close(self):
        super().close()
        if self.extra_stream is not None:
            self.extra_stream.close(); self.extra_stream = None
