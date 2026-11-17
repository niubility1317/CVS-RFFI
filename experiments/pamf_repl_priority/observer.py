"""Observe sampled same-state AdamW reference motion without changing live parameters."""
import gzip
import json
from pathlib import Path
import torch
from experiments.pamf_ssl_research import optimizer_step_guard as native
from experiments.pamf_ssl_research.u_gradprotect_objective import parameter_group


class ReferenceObserver(native.OptimizerStepGuard):
    def __init__(self, destination):
        super().__init__(destination)
        self.before = None
        self.source_step = None

    def _reference_step(self, model, named, optimizer, gradients, *, clip_impl):
        self.before = tuple(p.detach().clone() for _,p in named)
        return super()._reference_step(model, named, optimizer, gradients, clip_impl=clip_impl)

    @torch.no_grad()
    def after_step(self, optimizer, args, kwargs):
        if self.pending is None or self.reference is None or optimizer is not self.optimizer or self.before is None:
            raise ValueError('Sampled actual update and reference must share the same optimizer state')
        _, named, _ = self.pending
        groups = {}
        for group in ('t2', 'cls_head', 'other'):
            idx = [i for i,(name,_) in enumerate(named) if parameter_group(name)==group]
            zero = torch.zeros((), device=named[0][1].device, dtype=torch.float64)
            ref = [(self.reference[i]-self.before[i]).double() for i in idx]
            actual = [(named[i][1].detach()-self.before[i]).double() for i in idx]
            rr=sum((v.square().sum() for v in ref),zero); aa=sum((v.square().sum() for v in actual),zero)
            dot=sum(((a*b).sum() for a,b in zip(ref,actual)),zero)
            if not bool(torch.isfinite(rr) & torch.isfinite(aa) & torch.isfinite(dot)):
                raise ValueError('Finite observed actual and reference updates required')
            groups[group] = dict(reference_norm=float(rr.sqrt()), actual_norm=float(aa.sqrt()), dot=float(dot),
                cosine=float(dot/(rr*aa).sqrt()) if rr>0 and aa>0 else None)
        self.calls += 1; self.last_details = groups
        if self.destination is not None:
            if self.stream is None:
                self.stream=gzip.open(Path(self.destination),'xt',encoding='utf-8',compresslevel=3)
            self.stream.write(json.dumps(dict(source_step=self.source_step,groups=groups,
                copied_existing_moments=True,live_parameters_modified=False),allow_nan=False)+'\n')
            self.stream.flush()
        # This is a sampled observer, not a per-step guard.  Leave unsampled
        # optimizer steps untouched and register a fresh hook at the next sample.
        if self.hook is None:
            raise ValueError('Sampled observer hook is missing')
        self.hook.remove(); self.hook=None
        self.pending=None; self.reference=None; self.before=None; self.source_step=None
        scope=native._SCOPE.get()
        if scope is None or scope['pending'] is not self:
            raise ValueError('Observer escaped native optimizer scope')
        scope['pending']=None
