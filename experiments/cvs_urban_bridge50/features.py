"""Actual deployed joint and raw SIG112, with bounded hook lifetimes."""
import torch
from experiments.cvs_urban_branch50 import branches as b
from experiments.cvs_urban_bridge import signature


def forward(model,x):
    with signature.capture(model) as raw:out,values=b.forward(model,x)
    values['D']=raw['descriptor']
    return out,values


class NativeCapture(b.NativeCapture):
    def __init__(self,model,x):
        self.raw_scope=signature.capture(model);self.raw=self.raw_scope.__enter__();self.raw_closed=False
        try:super().__init__(model,x)
        except Exception:
            self.close_raw();raise

    def close_raw(self):
        if not self.raw_closed:
            self.raw_scope.__exit__(None,None,None);self.raw_closed=True

    def close(self):
        super().close();self.close_raw()

    def result(self):
        if not self.done or not self.closed or not self.raw_closed or 'descriptor' not in self.raw:
            raise ValueError('Actual complete labeled native capture required')
        return dict(self.values,D=self.raw['descriptor'])


def frozen_teacher(model,x):
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Own EMA must stay frozen')
    modes={m:m.training for m in model.modules()};model.eval()
    try:
        with torch.no_grad():out,values=forward(model,x)
    finally:
        for module,mode in modes.items():module.training=mode
    return out['tx_logits'].detach(),{k:values[k].detach() for k in (*b.BRANCHES,'J','D')}
