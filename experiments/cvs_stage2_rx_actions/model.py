import torch
from torch import nn
from . import design as d
from experiments.cvs_report_ssl.model import ReportIdentity,groups,Teacher
from experiments.cvs_stage2_decouple.common import build_model

class OriginalIdentity(ReportIdentity):
    def __init__(self,base):
        nn.Module.__init__(self);self.base=base;self._stage2_baseline_args=base._stage2_baseline_args
        self.nonlinear=None;self.nonlinear_gain=None


def build(c,device,checkpoint=None):
    d.validate(c);base=build_model(c['parent_contract'],device)
    model=(OriginalIdentity(base) if c['reference_mode']=='original' else ReportIdentity(base,False)).to(device)
    if checkpoint is not None:
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        if ck['config']!=c or ck['frontend_version']!=c['frontend']['version']:raise ValueError('Teacher/reference checkpoint semantics differ')
        model.load_state_dict(ck['model'],strict=True)
    return model


class CurrentStudentTarget:
    """No copy, no EMA, no parameter updates; deterministic detached weak view."""
    def __init__(self,student):self.model=student;self.calls=0;self.updates=0;self.is_ema=False

    @torch.no_grad()
    def probabilities(self,x):
        self.calls+=1
        modes=[(module,module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            return self.model(x).softmax(1).detach()
        finally:
            for module,mode in modes:module.training=mode

    def update(self,student):
        if student is not self.model:raise ValueError('Live target bound to wrong student')


def make_target(c,model):
    provider=c['semi']['target_provider']
    if provider=='EMA':return Teacher(model,.99)
    if provider=='current_student':return CurrentStudentTarget(model)
    raise ValueError('No target should be constructed for pure CE')
