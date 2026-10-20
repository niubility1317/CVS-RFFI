from torch import nn
from . import design as d
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_r17_compact.models import stages
from experiments.pamf_r17_fidelity.models import FidelityRestorer,distillation_loss
from experiments.pamf_r17_fidelity.models import compact_from_teacher as shared_from_teacher
from experiments.pamf_r17_calibrated.models import CalibratedRestorer
from experiments.pamf_r17_calibrated.models import compact_from_teacher as activated_from_teacher


class TaskRestorer(nn.Module):
    def __init__(self,arm,frozen=None,inner=None):
        super().__init__()
        if arm not in d.ARMS:raise ValueError('Registered R17 task-fidelity arm required')
        self.arm=arm;self.calibration_report=None
        if inner is None:
            frozen=ActionRestorer(ComplexTFUNet()) if frozen is None else frozen
            inner=FidelityRestorer('step_norm_fidelity',frozen) if arm=='step_norm_task' else CalibratedRestorer('activation_half',frozen)
        self.inner=inner

    @property
    def prior(self):return self.inner.prior

    @property
    def frozen(self):return self.inner.frozen

    def stage_outputs(self,received,context=None):return self.inner.stage_outputs(received,context)

    def training_outputs(self,received):return self.inner.training_outputs(received)

    def forward(self,received):return self.inner(received)


def compact_from_teacher(teacher,arm,moments):
    if arm=='step_norm_task':inner=shared_from_teacher(teacher,'step_norm_fidelity');reports=[]
    elif arm=='activation_task':inner,reports=activated_from_teacher(teacher,'activation_half',moments)
    else:raise ValueError('Registered R17 task-fidelity arm required')
    return TaskRestorer(arm,inner=inner),reports
