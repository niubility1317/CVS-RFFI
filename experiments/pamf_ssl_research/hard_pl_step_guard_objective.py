"""Keep the complete high-weight supervised-U recipe and guard its actual extra step."""
from pathlib import Path
from .hard_pl_branch_guard_objective import BranchGuardedNativeAs
from .optimizer_step_guard import OptimizerStepGuard


class StepGuardedNativeAs(BranchGuardedNativeAs):
    def __init__(self,c):
        super().__init__(c)
        destination=None if c.get('hard_pl_smoke') else Path(c['output_root'])/'step_guard_metrics.jsonl.gz'
        self.parameter_guard=OptimizerStepGuard(destination)

    def guide(self,labeled,student):
        self.step_student=student
        return super().guide(labeled,student)

    def capture_step_anchors(self,named,full,clean,physical):
        self.parameter_guard.capture(self.step_student,named,full,clean,physical)

    def state_dict(self):
        state=super().state_dict()
        state['U_parameter_step_guard']=dict(self.parameter_guard.state_dict(),recipe=self.c['hard_pl_step_guard'])
        return state
