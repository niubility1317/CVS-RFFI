"""Original Tri guard plus full-L protection of actual AdamW extra motion."""
from contextlib import contextmanager
from pathlib import Path
from experiments.pamf_repl_weight.pseudo_observation import PseudoObservationMixin
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from experiments.pamf_ssl_research.source_branch_gradients import native_reference_gradients
from experiments.pamf_ssl_research.optimizer_step_guard import OptimizerStepGuard


class StepProtectedREPL(PseudoObservationMixin,GuardedReceiverExtrapolation):
    def __init__(self,c):
        super().__init__(c)
        destination=None if c.get('hard_pl_smoke') else Path(c['output_root'])/'step_guard_metrics.jsonl.gz'
        self.parameter_guard=OptimizerStepGuard(destination)

    def guide(self,labeled,student):
        named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
        with native_reference_gradients() as gradients:
            value=super().guide(labeled,student)
        if len(gradients)!=2:raise ValueError('Exact original full L/U gradients required')
        empty=tuple(None for _ in named)
        # Empty extra anchors disable branch-specific constraints; the original
        # complete labeled objective is the sole step-space anchor.
        self.parameter_guard.capture(student,named,gradients[0],empty,empty)
        return value

    def state_dict(self):
        result=super().state_dict()
        result['U_parameter_step_guard']=self.parameter_guard.state_dict()
        return result


@contextmanager
def installed(c,holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from experiments.pamf_ssl_research.optimizer_step_guard import installed as optimizer_scope
    from .plan import validate
    with paper_scope(c,holder,ssl_enabled=True,objective_factory=StepProtectedREPL,validate=validate):
        with optimizer_scope():
            try:yield
            finally:
                objective=holder.get('objective')
                if objective is not None:
                    observations=objective.close_pseudo_observation()
                    if not c.get('hard_pl_smoke') and (observations['calls']!=objective.calls
                            or observations['seen']!=objective.seen or observations['accepted']!=objective.accepted):
                        raise ValueError('Pseudo observation counts differ from actual U execution')
